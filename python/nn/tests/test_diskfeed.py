"""The trainer's rows on disk (``fortyone_nn.train``: ``--feed disk``, class ``DiskRows``, the disk
``Feed`` and the choice of ``--feed auto``).

- ``DiskRows`` built a block at a time on threads holds exactly the in-memory compact columns
  (and ``with_weights``' weight column), and gathers any rows in any order, across files;
- the disk feed gives exactly the former batches (the same tensors, in order, over two epochs,
  through ``feed(b)`` and through the CUDA-graph step's ``put``/``read``), with prefetching;
- training with ``--feed disk`` gives the network of ``resident`` and ``staged``, byte for byte
  (the float file and the integer network's SHA-256), with row weights too, streamed shards too
  (straight to disk, and moved to disk mid-training by ``auto``), and with the CUDA-graph step's
  bookkeeping on the CPU (``train.Replayed``);
- in disk mode the rows are never all in memory: ``load_compact`` is never called, and a fresh
  process's anonymous memory while building and reading 5 times the rows grows by far less than
  the rows;
- ``--feed auto`` chooses disk under a faked ``MemAvailable`` (and a cgroup limit) that the rows
  and the reserve do not fit, memory otherwise, and the report records the choice.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # the numpy-only CI job
    raise unittest.SkipTest("PyTorch is not installed") from None

from fortyone_nn import train
from fortyone_nn.formats import load_shard, sha256_file

from test_train import drop, set_row_weights, synthetic_shard

GIB = 1 << 30


def fake_host(available, total=64 * GIB):
    """``host_memory`` with ``available`` bytes of ``total``."""
    return {"mem_available": available, "mem_total": total, "cgroup_limit": None, "cgroup_anon": None,
            "available": available, "source": "MemAvailable"}


class DiskRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.train = [synthetic_shard(os.path.join(d, "a.npy"), 700, 1), synthetic_shard(os.path.join(d, "b.npy"), 300, 2)]
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 200, 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def sizes(self, weighted):
        if not weighted:
            return [(700, 1.0, None), (300, 1.0, None)]
        rw = np.full(300, 256, dtype="<u2")
        rw[::3] = 640
        return [(700, 2.0, None), (300, 1.5, rw)]

    def disk(self, sizes, threads=3, block=64):
        shards = [load_shard(p) for p in self.train]
        specs, _ = train.compact_specs(shards[0])
        store = train.DiskRows(self.dir.name, specs)
        for sh, sz in zip(shards, sizes):
            store.join([store.build(sh, [sz], threads=threads, block=block)])
        return store

    def test_the_files_hold_the_compact_columns(self):
        for w in (False, True):
            sizes = self.sizes(w)
            want = train.with_weights(train.load_compact([load_shard(p) for p in self.train]), sizes)
            store = self.disk(sizes)
            try:
                self.assertEqual(store.rows, 1000)
                self.assertEqual(len(store.files()), 2)
                self.assertEqual(store.weighted(), w)
                self.assertEqual(store.compact_bytes(), sum(c.nbytes for c in want))
                names = [f"c{k}" for k in range(6)] + (["w"] if w else [])
                rng = np.random.default_rng(3)
                for idx in (np.arange(1000), rng.permutation(1000)[:333], np.array([999, 0, 700, 699, 5]),
                            np.array([], dtype=np.int64)):
                    outs = [np.zeros((len(idx),) + c.shape[1:], dtype=c.dtype) for c in want]
                    n = train.DiskRows.gather(store.segments, store.dtype, idx.astype(np.int64), outs, names)
                    self.assertEqual(n, len(idx))
                    for o, c in zip(outs, want):
                        self.assertTrue(np.array_equal(o, c[idx]), (w, len(idx)))
                # The same files from arrays (a streamed training's rows moving to disk).
                seg = store.build_arrays(want[:6], sizes, block=100)
                other = train.DiskRows(self.dir.name, store.specs)
                other.join([seg])
                self.assertEqual(np.load(seg["path"], mmap_mode="r").tobytes(),
                                 np.concatenate([np.load(p, mmap_mode="r") for p in store.files()]).tobytes())
                other.close()
            finally:
                d = store.dir
                store.close()
                self.assertFalse(os.path.exists(d), "close removes the files")

    def test_padding_keeps_rows_within_pages(self):
        specs, row = train.compact_specs(load_shard(self.train[0]))  # trick 4: 847 bytes + 4
        self.assertEqual(train.record_dtype(specs).itemsize, row + 4, "no padding worth an eighth of a row")
        t3 = [(np.dtype(np.int16), (96,)), (np.dtype(np.float32), (100,)), (np.dtype(np.float32), (4,)),
              (np.dtype(np.uint8), ()), (np.dtype(bool), (24,)), (np.dtype(np.float32), (24, 4))]
        self.assertEqual(train.record_dtype(t3).itemsize, 1024, "trick 3: 1,021 bytes padded to 1,024")

    def test_the_disk_feed_gives_the_former_batches(self):
        whole = train.tensors(np.concatenate([np.asarray(load_shard(p)) for p in self.train]))
        n = 1000
        cpu = torch.device("cpu")
        for w in (False, True):
            sizes = self.sizes(w)
            wcol = torch.from_numpy(train.weight_block(sizes, 0, n)) if w else None
            store = self.disk(sizes)
            for batch, threads, depth in ((64, 1, 1), (100, 4, 3), (64, 2, 40)):
                feed = train.Feed(store, cpu, batch, "disk", threads, depth)
                self.assertEqual(feed.facts["mode"], "disk")
                self.assertEqual(feed.facts["files"], store.files())
                bufs = feed.inputs(batch)
                rng = np.random.default_rng(5)
                for epoch in range(2):
                    perm = torch.from_numpy(rng.permutation(n))
                    order = feed.order(perm)
                    for i in range(0, n, batch):
                        b_old = perm[i : i + batch]
                        old = [c[b_old] for c in whole] + ([wcol[b_old]] if w else [])
                        b = order[i : i + batch]
                        self.assertTrue(torch.equal(b, b_old))
                        if epoch == 1 and len(b) == batch and i // batch % 3 == 0:
                            feed.put(b, bufs)  # the CUDA-graph step's way, in turn with feed(b)
                            got = feed.read(bufs)
                        else:
                            got = feed(b)
                        self.assertEqual(len(got), len(old))
                        for x, y in zip(old, got):
                            self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape))
                            self.assertTrue(torch.equal(x, y), (w, batch, i))
                # A batch that is not the permutation's next slice is gathered at once.
                b = torch.tensor([3, 999, 0])
                for x, y in zip([c[b] for c in whole], feed(b)):
                    self.assertTrue(torch.equal(x, y))
                feed.close()
            store.close()
        with self.assertRaises(ValueError):
            train.Feed(train.load_compact([load_shard(self.train[0])]), cpu, 64, "disk")


class DiskTrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.rows = os.path.join(d, "rows")
        cls.train = [synthetic_shard(os.path.join(d, "a.npy"), 700, 1), synthetic_shard(os.path.join(d, "b.npy"), 300, 2)]
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 200, 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_train(self, name, *extra, train_files=None):
        out = os.path.join(self.dir.name, name)
        files = self.train if train_files is None else train_files
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["--train", *files, "--val", self.val, "--out", out, "--widths", "32,16", "--epochs", "3",
                        "--batch", "64", "--seed", "5", "--threads", "1", "--device", "cpu", "--rows-dir", self.rows,
                        "--prefetch-threads", "3", "--prefetch-depth", "4", *extra])
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
        return (sha256_file(os.path.join(out, "float.safetensors")), sha256_file(os.path.join(out, "int.safetensors"))), rep

    def test_disk_trains_the_network_of_resident_and_staged(self):
        r, rep_r = self.run_train("resident", "--feed", "resident")
        s, _ = self.run_train("staged", "--feed", "staged")
        with mock.patch.object(train, "load_compact", side_effect=AssertionError("rows loaded into memory")):
            d, rep_d = self.run_train("disk", "--feed", "disk")
        self.assertEqual(s, r)
        self.assertEqual(d, r, "the float file and the integer network, byte for byte")
        self.assertEqual(rep_r["device"]["feed"]["mode"], "resident")
        f = rep_d["device"]["feed"]
        self.assertEqual((f["mode"], f["rows"], f["prefetch_threads"], f["prefetch_depth"]), ("disk", 1000, 3, 4))
        self.assertEqual(f["bytes"], rep_r["device"]["feed"]["bytes"])
        self.assertEqual(len(f["files"]), 2)
        self.assertEqual(f["host"]["rows"], "disk")
        self.assertFalse(os.path.exists(f["rows_dir"]), "the files are removed at the end")
        self.assertEqual(os.listdir(self.rows), [])
        self.assertEqual([h["val_loss"] for h in rep_d["history"]], [h["val_loss"] for h in rep_r["history"]])

    def test_disk_with_weights_and_the_graph_bookkeeping(self):
        d = os.path.join(self.dir.name, "weighted")
        os.makedirs(d)
        files = [synthetic_shard(os.path.join(d, "a.npy"), 700, 1), synthetic_shard(os.path.join(d, "b.npy"), 300, 2)]
        with open(files[0] + ".weight", "w") as f:
            f.write("2")
        rw = np.full(300, 256, dtype="<u2")
        rw[1::2] = 768
        set_row_weights(files[1], rw)
        r, _ = self.run_train("w-resident", "--feed", "resident", train_files=files)
        d1, rep = self.run_train("w-disk", "--feed", "disk", train_files=files)
        self.assertEqual(d1, r)
        self.assertEqual(rep["data"]["train_weights"], [2.0, 1.0])
        with mock.patch.object(train, "GRAPH_DEVICES", ("cpu",)), mock.patch.object(train, "GRAPH_CLASS", train.Replayed):
            g_r, _ = self.run_train("g-resident", "--feed", "resident", train_files=files)
            g_d, rep_g = self.run_train("g-disk", "--feed", "disk", train_files=files)
        self.assertEqual(g_d, g_r, "replayed steps read the disk feed's buffers as the resident ones")
        self.assertEqual(rep_g["device"]["train_step"]["mode"], "cuda-graph")
        self.assertGreater(rep_g["device"]["train_step"]["replays"], 0)

    def test_auto_chooses_by_the_host_memory(self):
        need = 1000 * (train.compact_specs(load_shard(self.train[0]))[1])
        reserve = 2 * GIB
        with mock.patch.object(train, "host_memory", return_value=fake_host(reserve + need - 1)):
            a, rep = self.run_train("auto-tight", "--host-reserve-gb", "2")
        self.assertEqual(rep["device"]["feed"]["mode"], "disk")
        h = rep["device"]["feed"]["host"]
        self.assertEqual((h["need_bytes"], h["reserve_bytes"], h["available_bytes"], h["rows"]),
                         (need, reserve, reserve + need - 1, "disk"))
        with mock.patch.object(train, "host_memory", return_value=fake_host(reserve + need)):
            b, rep = self.run_train("auto-fits", "--host-reserve-gb", "2")
        self.assertEqual(rep["device"]["feed"]["mode"], "resident", "the CPU, as before")
        self.assertEqual(rep["device"]["feed"]["host"]["rows"], "memory")
        self.assertEqual(a, b)
        # Other work's planned memory counts against the machine's memory, not what is free now.
        with mock.patch.object(train, "host_memory", return_value=fake_host(64 * GIB, 64 * GIB)):
            _, rep = self.run_train("auto-others", "--host-reserve-gb", "2", "--host-others-gb", "62")
        self.assertEqual(rep["device"]["feed"]["mode"], "disk")
        self.assertEqual(rep["device"]["feed"]["host"]["others_bytes"], 62 * GIB)
        with mock.patch.object(train, "host_memory", return_value=fake_host(64 * GIB, 64 * GIB + need)):
            _, rep = self.run_train("auto-others-fit", "--host-reserve-gb", "2", "--host-others-gb", "62")
        self.assertEqual(rep["device"]["feed"]["mode"], "resident")
        # An unknown host (nothing readable) keeps the rows in memory, as before.
        with mock.patch.object(train, "host_memory", return_value=fake_host(None, None)):
            _, rep = self.run_train("auto-unknown")
        self.assertEqual(rep["device"]["feed"]["mode"], "resident")
        # resident and staged are kept whatever the memory.
        with mock.patch.object(train, "host_memory", return_value=fake_host(0)):
            _, rep = self.run_train("staged-tight", "--feed", "staged")
        self.assertEqual(rep["device"]["feed"]["mode"], "staged")


class HostMemoryTest(unittest.TestCase):
    def write(self, root, files):
        for rel, text in files.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(text)

    def test_meminfo_and_cgroups(self):
        mi = "MemTotal:       67108864 kB\nMemFree:         1000000 kB\nMemAvailable:   50000000 kB\n"
        with tempfile.TemporaryDirectory() as root:
            self.write(root, {"proc/meminfo": mi})
            h = train.host_memory(root)
            self.assertEqual((h["available"], h["source"]), (50000000 * 1024, "MemAvailable"))
            self.write(root, {"sys/fs/cgroup/memory.max": "max\n"})
            self.assertEqual(train.host_memory(root)["available"], 50000000 * 1024, "no limit")
            self.write(root, {"sys/fs/cgroup/memory.max": f"{32 * GIB}\n",
                              "sys/fs/cgroup/memory.stat": f"anon {10 * GIB}\nfile {5 * GIB}\n"})
            h = train.host_memory(root)
            self.assertEqual((h["available"], h["cgroup_limit"], h["cgroup_anon"]), (22 * GIB, 32 * GIB, 10 * GIB))
        with tempfile.TemporaryDirectory() as root:
            self.write(root, {"proc/meminfo": mi, "sys/fs/cgroup/memory/memory.limit_in_bytes": f"{16 * GIB}\n",
                              "sys/fs/cgroup/memory/memory.stat": f"cache 1\ntotal_rss {4 * GIB}\n"})
            self.assertEqual(train.host_memory(root)["available"], 12 * GIB)
            # A v1 limit above MemTotal (no limit) leaves MemAvailable.
            self.write(root, {"sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n"})
            self.assertEqual(train.host_memory(root)["available"], 50000000 * 1024)
        # The process's own group (``/proc/self/cgroup``) and those above it: the tightest.
        with tempfile.TemporaryDirectory() as root:
            self.write(root, {"proc/meminfo": mi, "proc/self/cgroup": "5:cpu:/\n4:memory:/a/b\n0::/\n",
                              "sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
                              "sys/fs/cgroup/memory/a/memory.limit_in_bytes": f"{20 * GIB}\n",
                              "sys/fs/cgroup/memory/a/memory.stat": f"total_rss {GIB}\n",
                              "sys/fs/cgroup/memory/a/b/memory.limit_in_bytes": f"{8 * GIB}\n",
                              "sys/fs/cgroup/memory/a/b/memory.stat": f"total_rss {2 * GIB}\n"})
            h = train.host_memory(root)
            self.assertEqual((h["available"], h["cgroup_limit"]), (6 * GIB, 8 * GIB))
            self.write(root, {"proc/self/cgroup": "0::/x\n", "sys/fs/cgroup/x/memory.max": f"{4 * GIB}\n",
                              "sys/fs/cgroup/x/memory.stat": "anon 0\n"})
            self.assertEqual(train.host_memory(root)["available"], 4 * GIB, "v2 first")
        with tempfile.TemporaryDirectory() as root:
            h = train.host_memory(root)
            self.assertIsNone(h["available"])
            self.assertTrue(train.fits_host(10**15, 0, h), "unknown memory: as before")
        self.assertFalse(train.fits_host(GIB + 1, GIB, fake_host(2 * GIB)))
        self.assertTrue(train.fits_host(GIB, GIB, fake_host(2 * GIB)))
        self.assertFalse(train.fits_host(GIB, GIB, fake_host(2 * GIB, 10 * GIB), 8 * GIB + 1))
        self.assertTrue(train.fits_host(GIB, GIB, fake_host(2 * GIB, 10 * GIB), 8 * GIB))
        self.assertEqual(train.host_total({"mem_total": 10, "cgroup_limit": 4}), 4)


class DiskStreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.src = os.path.join(d, "src")
        os.makedirs(cls.src)
        cls.shards = [synthetic_shard(os.path.join(cls.src, n), r, s)
                      for n, r, s in (("a.npy", 700, 1), ("b.npy", 300, 2), ("c.npy", 500, 4))]
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 200, 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def streamed(self, name, *extra, host=None):
        """Shards a, b, c dropped one after each partial epoch (``--partial-epochs 1``)."""
        w = os.path.join(self.dir.name, name + "-watch")
        os.makedirs(w)
        drop(self.shards[0], w)
        out = os.path.join(self.dir.name, name)
        prog = out + ".progress.json"

        def feeder():
            for k in (1, 2):
                while True:
                    try:
                        with open(prog) as f:
                            p = json.load(f)
                        if p.get("step") == "wait" and p.get("shards_in") == k and "epoch" in p:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.02)
                drop(self.shards[k], w)

        t = threading.Thread(target=feeder, daemon=True)
        t.start()
        with contextlib.ExitStack() as st:
            if host is not None:
                st.enter_context(mock.patch.object(train, "host_memory", side_effect=host))
            st.enter_context(contextlib.redirect_stdout(io.StringIO()))
            train.main(["--watch-dir", w, "--expect", "3", "--partial-epochs", "1", "--val", self.val, "--out", out,
                        "--widths", "16,8", "--epochs", "2", "--batch", "64", "--seed", "5", "--threads", "1",
                        "--device", "cpu", "--progress", prog, "--watch-poll", "0.05", "--rows-dir",
                        os.path.join(self.dir.name, "rows"), "--prefetch-depth", "3", *extra])
        t.join(timeout=5)
        with open(os.path.join(out, "report.json")) as f:
            return sha256_file(os.path.join(out, "int.safetensors")), json.load(f)

    def test_streamed_shards_on_disk_train_the_same_network(self):
        a, rep_a = self.streamed("mem", "--feed", "resident")
        b, rep_b = self.streamed("disk", "--feed", "disk")
        self.assertEqual(b, a)
        self.assertEqual(rep_b["device"]["feed"]["mode"], "disk")
        self.assertEqual(len(rep_b["device"]["feed"]["files"]), 3)
        self.assertEqual([p["first_epoch"] for p in rep_b["stream"]["parts"]], [0, 1, 2])
        self.assertEqual(rep_b["stream"]["joins_in_place"], 3)
        self.assertEqual(rep_b["stream"]["rows_to_disk"]["rows"], 0)
        # auto: a, then b fit in memory; c does not (the faked memory falls): a and b go to disk
        # at c's join, with c.
        calls = []

        def host():
            calls.append(1)
            return fake_host(None if len(calls) < 3 else 0)

        c, rep_c = self.streamed("auto", host=host)
        self.assertEqual(c, a)
        self.assertEqual(rep_c["stream"]["rows_to_disk"]["rows"], 1000)
        self.assertEqual(rep_c["stream"]["rows_to_disk"]["epoch"], 2)
        self.assertEqual(rep_c["device"]["feed"]["mode"], "disk")
        self.assertEqual(len(rep_c["device"]["feed"]["files"]), 2, "the rows in memory as one file, then c")
        self.assertEqual(rep_c["stream"]["joins_in_place"], 3)
        self.assertEqual([h["rows"] for h in rep_c["history"]], [700, 1000, 1500, 1500])


class DiskMemoryTest(unittest.TestCase):
    def test_memory_does_not_grow_with_the_rows(self):
        """In a fresh process: build the disk rows of a shard and feed one epoch from them, sampling
        the process's anonymous memory (``RssAnon``); 5 times the rows (about 170 MB more compact
        rows) raise its peak by far less than the rows (measured 2026-10-19: 95 MB at 50,000 rows,
        124 MB at 250,000, mostly the build's block buffers, threads x ``LOAD_BLOCK`` rows)."""
        if not os.path.exists("/proc/self/status"):
            self.skipTest("needs /proc/self/status (Linux)")
        code = textwrap.dedent("""
            import json, os, sys, tempfile, threading, time
            import numpy as np, torch
            from fortyone_nn import train
            from fortyone_nn.formats import shard_dtype, PAD

            def anon():
                with open("/proc/self/status") as f:
                    return next(int(l.split()[1]) * 1024 for l in f if l.startswith("RssAnon"))

            rows, d = int(sys.argv[1]), sys.argv[2]
            path = os.path.join(d, f"s{rows}.npy")
            m = np.lib.format.open_memmap(path, mode="w+", dtype=shard_dtype(14), shape=(rows,))
            for j in range(0, rows, 20000):  # a shard written in blocks: never all in memory here
                b = np.zeros(min(20000, rows - j), dtype=m.dtype)
                b["active"] = PAD
                b["active"][:, 0] = np.arange(j, j + len(b)) % 809
                b["dense"][:, 0] = np.arange(j, j + len(b))
                m[j : j + len(b)] = b
            m.flush(); del m
            peak = [anon()]
            stop = threading.Event()
            def sample():
                while not stop.is_set():
                    peak[0] = max(peak[0], anon()); time.sleep(0.002)
            t = threading.Thread(target=sample, daemon=True); t.start()
            base = anon()
            sh = np.load(path, mmap_mode="r")
            specs, row = train.compact_specs(sh)
            store = train.DiskRows(d, specs)
            store.join([store.build(sh, [(rows, 1.0, None)], threads=2)])
            feed = train.Feed(store, torch.device("cpu"), 512, "disk", 4, 8)
            order = feed.order(torch.from_numpy(np.random.default_rng(1).permutation(rows)))
            ok = True
            for i in range(0, rows, 512):
                x = feed(order[i : i + 512])
                got = (x[1][:, 0].double() * (1 << 24)).long()
                ok = ok and torch.equal(got, order[i : i + 512])
            feed.close(); store.close(); stop.set(); t.join()
            print(json.dumps({"grew": peak[0] - base, "compact": rows * row, "ok": ok}))
        """)
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        got = {}
        with tempfile.TemporaryDirectory() as d:
            for rows in (50_000, 250_000):
                out = subprocess.run([sys.executable, "-c", code, str(rows), d], cwd=here, capture_output=True,
                                     text=True, check=True)
                got[rows] = json.loads(out.stdout.strip().splitlines()[-1])
                os.remove(os.path.join(d, f"s{rows}.npy"))
        for m in got.values():
            self.assertTrue(m["ok"], m)
        more = got[250_000]["compact"] - got[50_000]["compact"]
        self.assertGreater(more, 150_000_000)
        self.assertLess(got[250_000]["grew"] - got[50_000]["grew"], more // 4, got)
        self.assertLess(got[250_000]["grew"], got[250_000]["compact"], got)


if __name__ == "__main__":
    unittest.main()
