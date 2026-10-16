"""The width limit of the integer inference (``fortyone_nn.formats.MAX_WIDTH``) and the trainer's refusal.

A network with a layer wider than Rust's ``nn::mlp::MAX_WIDTH`` (4096; 1024 until 2026-10-16) cannot be
loaded by the integer inference, so training one (a trick-3 run trained 2048-1024-256 for 27 minutes and
then failed at the check, under the old limit) is wasted time. ``--widths`` is therefore checked before anything is loaded.

- ``parse_widths``: good lists, a width over the limit named by its layer (from 0) and the limit,
  the same words as Rust's ``width_error``, and malformed lists;
- parity with Rust: the constant equals the one in ``crates/nn/src/mlp.rs`` (read from the source,
  always) and ``nn-tool limits`` (when the binary is there, as ``test_intnet``), and the refusal text
  is the one ``nn-tool random --widths 8192,...`` prints.

numpy only (the CI's parity job): ``cd python/nn && uv run python -m unittest discover -s tests -v``.
"""

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from fortyone_nn.formats import MAX_WIDTH, parse_widths, width_error

from test_intnet import ROOT, nn_tool


class ParseWidths(unittest.TestCase):
    def test_good_lists(self):
        self.assertEqual(parse_widths("1024,512,128"), [1024, 512, 128])
        self.assertEqual(parse_widths("64, 32"), [64, 32])
        self.assertEqual(parse_widths("2048,1024,256"), [2048, 1024, 256])
        self.assertEqual(parse_widths("2048,2048,1024,256"), [2048, 2048, 1024, 256])
        self.assertEqual(parse_widths("4096"), [4096])  # exactly the limit passes
        self.assertEqual(MAX_WIDTH, 4096)

    def test_a_layer_over_the_limit_is_named(self):
        with self.assertRaises(ValueError) as e:
            parse_widths("8192,1024,256")
        msg = str(e.exception)
        for want in ("layer 0", "8192 units", "wider than 4096", "MAX_WIDTH", "crates/nn/src/mlp.rs"):
            self.assertIn(want, msg)
        self.assertEqual(msg, width_error(0, 8192))
        with self.assertRaises(ValueError) as e:  # the first layer over the limit, counted from 0
            parse_widths("512,4097,8192")
        self.assertEqual(str(e.exception), width_error(1, 4097))

    def test_malformed_lists(self):
        for text in ("", "1024,,128", "a,b", "512,0", "512,-4", "1.5", ","):
            with self.assertRaises(ValueError, msg=text):
                parse_widths(text)


class ParityWithRust(unittest.TestCase):
    def test_the_constant_is_the_one_in_the_rust_source(self):
        src = (ROOT / "crates" / "nn" / "src" / "mlp.rs").read_text()
        m = re.search(r"pub const MAX_WIDTH: usize = (\d+);", src)
        self.assertIsNotNone(m, "crates/nn/src/mlp.rs no longer defines MAX_WIDTH like this")
        self.assertEqual(MAX_WIDTH, int(m.group(1)))

    @unittest.skipIf(nn_tool() is None, "no nn-tool binary (cargo build --release -p nn)")
    def test_nn_tool_agrees(self):
        tool = nn_tool()
        out = subprocess.run([tool, "limits"], check=True, capture_output=True, text=True).stdout
        self.assertEqual(out.strip(), f"max_width={MAX_WIDTH}")
        with tempfile.TemporaryDirectory() as d:
            net = Path(d) / "net.safetensors"
            bad = subprocess.run(
                [tool, "random", "--trick", "4", "--widths", "8192,1024,256", "--out", str(net)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn(width_error(0, 8192), bad.stderr)  # Rust's refusal, word for word
            self.assertFalse(net.exists())


if __name__ == "__main__":
    unittest.main()
