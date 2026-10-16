"""Which device trains the network, and the facts about it (docs/m3-hybrid/vast.md; PLAN §12.3
step 5, the GPU check).

    python -m fortyone_nn.device [--want auto|cuda|cpu] [--json OUT]

``probe(want)`` decides between the GPU and the CPU and says why:

- ``cpu``: the CPU, as asked;
- ``auto`` / ``cuda``: the GPU when every check passes: ``nvidia-smi`` runs, this PyTorch build
  has CUDA, ``torch.cuda.is_available()``, the GPU's compute capability is one this build was
  compiled for (``torch.cuda.get_arch_list()``: a ``sm_XY`` kernel runs on the same major
  version at minor Y or above; ``compute_XY`` code is compiled on the spot for XY or above), and
  one tiny training step on the GPU gives a finite loss. Otherwise the CPU, with the failed check
  as ``reason`` (``auto``), or an error (``cuda``: exit status 3 from the command line).

The facts recorded (``facts``): Python, PyTorch and its CUDA and cuDNN versions, every GPU
``nvidia-smi`` lists (name, memory, driver, compute capability) and the CUDA version the driver
supports, the chosen GPU's name, capability and memory as PyTorch sees it, the build's
architectures, and the deterministic settings (``deterministic``).

Determinism: ``CUBLAS_WORKSPACE_CONFIG`` is set (if unset) when this module is imported, before
any CUDA work, as cuBLAS requires for deterministic results; ``deterministic(device)`` sets
``torch.use_deterministic_algorithms(True, warn_only=True)``, cuDNN's deterministic mode and full
float32 matrix products (no TF32) on a GPU. A GPU's float training is still not bit-identical to
another GPU's or the CPU's (different kernels sum in different orders): the integer network is
what gets validated, and its SHA-256 identifies it (docs/M3-HYBRID-SPEC.md §6).
"""

import argparse
import json
import os
import re
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402  (after CUBLAS_WORKSPACE_CONFIG)


def nvidia_smi():
    """Every GPU ``nvidia-smi`` lists, and the CUDA version its driver supports; None when the
    command is missing or fails (no GPU in this container)."""
    q = "name,memory.total,driver_version,compute_cap"
    try:
        p = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    gpus = []
    for line in p.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 4:
            gpus.append({"name": f[0], "memory_mib": _num(f[1]), "driver": f[2], "compute_cap": f[3]})
    cuda = None
    try:
        banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
        m = re.search(r"CUDA Version:\s*([0-9.]+)", banner)
        cuda = m.group(1) if m else None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"gpus": gpus, "driver_cuda": cuda}


def _num(s):
    try:
        return int(float(s))
    except ValueError:
        return None


def arch_supported(cap, arch_list):
    """Whether a GPU of compute capability ``cap`` = (major, minor) can run a build compiled for
    ``arch_list`` (e.g. ``["sm_50", "sm_60", "sm_86", "compute_90"]``): a ``sm_XY`` kernel binary
    runs on major X at minor Y or above (an ``sm_XYa`` one only on exactly X.Y); ``compute_XY``
    (PTX) is compiled by the driver for X.Y or above."""
    major, minor = cap
    for a in arch_list:
        m = re.fullmatch(r"(sm|compute)_(\d+)(\d)([a-z]?)", a)
        if not m:
            continue
        kind, x, y, suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if kind == "sm" and x == major and (y == minor if suffix else y <= minor):
            return True
        if kind == "compute" and (x, y) <= (major, minor) and not suffix:
            return True
    return False


def tiny_step(device):
    """One training step of a small network on ``device``; raises if the loss is not finite."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 32, generator=g).to(device)
    y = torch.randint(0, 4, (64,), generator=g).to(device)
    net = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"the loss is {float(loss)}")


def deterministic(device):
    """Sets PyTorch's deterministic switches for ``device``; returns what was set."""
    out = {"use_deterministic_algorithms": "True, warn_only=True"}
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Full float32 products: TF32 would round the inputs to 10 bits of mantissa.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        out.update({
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn.deterministic": True, "cudnn.benchmark": False, "tf32": False,
            "float32_matmul_precision": "highest",
        })
    return out


def facts_base():
    cudnn = None
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001 (no cuDNN in a CPU build)
        cudnn = None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": cudnn,
        "cpu_threads_torch": torch.get_num_threads(),
    }


def probe(want="auto"):
    """The device to train on and the facts about it (see the module notes)."""
    if want not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device {want!r}: auto, cuda or cpu")
    f = facts_base()
    f["want"] = want
    f["nvidia_smi"] = nvidia_smi()
    if want == "cpu":
        f.update(device="cpu", usable_gpu=False, reason="the CPU was asked for")
        return f
    reason = None
    if f["nvidia_smi"] is None:
        reason = "nvidia-smi is missing or failed: no GPU in this container"
    elif not f["nvidia_smi"]["gpus"]:
        reason = "nvidia-smi lists no GPU"
    elif torch.version.cuda is None:
        reason = f"this PyTorch build ({torch.__version__}) has no CUDA"
    elif not torch.cuda.is_available():
        reason = (f"torch.cuda.is_available() is False (PyTorch CUDA {torch.version.cuda}; driver supports "
                  f"CUDA {f['nvidia_smi'].get('driver_cuda')}): the driver may be too old")
    else:
        cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        arch = torch.cuda.get_arch_list()
        f.update({
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "capability": f"{cap[0]}.{cap[1]}",
            "gpu_memory_mib": int(props.total_memory) >> 20,
            "arch_list": arch,
        })
        if not arch_supported(cap, arch):
            reason = f"compute capability {cap[0]}.{cap[1]} is not among this build's architectures {arch}"
        else:
            try:
                tiny_step(torch.device("cuda"))
            except Exception as e:  # noqa: BLE001 (any failure means: do not train on this GPU)
                reason = f"a tiny training step on the GPU failed: {type(e).__name__}: {str(e)[:300]}"
    if reason:
        f.update(device="cpu", usable_gpu=False, reason=reason)
    else:
        f.update(device="cuda", usable_gpu=True, reason="every GPU check passed")
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--want", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--json", help="write the facts here as well")
    args = ap.parse_args(argv)
    f = probe(args.want)
    text = json.dumps(f, indent=1, sort_keys=True)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    print(text)
    if args.want == "cuda" and f["device"] != "cuda":
        print(f"no usable GPU: {f['reason']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
