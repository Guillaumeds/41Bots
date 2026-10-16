"""Which vast.ai offers the launcher may rent: the thresholds and checks, in one place.

launch.py builds its search query and checks a named offer (--offer-id) from these numbers and
functions, and the market tools (python/tools/vast_market.py, vast_watch/model.py, vast_status.py)
label an offer "the launcher can rent it" with the same ones, so the two cannot drift apart. Change a
rule here, not in a copy. Standard library only, no side effects on import.

The rules (owner, 2026-10-16, unless dated otherwise):
- reliability at least MIN_RELIABILITY (0.90);
- the whole machine (gpu_frac 1), on-demand, rentable;
- verified hosts are preferred, but unverified or deverified hosts are allowed (the launcher
  searches verified hosts first and only then the rest); `verified_only` turns that off;
- download speed (vast's `inet_down`, Mb/s) at least MIN_INET_DOWN (500) in every search, because the
  base image is several GB; a named offer with a slower link is reported, not refused (launch.py);
- network cost at most MAX_NET_COST ($0.10 per GB, up and down; owner 2026-10-14);
- at least one GPU (the search asks num_gpus>=1); a GPU job asks for exactly N, with
  compute_cap from MIN_COMPUTE_CAP (600, Pascal) up to below MAX_COMPUTE_CAP (1000, no Blackwell) and a
  driver that supports CUDA MIN_CUDA (12.6), the limits of the PyTorch 2.14.1+cu126 build. A CPU-only
  job (the default) leaves the compute-capability and CUDA limits out.
"""

from __future__ import annotations

MIN_RELIABILITY = 0.90  # owner, 2026-10-16: hosts with reliability of at least 90% are accepted, for every rental
MAX_NET_COST = 0.10  # $/GB up or down (owner 2026-10-14: 0.10; our traffic is tiny)
MIN_INET_DOWN = 500.0  # Mbps of download speed an offer must report (0 = no filter)
SLOW_LINK_MBPS = 200.0  # below this download speed the launcher's ranking adds a penalty (link_factor)
MIN_COMPUTE_CAP = 600  # vast's units: 600 = compute capability 6.0, the lowest the PyTorch 2.14.1+cu126 build supports
MAX_COMPUTE_CAP = 1000  # exclusive: no Blackwell (10.0, 12.0) in the cu126 build
MIN_CUDA = 12.6  # the driver must support CUDA 12.6


def is_verified(o: dict) -> bool:
    """Whether vast.ai lists the offer's host as verified (the offer's `verified` flag, or its
    `verification` word)."""
    return o.get("verified") is True or (o.get("verification") or "") == "verified"


def host_status(o: dict) -> str:
    """The host's verification word for the logs: verified, unverified, deverified, or unknown."""
    if is_verified(o):
        return "verified"
    word = o.get("verification")
    if isinstance(word, str) and word:
        return word
    return "unverified" if o.get("verified") is False else "unknown"


def host_flag(o: dict) -> str:
    """"" for a verified host, else its status word ("unverified", "deverified", "unknown"): what a
    label shows beside "the launcher can rent it"."""
    return "" if is_verified(o) else host_status(o)


def number(v) -> float | None:
    """A record's value as a float, or None when absent or not a number."""
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def inet_down_of(o: dict) -> float | None:
    """The offer's download speed in Mb/s as vast reports it (`inet_down`), or None when absent."""
    return number(o.get("inet_down"))


def link_factor(o: dict) -> float:
    """The ranking penalty for a slow download link: 1.0 at SLOW_LINK_MBPS (200) Mbps or more, and
    1 + (200 - Mbps) / 200 below it (x1.5 at 100 Mbps, x2.0 at 0). An offer that reports no speed
    gets no penalty (nothing is known). The base image is several GB, so a slow link means a long
    pull, which is paid for before any work starts. Not applied with --cheapest."""
    mbps = inet_down_of(o)
    if mbps is None or mbps >= SLOW_LINK_MBPS:
        return 1.0
    return 1.0 + (SLOW_LINK_MBPS - max(mbps, 0.0)) / SLOW_LINK_MBPS


def check_failures(o: dict, floor: float = MIN_RELIABILITY, verified_only: bool = False) -> list[str]:
    """Why an offer fails the launcher's safety checks (what the search query filters on, applied
    to one offer's record): reliability at least `floor` (0.90), the whole machine (no sharing),
    on-demand and rentable; with `verified_only` also a verified host (owner, 2026-10-16: unverified
    and deverified hosts are otherwise accepted). Empty when it passes. These are the checks a
    named offer (--offer-id) gets."""
    failed = []
    if verified_only and not is_verified(o):
        failed.append(f"host not verified ({o.get('verification') or o.get('verified')}), and --verified-only is set")
    rel = number(o.get("reliability2") if o.get("reliability2") is not None else o.get("reliability") or 0)
    if rel is None:
        rel = 0.0
    if rel < floor:
        failed.append(f"reliability {rel:.3f} below {floor}")
    frac = number(o.get("gpu_frac") if o.get("gpu_frac") is not None else 0)
    if frac is None:
        frac = 0.0
    if frac < 1.0:
        failed.append(f"shared machine (gpu_frac {o.get('gpu_frac')})")
    if o.get("is_bid"):
        failed.append("an interruptible (bid) offer, not on-demand")
    if o.get("rentable") is False:
        failed.append("not rentable")
    return failed


def gpu_failures(o: dict, gpus: int, min_gpu_ram_gb: float = 0, min_compute_cap: int = MIN_COMPUTE_CAP) -> list[str]:
    """Why an offer fails a GPU job's needs (exactly `gpus` GPUs, `min_gpu_ram_gb`, `min_compute_cap`, and the
    PyTorch build's limits), from vast's raw record (gpu_ram in MB, compute_cap x 100). With gpus 0 the
    GPU count is not checked (a job that wants the limits but no particular count)."""
    failed = []
    n, ram, cc, cuda = (number(o.get(k)) for k in ("num_gpus", "gpu_ram", "compute_cap", "cuda_max_good"))
    if gpus and (n is None or int(n) != int(gpus)):
        failed.append(f"{o.get('num_gpus')} GPUs, not the {gpus} asked for")
    if min_gpu_ram_gb and (ram is None or ram < min_gpu_ram_gb * 1000):
        failed.append(f"GPU memory {o.get('gpu_ram')} MB, below --min-gpu-ram-gb {min_gpu_ram_gb}")
    lo = max(MIN_COMPUTE_CAP, int(min_compute_cap or MIN_COMPUTE_CAP))
    if cc is None or not lo <= cc < MAX_COMPUTE_CAP:
        failed.append(f"compute capability {o.get('compute_cap')} outside {lo}..{MAX_COMPUTE_CAP - 1} "
                      f"(the PyTorch 2.14.1+cu126 build)")
    if cuda is None or cuda < MIN_CUDA:
        failed.append(f"driver supports CUDA {o.get('cuda_max_good')}, below {MIN_CUDA}")
    return failed


def search_failures(o: dict, *, min_inet_down: float | None = None, floor: float = MIN_RELIABILITY,
                    verified_only: bool = False, max_net_cost: float = MAX_NET_COST, gpus: int = 0,
                    cpu_only: bool = True, min_gpu_ram_gb: float = 0, min_compute_cap: int = MIN_COMPUTE_CAP) -> list[str]:
    """Why the launcher's SEARCH would not find an offer: `check_failures` plus the other filters of its
    query (download speed, network cost, at least one GPU, and for a GPU job or a job that is not
    CPU-only the compute-capability and CUDA limits). `min_inet_down` None means the default
    MIN_INET_DOWN, 0 means no filter. An offer that does not report a value the search filters on
    (download speed, network cost) fails: the search cannot match it. Empty = the launcher can rent it
    (a verified host is preferred, see `host_flag`)."""
    failed = check_failures(o, floor, verified_only)
    want = MIN_INET_DOWN if min_inet_down is None else max(float(min_inet_down), 0.0)
    if want > 0:
        link = inet_down_of(o)
        if link is None:
            failed.append("download speed (inet_down) not reported")
        elif link < want:
            failed.append(f"download link {link:.0f} Mbps below {want:g}")
    for key in ("inet_down_cost", "inet_up_cost"):
        cost = number(o.get(key))
        if cost is not None and cost > max_net_cost:
            failed.append(f"network cost {key} ${cost:g}/GB over ${max_net_cost:g}")
    n = number(o.get("num_gpus"))
    if not gpus and (n is None or n < 1):
        failed.append("no GPU (the launcher's search asks for at least one)")
    if gpus or not cpu_only:
        failed += gpu_failures(o, gpus, min_gpu_ram_gb, min_compute_cap)
    return failed
