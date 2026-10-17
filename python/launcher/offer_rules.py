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
- download speed (vast's `inet_down`, Mb/s) at least MIN_INET_DOWN (100; owner 2026-10-16: the baked 5-6 GB
  image pulls in about 7 minutes at 100 Mbps) in every search; a named offer with a slower link is reported,
  not refused (launch.py);
- price at most MAX_DPH ($1.70 per hour; owner 2026-10-16), for every search and for a named offer too, unless
  the run raises it (--max-dph, or the manifest's machine block);
- network cost at most MAX_NET_COST ($0.10 per GB, up and down; owner 2026-10-14);
- at least one GPU (the search asks num_gpus>=1); a GPU job asks for AT LEAST N (owner 2026-10-16: the price is
  per machine, a 2-GPU box that is cheap per CPU thread is fine; the trainer uses one GPU, cuda:0), with
  compute_cap from MIN_COMPUTE_CAP (600, Pascal) and a driver that supports CUDA MIN_CUDA (12.6): the
  limits of the PyTorch 2.14.1+cu126 build, which runs every card below compute capability 10.0. A CPU-only
  job (the default) leaves the compute-capability and CUDA limits out;
- Blackwell cards (BLACKWELL_COMPUTE_CAP, 1000: 10.0 B200, 12.0 RTX 50-series) are OPT-IN per run (owner,
  2026-10-17), until a smoke test has proven the image with both PyTorch builds: a run that does not say so
  (manifest machine.blackwell: true, or --allow-blackwell) gets exactly the rules above, the compute_cap<1000
  query token included, and a Blackwell offer is refused with `blackwell_optin_failure`. The reason: a run
  dispatched with a runner binary from before the cu130 build knows only cu126 and would fail on such a card.
  With the opt-in the runner installs the 2.14.1+cu130 build, so the host's driver must support CUDA
  MIN_CUDA_BLACKWELL (13.0). The search query cannot say "this card needs that driver", so it drops the
  compute_cap<1000 token and keeps cuda_vers>=12.6, and the launcher drops the Blackwell offers with an older
  driver (`blackwell_driver_failure`);
- GPU model: a run may name the models it accepts (`gpu_names`, spelled as vast reports them in an offer's
  `gpu_name`, like "RTX 4090"); an offer of any other model is refused. No list, no filter.
"""

from __future__ import annotations

MIN_RELIABILITY = 0.90  # owner, 2026-10-16: hosts with reliability of at least 90% are accepted, for every rental
MAX_NET_COST = 0.10  # $/GB up or down (owner 2026-10-14: 0.10; our traffic is tiny)
MIN_INET_DOWN = 100.0  # Mbps of download speed an offer must report (owner, 2026-10-16: 100; 0 = no filter)
MAX_DPH = 1.70  # $ per hour, the default price cap of every search and of a named offer (owner, 2026-10-16); 0 = no cap
SLOW_LINK_MBPS = 200.0  # below this download speed the launcher's ranking adds a penalty (link_factor)
MIN_COMPUTE_CAP = 600  # vast's units: 600 = compute capability 6.0, the lowest the PyTorch 2.14.1+cu126 build supports
BLACKWELL_COMPUTE_CAP = 1000  # 10.0 (B200) and above (12.0, RTX 50-series): the cards the runner gives the PyTorch cu130 build
MIN_CUDA = 12.6  # the driver must support CUDA 12.6 (every GPU job; the query's cuda_vers floor)
MIN_CUDA_BLACKWELL = 13.0  # ...and CUDA 13.0 for a card of BLACKWELL_COMPUTE_CAP or above (PyTorch 2.14.1+cu130)


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


def gpu_name_key(name) -> str:
    """A GPU model name as `gpu_name_failures` compares it: lower case, every run of whitespace one space."""
    return " ".join(name.split()).lower() if isinstance(name, str) else ""


def gpu_name_failures(o: dict, gpu_names) -> list[str]:
    """Why an offer's GPU model (vast's `gpu_name`, like "RTX 4090") is not one of `gpu_names`, the models a run
    accepts, spelled as vast reports them and compared without case or extra whitespace. None or an empty list
    means the run named none: no filter. With a list, an offer that reports no model fails."""
    if not gpu_names:
        return []
    key = gpu_name_key(o.get("gpu_name"))
    if key and key in {gpu_name_key(x) for x in gpu_names}:
        return []
    return [f"GPU {o.get('gpu_name')} not in gpu_names" if key else "GPU model not reported, and gpu_names is set"]


def blackwell_optin_failure(o: dict) -> str | None:
    """Why an offer's card is refused for a run that did not opt in to Blackwell cards: it is one (compute_cap
    BLACKWELL_COMPUTE_CAP or above). None for any other offer. The run opts in with the manifest's machine.blackwell
    (true) or --allow-blackwell, and needs a runner binary that knows the cu130 build."""
    cc = number(o.get("compute_cap"))
    if cc is None or cc < BLACKWELL_COMPUTE_CAP:
        return None
    return (f"Blackwell card (compute capability {o.get('compute_cap')}): this run did not opt in "
            f"(machine.blackwell: true, with a runner binary from eng-cu130 or later)")


def blackwell_driver_failure(o: dict) -> str | None:
    """Why an offer's card cannot be used: it is a Blackwell card (compute_cap BLACKWELL_COMPUTE_CAP or above), which
    needs the PyTorch 2.14.1+cu130 build, and its host's driver supports a CUDA below MIN_CUDA_BLACKWELL (13.0) or
    reports none (vast's `cuda_max_good`). None for every other offer (a card below 10.0, one that reports no compute
    capability: `gpu_failures` says that) and for a Blackwell one with a new enough driver."""
    cc, cuda = number(o.get("compute_cap")), number(o.get("cuda_max_good"))
    if cc is None or cc < BLACKWELL_COMPUTE_CAP or (cuda is not None and cuda >= MIN_CUDA_BLACKWELL):
        return None
    return (f"compute capability {o.get('compute_cap')} (Blackwell) runs the PyTorch 2.14.1+cu130 build, which needs a "
            f"driver that supports CUDA {MIN_CUDA_BLACKWELL:.1f}; this host's supports {o.get('cuda_max_good')}")


def gpu_failures(o: dict, gpus: int, min_gpu_ram_gb: float = 0, min_compute_cap: int = MIN_COMPUTE_CAP,
                 allow_blackwell: bool = False) -> list[str]:
    """Why an offer fails a GPU job's needs (at least `gpus` GPUs: a machine with more is fine, the price is per
    machine; `min_gpu_ram_gb`, `min_compute_cap`, and the PyTorch build's limits), from vast's raw record
    (gpu_ram in MB, compute_cap x 100). With gpus 0 the GPU count is not checked (a job that wants the limits
    but no particular count). A Blackwell card is refused unless `allow_blackwell` (the run opted in), and then
    needs a driver that supports CUDA MIN_CUDA_BLACKWELL."""
    failed = []
    n, ram, cc, cuda = (number(o.get(k)) for k in ("num_gpus", "gpu_ram", "compute_cap", "cuda_max_good"))
    if gpus and (n is None or int(n) < int(gpus)):
        failed.append(f"{o.get('num_gpus')} GPUs, fewer than the {gpus} asked for")
    if min_gpu_ram_gb and (ram is None or ram < min_gpu_ram_gb * 1000):
        failed.append(f"GPU memory {o.get('gpu_ram')} MB, below --min-gpu-ram-gb {min_gpu_ram_gb}")
    lo = max(MIN_COMPUTE_CAP, int(min_compute_cap or MIN_COMPUTE_CAP))
    if cc is None or cc < lo:
        failed.append(f"compute capability {o.get('compute_cap')} below {lo} (the lowest the PyTorch 2.14.1+cu126 build supports)")
    elif not allow_blackwell and (why := blackwell_optin_failure(o)):
        failed.append(why)
    if cuda is None or cuda < MIN_CUDA:
        failed.append(f"driver supports CUDA {o.get('cuda_max_good')}, below {MIN_CUDA}")
    elif allow_blackwell and (why := blackwell_driver_failure(o)):
        failed.append(why)
    return failed


def search_failures(o: dict, *, min_inet_down: float | None = None, floor: float = MIN_RELIABILITY,
                    verified_only: bool = False, max_net_cost: float = MAX_NET_COST, gpus: int = 0,
                    cpu_only: bool = True, min_gpu_ram_gb: float = 0, min_compute_cap: int = MIN_COMPUTE_CAP,
                    max_dph: float | None = None, gpu_names: list[str] | None = None,
                    allow_blackwell: bool = False) -> list[str]:
    """Why the launcher's SEARCH would not find an offer: `check_failures` plus the other filters of its
    query (download speed, price, network cost, at least one GPU, and for a GPU job or a job that is not
    CPU-only the compute-capability and CUDA limits) and, when given, the GPU models a run accepts (`gpu_names`).
    `min_inet_down` None means the default MIN_INET_DOWN, 0 means no filter; `max_dph` None means the default
    MAX_DPH, 0 means no cap. An offer
    that does not report a value the search filters on (download speed) fails: the search cannot match
    it. Empty = the launcher can rent it (a verified host is preferred, see `host_flag`)."""
    failed = check_failures(o, floor, verified_only)
    cap = MAX_DPH if max_dph is None else float(max_dph)
    price = number(o.get("dph_total"))
    if cap > 0 and price is not None and price > cap:
        failed.append(f"price ${price:.2f}/h above the cap ${cap:.2f}/h")
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
        failed += gpu_failures(o, gpus, min_gpu_ram_gb, min_compute_cap, allow_blackwell)
    return failed + gpu_name_failures(o, gpu_names)
