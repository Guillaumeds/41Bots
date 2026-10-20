"""Rent one dedicated sx50_2 replacement and leave both machines running for a verified handoff."""

import json
import time

import launch

SOURCE_ID = 56879188
LABEL = "retained-sx50_2-transfer"
MAX_DPH = 0.999
MIN_THREADS = 191
POLL_S = 60


def instances():
    rows = launch.vast("show", "instances")
    if not isinstance(rows, list):
        raise RuntimeError("cannot read Vast instances; refusing to rent")
    return rows


def inventory(rows):
    source = [row for row in rows if row.get("id") == SOURCE_ID]
    if len(source) != 1 or source[0].get("actual_status") != "running":
        raise RuntimeError("silver-3 source not confirmed running; refusing to rent")
    matches = [row for row in rows if row.get("label") == LABEL]
    if len(matches) > 1:
        raise RuntimeError("multiple transfer rentals found; manual inspection required")
    return matches[0] if matches else None


def settings():
    a = launch.build_parser().parse_args([
        "plan", "--min-cores", str(MIN_THREADS), "--min-ram-gb", "64",
        "--disk-gb", "40", "--min-inet-down", "100", "--max-dph", str(MAX_DPH),
        "--min-reliability", "0.90", "--cpu-only", "--any-host", "--cheapest",
        "--price-steps", "off", "--wait-for-offer-min", "0",
    ])
    a.exclude_machines = set()
    a.exclude_offers = set()
    launch.finalize_settings(a)
    return a


def verify_offer(offer):
    if not offer or float(offer.get("cpu_cores_effective") or 0) <= 190:
        raise RuntimeError("offer does not have >190 usable threads")
    if float(offer.get("dph_total") or 999) >= 1:
        raise RuntimeError("offer is not strictly below $1/hour")
    if float(offer.get("reliability2") or 0) < 0.90:
        raise RuntimeError("offer reliability below 90%")
    if float(offer.get("cpu_ram") or 0) < 64_000:
        raise RuntimeError("offer lacks 64 GB RAM")
    if float(offer.get("disk_space") or 0) < 40:
        raise RuntimeError("offer lacks 40 GB disk")
    if float(launch.inet_down_of(offer) or 0) < 100:
        raise RuntimeError("offer lacks 100 Mbps download")
    if float(offer.get("num_gpus") or 0) < 1 or float(offer.get("gpu_frac") or 0) < 1:
        raise RuntimeError("offer lacks a whole GPU")
    if offer.get("is_bid") or offer.get("rentable") is False:
        raise RuntimeError("offer is not a rentable on-demand machine")


def work_complete():
    response = launch._aws("s3api", "head-object", "--bucket", launch.os.environ["R2_BUCKET"],
                           "--key", "runs/sx50_2/DONE", check=False)
    if response.returncode == 0:
        return True
    stderr = response.stderr.decode(errors="replace")
    if "404" in stderr or "Not Found" in stderr:
        return False
    raise RuntimeError("cannot verify sx50_2 completion in R2; refusing to rent")


def rent(offer):
    # No boot entrypoint or runner credentials: the machine waits for a deliberate, checked transfer.
    try:
        response = launch.vast("create", "instance", str(offer["id"]),
                               "--image", "vastai/base-image:cuda-13.0.3-auto",
                               "--disk", "40", "--label", LABEL, "--cancel-unavail", timeout=None)
    except RuntimeError as exc:
        response = None
        print(f"Create response failed ({launch.redact(str(exc))}); checking inventory before any retry", flush=True)
    instance_id = response.get("new_contract") if isinstance(response, dict) else None
    # An unreadable create reply can still mean a successful rental. Never issue a second create.
    for attempt in range(6):
        rows = instances()
        found = inventory(rows)
        if found:
            if instance_id and int(found["id"]) != int(instance_id):
                raise RuntimeError("created id and labelled instance disagree; manual inspection required")
            return found
        if attempt < 5:
            time.sleep(10)
    if instance_id:
        raise RuntimeError(f"Vast returned instance {instance_id}, but it is not visible yet; inspect manually")
    raise RuntimeError("create may have succeeded without a reply; inspect inventory before retrying")


def watch(hours=5):
    a = settings()
    deadline = time.monotonic() + hours * 3600
    while True:
        existing = inventory(instances())
        if work_complete():
            print("sx50_2 is already complete in R2; no replacement needed", flush=True)
            return None
        if existing:
            print("ACQUIRED " + json.dumps({k: existing.get(k) for k in
                  ("id", "actual_status", "cpu_name", "cpu_cores_effective", "cpu_ram", "dph_total", "gpu_name", "label")}), flush=True)
            return existing
        offer = launch.choose(a)
        if offer:
            verify_offer(offer)
            existing = inventory(instances())
            if existing:
                print("ACQUIRED " + json.dumps({k: existing.get(k) for k in
                      ("id", "actual_status", "cpu_name", "cpu_cores_effective", "cpu_ram", "dph_total", "gpu_name", "label")}), flush=True)
                return existing
            if work_complete():
                print("sx50_2 completed during offer search; no replacement needed", flush=True)
                return None
            print(f"Renting offer {offer['id']} at ${offer['dph_total']}/hour; source {SOURCE_ID} is untouched", flush=True)
            found = rent(offer)
            print("ACQUIRED " + json.dumps({k: found.get(k) for k in
                  ("id", "actual_status", "cpu_name", "cpu_cores_effective", "cpu_ram", "dph_total", "gpu_name", "label")}), flush=True)
            return found
        if time.monotonic() >= deadline:
            print("No matching offer in this watch window; the next scheduled worker will continue", flush=True)
            return None
        time.sleep(POLL_S)


if __name__ == "__main__":
    watch()
