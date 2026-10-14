# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Charts and a self-contained HTML report from `search curves --json` files (docs/M3-SPEC.md
section 13.1 item 10). Every number comes from the JSON; this script only draws and explains.

Usage:
    uv run python/tools/ncurves_report.py --out docs/m3-ncurves [--score score-t2.json] \\
        curves-t2.json curves-t3.json curves-t4.json

Writes <out>/png/*.png and <out>/index.html (the images are inlined, so the HTML stands alone).
"""

import argparse
import base64
import html
import io
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Reference palette (validated): measured = blue, predicted = orange, reference marks = grey.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
GRIDC = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": GRIDC,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRIDC,
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def pct(x, digits=4):
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def pct_axis(x, _pos):
    v = 100 * x
    if v >= 0.01:
        return f"{v:.2f}%"
    if v >= 0.001:
        return f"{v:.3f}%"
    return f"{v:.4f}%"


def n_text(x, cap=1e9):
    if x is None or not math.isfinite(x):
        return "n/a"
    if x >= cap * 0.999:
        return "never"
    if x > 1e6:
        return "&gt;1,000,000"
    return f"{x:,.0f}"


def thr_label(th):
    return f"{100 * th:g}%"


def ratio(log_ratio):
    if log_ratio is None or not math.isfinite(log_ratio):
        return "n/a"
    return f"{math.exp(log_ratio):.2f}×"


def at_n(t, series, n):
    """A group's series value at grid N (None when N is not on the grid)."""
    if n not in t["grid"]:
        return None
    i = t["grid"].index(n)
    return series[i] if i < len(series) else None


def positive(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and math.isfinite(y) and y > 0]
    return [p[0] for p in pts], [p[1] for p in pts]


def save(fig, out_dir, name, images):
    from PIL import Image  # matplotlib's own dependency

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    # A 128-colour palette keeps the charts sharp at a third of the size.
    buf.seek(0)
    im = Image.open(buf).convert("RGB").quantize(colors=128, method=Image.Quantize.FASTOCTREE)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    data = out.getvalue()
    os.makedirs(os.path.join(out_dir, "png"), exist_ok=True)
    with open(os.path.join(out_dir, "png", name), "wb") as f:
        f.write(data)
    images[name] = base64.b64encode(data).decode("ascii")


def panel(ax, t, g, legend=False):
    """One scenario: measured cv points (filled = used in the fit, hollow = held out) with their
    95% band, the fitted curve, the plain estimate against all M worlds (grey), and the predicted
    total loss L(N) with its band out to where it crosses the thresholds."""
    grid = t["grid"]
    k, fit_max = t["k"], t["fit_max"]
    cv = g["cv"]
    ncv = grid[: len(cv["mean"])]
    # Band.
    lo = [max(v, 1e-9) if v is not None else None for v in cv["lo"]]
    hi = cv["hi"]
    bx = [n for n, a, b in zip(ncv, lo, hi) if a is not None and b is not None and b > 0 and n < k]
    blo = [a for n, a, b in zip(ncv, lo, hi) if a is not None and b is not None and b > 0 and n < k]
    bhi = [b for n, a, b in zip(ncv, lo, hi) if a is not None and b is not None and b > 0 and n < k]
    if bx:
        ax.fill_between(bx, blo, bhi, color=BLUE, alpha=0.14, linewidth=0, label="measured: 95% band")
    used = [(n, y) for n, y in zip(ncv, cv["mean"]) if n <= fit_max]
    held = [(n, y) for n, y in zip(ncv, cv["mean"]) if fit_max < n < k]
    x, y = positive([p[0] for p in used], [p[1] for p in used])
    ax.plot(x, y, "o", ms=4.5, color=BLUE, label=f"measured extra loss vs a {k}-deal choice (fit)")
    x, y = positive([p[0] for p in held], [p[1] for p in held])
    ax.plot(x, y, "o", ms=4.5, mfc=SURFACE, mec=BLUE, mew=1.2, label="measured, held out of the fit")
    # Fitted cv curve.
    fx, fy = positive(ncv, g["fit_cv"])
    if fx:
        ax.plot(fx, fy, "--", color=BLUE, lw=1.4, label="fitted curve for the measured quantity")
    # Plain estimate against all M worlds.
    px, py = positive(grid, g["plain"]["mean"])
    ax.plot(px, py, "x", ms=3.5, color=GREY, mew=1.0,
            label=f"plain: random N of {t['worlds']} vs all {t['worlds']} (reads low near {t['worlds']})")
    # Predicted total loss.
    if g["loss"]:
        smallest = min(t["thresholds"])
        n_last = max([e["v"] for e in g["n_for"]] + [t["worlds"]])
        n_end = min(max(n_last * 1.6, t["worlds"] * 2), 2e5)
        pn = [n for n in t["pred_n"] if n <= n_end]
        lv = g["loss"][: len(pn)]
        ax.fill_between(pn, [max(e["lo"], 1e-9) for e in lv], [e["hi"] for e in lv],
                        color=ORANGE, alpha=0.12, linewidth=0, label="predicted total loss: 95% band")
        ax.plot(pn, [e["v"] for e in lv], "-", color=ORANGE, lw=2,
                label="predicted total loss L(N) = a·N^-b")
        ax.set_xlim(grid[0] * 0.85, n_end)
        ymin = min(smallest * 0.4, min([v for v in cv["mean"] if v and v > 0] or [smallest]) * 0.7)
    else:
        ymin = 1e-6
    ymax = max([v for v in cv["hi"] + g["plain"]["mean"] if v is not None and v > 0] or [1e-3]) * 1.6
    if g["loss"]:
        ymax = max(ymax, g["loss"][0]["v"] * 1.3)
    ymin = max(ymin, 2e-7)
    ax.set_ylim(ymin, ymax)
    if g["loss"]:
        for th in t["thresholds"]:
            if ymin < th < ymax:
                ax.axhline(th, color=GREY, lw=0.8, ls=":")
                ax.text(n_end, th, " " + thr_label(th), va="center", ha="left", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_axis))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    nf = g["n_for"][0] if g["n_for"] else None
    flag = "" if g["enough"] else " (few decisions)"
    ax.set_title(f"{g['name']}  ·  {g['decisions']} decisions{flag}", loc="left")
    if nf:
        ax.text(0.02, 0.03,
                f"N for {thr_label(t['thresholds'][0])}: {n_text(nf['v'])} ({n_text(nf['lo'])}–{n_text(nf['hi'])})",
                transform=ax.transAxes, fontsize=7.5, color=INK, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRIDC, lw=0.6))
    if legend:
        ax.legend(loc="upper right", fontsize=7)


def group_figure(t, groups, name, out_dir, images, cols=4):
    if not groups:
        return None
    rows = math.ceil(len(groups) / cols)
    if len(groups) == 1:
        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        panel(ax, t, groups[0])
        ax.set_ylabel("money-chance loss per decision")
        ax.set_xlabel("deals (worlds) N")
        ax.legend(loc="center left", bbox_to_anchor=(1.07, 0.5), fontsize=8)
        fig.tight_layout()
        save(fig, out_dir, name, images)
        return name
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for i, g in enumerate(groups):
        panel(axes[i // cols][i % cols], t, g)
    for j in range(len(groups), rows * cols):
        axes[j // cols][j % cols].axis("off")
    for r in range(rows):
        axes[r][0].set_ylabel("money-chance loss per decision")
    for c in range(cols):
        axes[rows - 1][c].set_xlabel("deals (worlds) N")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02 - 0.04 / rows), fontsize=8)
    fig.tight_layout(rect=(0, 0.06 if rows > 1 else 0.12, 1, 1))
    save(fig, out_dir, name, images)
    return name


def overview_n(tricks, out_dir, images):
    """N needed for the first threshold, per position, both tricks (dots with 95% whiskers)."""
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    colors = [BLUE, ORANGE, "#1baf7a", "#6250d6"]
    width = 0.18
    for i, t in enumerate(tricks):
        pos = [g for g in t["groups"] if g["kind"] == "position"]
        xs = [int(g["name"][3:]) + (i - (len(tricks) - 1) / 2) * width for g in pos]
        v = [g["n_for"][0]["v"] for g in pos]
        lo = [g["n_for"][0]["lo"] for g in pos]
        hi = [g["n_for"][0]["hi"] for g in pos]
        ax.errorbar(xs, v, yerr=[[a - b for a, b in zip(v, lo)], [c - a for a, c in zip(v, hi)]],
                    fmt="o", ms=6, color=colors[i % 4], ecolor=colors[i % 4], elinewidth=1.4,
                    capsize=0, label=f"trick {t['trick']}")
        allg = next(g for g in t["groups"] if g["kind"] == "all")
        ax.axhline(allg["n_for"][0]["v"], color=colors[i % 4], lw=1, ls="--", alpha=0.7,
                   label=f"trick {t['trick']}, all decisions")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([f"{i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'}" for i in range(1, 9)])
    ax.set_xlabel("position in the trick (1st = leader)")
    ax.set_ylabel("deals needed")
    th = tricks[0]["thresholds"][0]
    ax.set_title(f"Deals needed per decision for an expected loss of {thr_label(th)} money chance (dot), with 95% range", loc="left")
    ax.set_xlim(0.5, 8.5)
    ax.legend(loc="lower left", ncol=2, fontsize=7.5)
    fig.tight_layout()
    save(fig, out_dir, "overview-n-by-position.png", images)


def overview_curves(tricks, out_dir, images):
    """Predicted L(N) for every position, one panel per trick (8 lines, legend + end labels)."""
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, axes = plt.subplots(1, len(tricks), figsize=(5.2 * len(tricks), 3.8), squeeze=False)
    for ax, t in zip(axes[0], tricks):
        pn = [n for n in t["pred_n"] if n <= 5000]
        for g in [g for g in t["groups"] if g["kind"] == "position"]:
            p = int(g["name"][3:])
            ys = [e["v"] for e in g["loss"][: len(pn)]]
            ax.plot(pn, ys, color=pal[p - 1], lw=1.8, label=f"position {p}")
        allg = next(g for g in t["groups"] if g["kind"] == "all")
        ax.plot(pn, [e["v"] for e in allg["loss"][: len(pn)]], color=INK, lw=1.2, ls="--", label="all decisions")
        for th in t["thresholds"]:
            ax.axhline(th, color=GREY, lw=0.8, ls=":")
            ax.text(pn[0], th, thr_label(th), va="bottom", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(pct_axis))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
        ax.set_xlabel("deals (worlds) N")
        ax.set_ylabel("predicted money-chance loss per decision")
        b = t["b"]
        ax.set_title(f"Trick {t['trick']}: predicted loss by position (b = {b['v']:.2f})", loc="left")
        ax.set_ylim(min(t["thresholds"]) * 0.3, None)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out_dir, "overview-predicted-loss.png", images)


def overview_b(tricks, out_dir, images):
    """Each main group's own exponent with its 95% range, against the shared one."""
    fig, axes = plt.subplots(1, len(tricks), figsize=(5.2 * len(tricks), 3.4), squeeze=False)
    for ax, t in zip(axes[0], tricks):
        gs = [g for g in t["groups"] if g["free_b"]]
        ys = list(range(len(gs)))[::-1]
        for y, g in zip(ys, gs):
            fb = g["free_b"]
            ax.plot([fb["lo"], fb["hi"]], [y, y], color=BLUE, lw=1.6)
            ax.plot([fb["v"]], [y], "o", color=BLUE, ms=5)
        b = t["b"]
        ax.axvspan(b["lo"], b["hi"], color=ORANGE, alpha=0.12, lw=0)
        ax.axvline(b["v"], color=ORANGE, lw=1.8, label=f"shared b = {b['v']:.2f} ({b['lo']:.2f}–{b['hi']:.2f})")
        ax.set_yticks(ys)
        ax.set_yticklabels([g["name"] for g in gs])
        ax.set_xlim(0, 3)
        ax.set_xlabel("exponent b (higher = loss falls faster as N grows)")
        ax.set_title(f"Trick {t['trick']}: each group's own b (dot, 95% range)", loc="left")
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "overview-exponents.png", images)


def exact_at(t, g, n):
    """Exact PIMC's predicted loss at N for a group: a·N^-b, with the range interpolated on the
    log-log prediction grid."""
    if not g["loss"] or not g["a"]:
        return None
    v = g["a"]["v"] * n ** (-t["b"]["v"])
    pn = t["pred_n"]
    j = max(i for i in range(len(pn)) if pn[i] <= n) if n >= pn[0] else 0
    j = min(j, len(pn) - 2)
    f = (math.log(n) - math.log(pn[j])) / (math.log(pn[j + 1]) - math.log(pn[j]))
    lo = math.exp((1 - f) * math.log(g["loss"][j]["lo"]) + f * math.log(g["loss"][j + 1]["lo"]))
    hi = math.exp((1 - f) * math.log(g["loss"][j]["hi"]) + f * math.log(g["loss"][j + 1]["hi"]))
    return {"v": v, "lo": lo, "hi": hi}


def rollout_section(score, tricks, out_dir, images):
    """Rollout-PIMC (rules-v1 inner player) against the exact yardstick, and exact PIMC with the
    same number of deals, from a `search score --json` file."""
    methods = sorted([m for m in score["methods"] if m["kind"] == "rollout"], key=lambda m: m["n"])
    if not methods:
        return ""
    tr = methods[0]["trick"]
    t = next((t for t in tricks if t["trick"] == tr), None)
    if t is None:
        return ""
    big = methods[-1]
    nb = big["n"]
    groups = {g["name"]: g for g in t["groups"]}
    # Chart 1: loss against N, all decisions.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ns = [m["n"] for m in methods]
    alls = [next(g for g in m["groups"] if g["name"] == "all") for m in methods]
    ax.errorbar(ns, [g["loss"]["v"] for g in alls],
                yerr=[[g["loss"]["v"] - g["loss"]["lo"] for g in alls], [g["loss"]["hi"] - g["loss"]["v"] for g in alls]],
                fmt="o", ms=6, color=BLUE, elinewidth=1.4, capsize=0, label="rollout-PIMC (measured)")
    ga = groups["all"]
    pn = [n for n in t["pred_n"] if n <= 400]
    lv = ga["loss"][: len(pn)]
    ax.fill_between(pn, [e["lo"] for e in lv], [e["hi"] for e in lv], color=ORANGE, alpha=0.12, lw=0)
    ax.plot(pn, [e["v"] for e in lv], color=ORANGE, lw=2, label="exact PIMC (predicted)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_axis))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.set_xlabel("deals (worlds) N")
    ax.set_ylabel("money-chance loss per decision")
    ax.set_title(f"Trick {tr}, all decisions: loss against N", loc="left")
    ax.legend(loc="lower left")
    # Chart 2: per position at the largest N.
    ax = axes[1]
    pos = [g for g in big["groups"] if g["name"].startswith("pos")]
    xs = [int(g["name"][3:]) for g in pos]
    ax.errorbar([x - 0.12 for x in xs], [g["loss"]["v"] for g in pos],
                yerr=[[g["loss"]["v"] - g["loss"]["lo"] for g in pos], [g["loss"]["hi"] - g["loss"]["v"] for g in pos]],
                fmt="o", ms=6, color=BLUE, elinewidth=1.4, capsize=0, label=f"rollout-PIMC, {nb} deals")
    ex = [exact_at(t, groups[g["name"]], nb) for g in pos]
    ax.errorbar([x + 0.12 for x in xs], [e["v"] for e in ex],
                yerr=[[e["v"] - e["lo"] for e in ex], [e["hi"] - e["v"] for e in ex]],
                fmt="s", ms=5.5, color=ORANGE, elinewidth=1.4, capsize=0, label=f"exact PIMC, {nb} deals (predicted)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_axis))
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("position in the trick (1 = leader)")
    ax.set_title(f"Trick {tr}: loss per position with {nb} deals (95% range)", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, f"t{tr}-rollout-vs-exact.png", images)
    rows = []
    for g in [next(g for g in big["groups"] if g["name"] == "all")] + pos + [g for g in big["groups"] if g["name"].startswith("S")]:
        e = exact_at(t, groups[g["name"]], nb)
        ag, lo = g["agreement"], g["loss"]
        rows.append(
            f"<tr><td>{g['name']}</td><td>{g['decisions']}</td>"
            f"<td>{100 * ag['v']:.0f}% <span class=r>({100 * ag['lo']:.0f}–{100 * ag['hi']:.0f}%)</span></td>"
            f"<td>{pct(lo['v'], 3)} <span class=r>({pct(lo['lo'], 3)}–{pct(lo['hi'], 3)})</span></td>"
            f"<td>{pct(g['max_loss'], 1)}</td>"
            f"<td>{pct(e['v'], 3)} <span class=r>({pct(e['lo'], 3)}–{pct(e['hi'], 3)})</span></td>"
            f"<td>{lo['v'] / e['v']:.0f}×</td></tr>")
    trend = ", ".join(
        f"{m['n']} deals: {100 * next(g for g in m['groups'] if g['name'] == 'all')['agreement']['v']:.0f}% agree, "
        f"loss {pct(next(g for g in m['groups'] if g['name'] == 'all')['loss']['v'], 2)}" for m in methods)
    a_all = next(g for g in big["groups"] if g["name"] == "all")
    e_all = exact_at(t, ga, nb)
    worst = max(pos, key=lambda g: g["max_loss"])
    early = sum(g["loss"]["v"] for g in pos if int(g["name"][3:]) <= 4) / max(1, sum(1 for g in pos if int(g["name"][3:]) <= 4))
    late = sum(g["loss"]["v"] for g in pos if int(g["name"][3:]) > 4) / max(1, sum(1 for g in pos if int(g["name"][3:]) > 4))
    return f"""
<h2 id=rollout>Rollout-PIMC against the exact yardstick (trick {tr})</h2>
<p>Rollout-PIMC is the cheap early-trick teacher: instead of solving each imagined deal exactly, it plays
each deal out with everyone following the simple rules bot (rules-v1). Here its choices, with {', '.join(str(n) for n in ns)}
deals of its own, are scored against exact PIMC with all {score['n_ref']} deals (independent deals, so the
referee does not flatter it).</p>
<div class=box><ul>
<li><b>Agreement:</b> with {nb} deals rollout-PIMC picks one of exact PIMC's best cards in
{100 * a_all['agreement']['v']:.0f}% of decisions.</li>
<li><b>Money:</b> it loses {pct(a_all['loss']['v'], 2)} money chance per decision on average
(range {pct(a_all['loss']['lo'], 2)}–{pct(a_all['loss']['hi'], 2)}), about {a_all['loss']['v'] / e_all['v']:.0f}× what
exact PIMC with the same {nb} deals is predicted to lose ({pct(e_all['v'], 3)}).</li>
<li><b>More deals barely help:</b> {trend}. The error comes from assuming everyone plays like the rules bot,
not from too few deals.</li>
<li><b>Position matters:</b> positions 1–4 lose {pct(early, 2)} on average, positions 5–8 {pct(late, 2)}. Losses are lopsided: the single worst decision
(decision {worst['max_loss_id']}, {worst['name']}) loses {pct(worst['max_loss'], 0)} money chance on its own, which
dominates that position's average, hence its wide range.</li>
</ul></div>
{img(images, f"t{tr}-rollout-vs-exact.png", "Rollout-PIMC against exact PIMC")}
<table><thead><tr><th>group</th><th>decisions</th><th>rollout agrees</th><th>rollout loss</th>
<th>worst single loss</th><th>exact PIMC loss, same deals</th><th>ratio</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class=sub>Rollout numbers are measured; the exact-PIMC column is the fitted curve of the trick-{tr} section at
N = {nb}. The yardstick itself (exact PIMC, {score['n_ref']} deals) is predicted to lose about
{pct(exact_at(t, ga, score['n_ref'])['v'], 4)}, far below these losses.</p>
"""


def img(images, name, alt):
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{images[name]}"></figure>'


def est_text(e):
    return f"{n_text(e['v'])} <span class=r>({n_text(e['lo'])}–{n_text(e['hi'])})</span>"


def n_table(t, groups):
    th = "".join(f"<th>N for {thr_label(x)}</th>" for x in t["thresholds"])
    rows = []
    for g in groups:
        nf = "".join(f"<td>{est_text(e)}</td>" for e in g["n_for"]) or "<td colspan=3>no fit</td>"
        a = g["a"]
        fb = g["free_b"]
        warn = "" if g["enough"] else " <span class=warn>few</span>"
        rows.append(
            f"<tr><td>{html.escape(g['name'])}{warn}</td><td>{g['decisions']}</td>"
            f"<td>{pct(g['cv']['mean'][0])}</td>"
            f"<td>{(f'{fb['v']:.2f} <span class=r>({fb['lo']:.2f}–{fb['hi']:.2f})</span>') if fb else '–'}</td>"
            f"{nf}</tr>"
        )
    return (f"<table><thead><tr><th>group</th><th>decisions</th><th>measured loss at N = {t['grid'][0]}</th>"
            f"<th>own b</th>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


FEATURE_KINDS = ("has_five", "robber", "voids")
FEATURE_ORDER = ["has5-yes", "has5-no", "robber-yes", "robber-no", "voids0", "voids1", "voids2+"]
FEATURE_TEXT = {
    "has5-yes": "holds the 5 of trumps",
    "has5-no": "does not hold the 5",
    "robber-yes": "robber (took a turned card)",
    "robber-no": "not a robber",
    "voids0": "no void plain suit",
    "voids1": "void in one plain suit",
    "voids2+": "void in two or more",
}
CONTRAST_TEXT = {
    ("has5-yes", "has5-no"): "Holding the 5 of trumps",
    ("robber-yes", "robber-no"): "Being the robber",
    ("voids1", "voids0"): "One void plain suit rather than none",
    ("voids2+", "voids1"): "Two or more voids rather than one",
}


def feature_groups(t):
    return [g for g in t["groups"] if g["kind"] in FEATURE_KINDS]


def ratio_text(e):
    return f"{e['v']:.1f}× <span class=r>({e['lo']:.1f}–{e['hi']:.1f}×)</span>"


def verdict(e):
    if e["lo"] > 1:
        return "more deals, clearly"
    if e["hi"] < 1:
        return "fewer deals, clearly"
    return "no clear difference yet"


def overview_features(tricks, out_dir, images):
    """N for the first threshold per hand-feature group (dots with 95% whiskers), one series per
    trick (fixed colours and marker shapes, as in the position chart)."""
    colors = [BLUE, ORANGE, "#1baf7a", "#6250d6"]
    marks = ["o", "s", "D", "^"]
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    width = 0.2
    for i, t in enumerate(tricks):
        by_name = {g["name"]: g for g in feature_groups(t) if g["n_for"]}
        xs, v, lo, hi = [], [], [], []
        for j, name in enumerate(FEATURE_ORDER):
            g = by_name.get(name)
            if g is None or not g["enough"]:
                continue
            e = g["n_for"][0]
            xs.append(j + (i - (len(tricks) - 1) / 2) * width)
            v.append(e["v"])
            lo.append(e["lo"])
            hi.append(e["hi"])
        ax.errorbar(xs, v, yerr=[[a - b for a, b in zip(v, lo)], [c - a for a, c in zip(v, hi)]],
                    fmt=marks[i % 4], ms=6, color=colors[i % 4], ecolor=colors[i % 4], elinewidth=1.4,
                    capsize=0, label=f"trick {t['trick']}")
    for x in (1.5, 3.5):
        ax.axvline(x, color=GRIDC, lw=1.2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.set_xticks(range(len(FEATURE_ORDER)))
    ax.set_xticklabels([n.replace("-", " ") for n in FEATURE_ORDER])
    ax.set_xlim(-0.5, len(FEATURE_ORDER) - 0.5)
    ax.set_ylabel("deals needed")
    th = tricks[0]["thresholds"][0]
    ax.set_title(f"Deals needed for an expected loss of {thr_label(th)}, by hand feature (dot, 95% range)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(tricks), fontsize=7.5)
    fig.tight_layout()
    save(fig, out_dir, "overview-hand-features.png", images)


def features_section(tricks, out_dir, images):
    """The hand features added on 2026-10-14 (holds the 5 of trumps, robber, voids): N for the
    first threshold per group and trick, and the paired comparisons."""
    have = [t for t in tricks if feature_groups(t)]
    if not have:
        return ""
    overview_features(have, out_dir, images)
    th = have[0]["thresholds"][0]
    head = "".join(f"<th>trick {t['trick']}: decisions</th><th>N for {thr_label(th)}</th>" for t in have)
    rows = []
    for name in FEATURE_ORDER:
        cells = []
        for t in have:
            g = next((g for g in feature_groups(t) if g["name"] == name), None)
            if g is None:
                cells.append("<td>0</td><td>–</td>")
                continue
            warn = "" if g["enough"] else " <span class=warn>few</span>"
            nf = est_text(g["n_for"][0]) if g["n_for"] else "no fit"
            cells.append(f"<td>{g['decisions']}{warn}</td><td>{nf}</td>")
        rows.append(f"<tr><td>{html.escape(name)} <span class=r>{FEATURE_TEXT[name]}</span></td>{''.join(cells)}</tr>")
    table = (f"<table><thead><tr><th>group</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")
    chead = "".join(f"<th>trick {t['trick']}</th>" for t in have)
    crow = []
    for (a, b), label in CONTRAST_TEXT.items():
        cells = []
        for t in have:
            c = next((c for c in t.get("contrasts", []) if c["group"] == a and c["vs"] == b), None)
            if c is None:
                cells.append("<td>–</td>")
                continue
            enough = all(g["enough"] for g in feature_groups(t) if g["name"] in (a, b))
            warn = "" if enough else " <span class=warn>few</span>"
            cells.append(f"<td>{ratio_text(c['n_ratio'])}{warn}<br><span class=r>{verdict(c['n_ratio'])}</span></td>")
        crow.append(f"<tr><td>{label} <span class=r>({a} ÷ {b})</span></td>{''.join(cells)}</tr>")
    ctable = (f"<table><thead><tr><th>deals needed by …</th>{chead}</tr></thead>"
              f"<tbody>{''.join(crow)}</tbody></table>")
    # Plain-English findings, from the numbers.
    facts = []
    for (a, b), label in CONTRAST_TEXT.items():
        parts = []
        for t in have:
            c = next((c for c in t.get("contrasts", []) if c["group"] == a and c["vs"] == b), None)
            if c is None:
                continue
            ga = next(g for g in feature_groups(t) if g["name"] == a)
            gb = next(g for g in feature_groups(t) if g["name"] == b)
            few = "" if ga["enough"] and gb["enough"] else ", too few decisions to judge"
            parts.append(f"trick {t['trick']}: {n_text(ga['n_for'][0]['v'])} against {n_text(gb['n_for'][0]['v'])} "
                         f"deals, {c['n_ratio']['v']:.1f}× ({verdict(c['n_ratio'])}{few})")
        if parts:
            facts.append(f"<li><b>{label}:</b> " + "; ".join(parts) + ".</li>")
    def size(t, name):
        return next((g["decisions"] for g in feature_groups(t) if g["name"] == name), 0)

    yes = [size(t, n) for t in have for n in ("has5-yes", "robber-yes")]
    no_void = ", ".join(f"{size(t, 'voids0')} at trick {t['trick']}" for t in have)
    sparse = [str(t["trick"]) for t in have if size(t, "voids0") < 25]
    return f"""
<h2 id=features>The 5 of trumps, robbing and voids</h2>
<p>Three more facts about the deciding player's hand, each something the player can see for themselves
(owner, 2026-10-14): whether they <b>hold the 5 of trumps</b> (the bonus card), whether they are
<b>the robber</b> (they took a turned card this hand: robbed it with the ace of trumps, or, as dealer,
robbed a turned ace or took a turned Joker), and how many <b>plain suits they are void in</b> [hold no card
of]. Partners always sit across, so there is no feature for that. These were worked out again from the saved
decisions (each one replayed from its deal and the cards played), so nothing was solved again.</p>
{img(images, "overview-hand-features.png", "Deals needed by hand feature, per trick")}
<p class=sub>Groups with fewer than 25 decisions are left out of the chart{
f" (the no-void group at trick{'s' if len(sparse) > 1 else ''} {' and '.join(sparse)})" if sparse else ""}. Deals needed for an expected loss of {thr_label(th)} per decision, with the 95% range in brackets:</p>
{table}
<p>The fairest comparison is the ratio of the deals two groups need, because both groups are refitted on the
same 1,000 re-draws of the decisions. A ratio above 1 means the first group needs more deals; when the whole
range is above 1 the difference is clear, when the range straddles 1 these samples cannot tell yet.</p>
{ctable}
<ul>{''.join(facts)}</ul>
<p class=sub>The holders of the 5 and the robbers are only {min(yes)}–{max(yes)} decisions per trick (the
decisions were not chosen with these features in mind), so most ranges are wide; like every group, their curves
use the trick's shared exponent. Players hold fewer cards as the hand goes on, so hands with no void get rare
(decisions with no void: {no_void}). The trick-1 set can sample more decisions in a group that turns out to need
many deals (M3-SPEC §13.1 item 9).</p>
"""


def compare_table(t, groups, ns):
    """Measured loss at chosen N by the three readings: cross-fitted, plain, nested prefix."""
    head = "".join(f"<th>N = {n}</th>" for n in ns)
    rows = []
    for g in groups:
        def cell(series, n):
            i = t["grid"].index(n) if n in t["grid"] else None
            if i is None or i >= len(series["mean"]):
                return "–"
            return pct(series["mean"][i])
        rows.append(f"<tr><td rowspan=3>{html.escape(g['name'])}</td><td>split-half (fair referee)</td>"
                    + "".join(f"<td>{cell(g['cv'], n)}</td>" for n in ns) + "</tr>")
        rows.append(f"<tr><td>plain, vs all {t['worlds']}</td>"
                    + "".join(f"<td>{cell(g['plain'], n)}</td>" for n in ns) + "</tr>")
        rows.append("<tr><td>first-N only (one nested run)</td>"
                    + "".join(f"<td>{cell(g['prefix'], n)}</td>" for n in ns) + "</tr>")
    return f"<table><thead><tr><th>group</th><th>reading</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --rule:#e4e3df; --card:#ffffff;
        --accent:#2a78d6; --warn:#b4540f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --rule:#3a3936; --card:#232321;
  --accent:#3987e5; --warn:#f0a35e; } }
:root[data-theme="dark"] { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --rule:#3a3936;
  --card:#232321; --accent:#3987e5; --warn:#f0a35e; }
body { background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
       margin:0; padding:0 16px; }
main { max-width:1080px; margin:0 auto; padding:24px 0 64px; }
h1 { font-size:26px; margin:8px 0 4px; } h2 { font-size:20px; margin:36px 0 8px; border-bottom:1px solid var(--rule); padding-bottom:4px; }
h3 { font-size:16px; margin:24px 0 6px; }
p, li { color:var(--ink); max-width:72ch; } .sub { color:var(--ink2); }
.box { background:var(--card); border:1px solid var(--rule); border-radius:8px; padding:12px 18px; margin:14px 0; }
figure { margin:12px 0; background:#fcfcfb; border-radius:6px; padding:6px; overflow-x:auto; }
figure img { max-width:100%; height:auto; display:block; }
table { border-collapse:collapse; font-size:13px; margin:8px 0 16px; display:block; overflow-x:auto; }
th, td { border-bottom:1px solid var(--rule); padding:4px 10px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } th { color:var(--ink2); font-weight:600; }
.r { color:var(--ink2); font-size:12px; } .warn { color:var(--warn); font-size:11px; }
code { font-size:13px; } details summary { cursor:pointer; color:var(--accent); }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", action="append", default=[],
                    help="a `search score --json` file of rollout-PIMC against the exact yardstick")
    ap.add_argument("json", nargs="+")
    a = ap.parse_args()
    tricks = []
    for p in a.json:
        with open(p) as f:
            tricks.extend(json.load(f)["tricks"])
    tricks.sort(key=lambda t: t["trick"])
    images = {}
    overview_n(tricks, a.out, images)
    overview_curves(tricks, a.out, images)
    overview_b(tricks, a.out, images)
    features = features_section(tricks, a.out, images)
    rollout = ""
    for sp in a.score:
        with open(sp) as f:
            rollout += rollout_section(json.load(f), tricks, a.out, images)
    sections = []
    for t in tricks:
        tr = t["trick"]
        by = lambda kind: [g for g in t["groups"] if g["kind"] == kind]  # noqa: E731
        allg = by("all")
        group_figure(t, allg, f"t{tr}-all.png", a.out, images, cols=1)
        group_figure(t, by("position"), f"t{tr}-positions.png", a.out, images)
        group_figure(t, by("stratum"), f"t{tr}-strata.png", a.out, images)
        hand = by("trumps") + by("top_trumps") + by("longest_plain")
        group_figure(t, hand, f"t{tr}-hand-types.png", a.out, images)
        feats = feature_groups(t)
        if feats:
            group_figure(t, feats, f"t{tr}-hand-features.png", a.out, images)
        cells = [g for g in t["groups"] if "×" in g["kind"]]
        b = t["b"]
        ho_in, ho_tot = t["heldout_inside"]
        hf_in, hf_tot = t["half_inside"]
        drift = ", ".join(f"up to N = {n}: {v:.2f}" for n, v in t["b_by_range"])
        bh = t["b_half"]
        ns = [n for n in (10, 20, 40, 80, 160, 320) if n in t["grid"]]
        s = [f"<h2 id=t{tr}>Trick {tr}</h2>",
             f"<p class=sub>{t['decisions']} decisions, {t['worlds']} exactly solved deals each; "
             f"fair referee = {t['k']} deals; curve fitted on N up to {t['fit_max']}; "
             f"{t['splits']} random splits per decision; ranges from {t['boots']} bootstrap re-draws of the decisions.</p>",
             "<div class=box><ul>",
             f"<li><b>Exponent b = {b['v']:.2f}</b> (95% range {b['lo']:.2f}–{b['hi']:.2f}): doubling the deals "
             f"cuts the expected loss by about {100 * (1 - 2 ** -b['v']):.0f}%.</li>",
             f"<li>Fitted on different ranges of N the exponent stays put ({drift}), so the curve is not bending "
             f"within the measured range.</li>",
             f"<li><b>Held-out check:</b> fitted on N &le; {t['fit_max']} only, the curve predicted the measurements for "
             f"with N between {t['fit_max']} and {t['k']} inside their 95% band "
             f"{ho_in} times out of {ho_tot} (all decisions and the 8 positions); on average the prediction was "
             f"{ratio(t['heldout_log_ratio'])} the measurement.</li>",
             f"<li><b>Extrapolation check:</b> the same recipe run on only the first {t['worlds'] // 2} deals "
             f"(b = {bh['v']:.2f}, range {bh['lo']:.2f}–{bh['hi']:.2f}) predicted the {t['worlds']}-deal measurements "
             f"inside their band {hf_in} times out of {hf_tot}, averaging {ratio(t['half_log_ratio'])} the measurement.</li>",
             f"<li><b>All decisions:</b> about {n_text(allg[0]['n_for'][0]['v'])} deals for {thr_label(t['thresholds'][0])} "
             f"(range {n_text(allg[0]['n_for'][0]['lo'])}–{n_text(allg[0]['n_for'][0]['hi'])}), "
             f"{n_text(allg[0]['n_for'][1]['v'])} for {thr_label(t['thresholds'][1])}, "
             f"{n_text(allg[0]['n_for'][2]['v'])} for {thr_label(t['thresholds'][2])}.</li>",
             "</ul></div>",
             "<h3>All decisions together</h3>", img(images, f"t{tr}-all.png", f"Trick {tr} loss curve, all decisions"),
             "<h3>By position in the trick</h3>", img(images, f"t{tr}-positions.png", f"Trick {tr} loss curves by position"),
             n_table(t, allg + by("position")),
             "<h3>By match situation (S1 early … S4 near the money)</h3>",
             img(images, f"t{tr}-strata.png", f"Trick {tr} loss curves by stratum"),
             n_table(t, by("stratum")),
             "<h3>By hand type (trumps held, top trumps held, longest plain suit)</h3>",
             img(images, f"t{tr}-hand-types.png", f"Trick {tr} loss curves by hand type"),
             n_table(t, hand)]
        if feats:
            s += ["<h3>By hand feature (holds the 5 of trumps, robber, void plain suits)</h3>",
                  img(images, f"t{tr}-hand-features.png", f"Trick {tr} loss curves by hand feature"),
                  n_table(t, feats)]
        s += ["<details><summary>Cells: position × situation × trumps held (small groups; indicative only)</summary>",
              n_table(t, cells), "</details>",
              "<details><summary>The three readings of the measured loss, side by side</summary>",
              "<p>Split-half is what the curve is fitted to. The plain reading (random N deals judged against all "
              f"{t['worlds']}) reads low as N approaches {t['worlds']}, because its referee shares deals with the player. "
              "The first-N reading is the single nested run the agreement report uses: noisy, since it is one draw.</p>",
              compare_table(t, allg + by("position"), ns), "</details>"]
        sections.append("\n".join(s))
    t0 = tricks[0]
    th = t0["thresholds"]
    summary_rows = []
    for t in tricks:
        allg = next(g for g in t["groups"] if g["kind"] == "all")
        pos = [g for g in t["groups"] if g["kind"] == "position"]
        cells = "".join(f"<td>{est_text(g['n_for'][0])}</td>" for g in pos)
        summary_rows.append(f"<tr><td>trick {t['trick']}</td><td>{t['b']['v']:.2f} <span class=r>({t['b']['lo']:.2f}–{t['b']['hi']:.2f})</span></td>"
                            f"<td>{est_text(allg['n_for'][0])}</td>{cells}</tr>")
    facts = []
    for t in tricks:
        allg = next(g for g in t["groups"] if g["kind"] == "all")
        n0, k, m = t["grid"][0], t["k"], t["worlds"]
        c0, ck = at_n(t, allg["plain_change"], n0), at_n(t, allg["plain_change"], k)
        l0 = at_n(t, allg["plain"]["mean"], n0)
        near_m = min(range(len(t["pred_n"])), key=lambda i: abs(math.log(t["pred_n"][i] / m)))
        lm = allg["loss"][near_m]
        facts.append(
            f"<li><b>Trick {t['trick']}:</b> a {n0}-deal choice differs from the {m}-deal choice in "
            f"{100 * c0:.0f}% of decisions, yet costs only {pct(l0, 3)} money chance on average; even "
            f"{k} deals still differ in {100 * ck:.0f}%. Changed choices are nearly always near-ties. "
            f"The {m}-deal yardstick itself is predicted to lose about {pct(lm['v'], 4)} per decision "
            f"(range {pct(lm['lo'], 4)}–{pct(lm['hi'], 4)}).</li>")
    # Groups whose own exponent range leaves out the shared value.
    outliers = []
    for t in tricks:
        b = t["b"]["v"]
        for g in t["groups"]:
            fb = g["free_b"]
            if fb and not (fb["lo"] <= b <= fb["hi"]):
                side = ("falls faster than the shared curve says, so its predicted N errs on the safe (high) side"
                        if fb["lo"] > b else
                        "falls more slowly than the shared curve says, so its predicted N may be too low")
                outliers.append(f"trick {t['trick']} {g['name']} (own b {fb['v']:.2f}, range "
                                f"{fb['lo']:.2f}–{fb['hi']:.2f}): its loss {side}")
    b_note = ("Every group's own range includes the shared value, which supports using one exponent."
              if not outliers else
              "Most groups' own ranges include the shared value; the exceptions: " + "; ".join(outliers) + ".")
    summary = ("<table><thead><tr><th></th><th>b</th><th>all</th>"
               + "".join(f"<th>pos {i}</th>" for i in range(1, 9))
               + f"</tr></thead><tbody>{''.join(summary_rows)}</tbody></table>")
    # One line on the hand features: the comparisons whose whole range is on one side of 1.
    clear = []
    for t in tricks:
        for c in t.get("contrasts", []):
            e = c["n_ratio"]
            if e["lo"] > 1 or e["hi"] < 1:
                label = CONTRAST_TEXT.get((c["group"], c["vs"]), f"{c['group']} vs {c['vs']}")
                clear.append(f"{label.lower()} at trick {t['trick']} ({e['v']:.1f}×, range {e['lo']:.1f}–{e['hi']:.1f}×)")
    features_bullet = ""
    if features:
        features_bullet = (
            "<li><b>Hand features</b> (<a href=#features>details</a>): "
            + ("the clear differences in deals needed: " + "; ".join(clear) + ". "
               if clear else "no comparison shows a clear difference yet. ")
            + "The other comparisons have ranges on both sides of 1, so these samples cannot call them yet.</li>")
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>How Many Deals</title><style>{CSS}</style></head><body><main>
<h1>How many deals does the bot need to imagine?</h1>
<p class=sub>Tricks {', '.join(str(t['trick']) for t in tricks)} · M3 early-trick yardstick (M3-SPEC §13.1 item 10) ·
made by <code>search curves</code> and <code>python/tools/ncurves_report.py</code></p>

<div class=box>
<p><b>The question.</b> When the bot chooses a card it imagines N possible deals [complete guesses of where the
hidden cards are], solves each one exactly, and plays the card that does best on average. More deals means a
better choice but more time. How much money chance does a bot with N deals give away, per decision, compared
with a bot that could imagine every possible deal? And how many deals are needed to keep that loss below
{thr_label(th[0])}, {thr_label(th[1])} or {thr_label(th[2])} money chance?</p>
<p>(For scale: {thr_label(th[0])} money chance is a 1-in-10,000 chance of the match, about 0.0008 stakes per
player per decision, since a match swings 8 stakes.)</p>
</div>

<h2>The short answer</h2>
<p>Deals needed for an expected loss of {thr_label(th[0])} per decision, with the 95% range in brackets:</p>
{summary}
{img(images, "overview-n-by-position.png", "Deals needed per position, both tricks")}
{img(images, "overview-predicted-loss.png", "Predicted loss against deals, by position")}
<ul>
{features_bullet}
<li>The loss shrinks like a power of N: <b>loss = a × N<sup>−b</sup></b>. Each scenario has its own size <i>a</i>;
one exponent <i>b</i> is shared by all scenarios of a trick, as asked. With b around 0.7–0.8, doubling the
deals cuts the loss by roughly 40–45%. (If b were 1, doubling would halve it.)</li>
{''.join(facts)}
<li>This is why the agreement report's plateau rule (fewer than 1% of choices change when the deals are
doubled) is never met on these grids: choices keep flipping between near-equal cards long after the money
at stake is negligible. The money-loss threshold is the useful yardstick.</li>
<li>Predictions for the stricter thresholds lie beyond the {t0['worlds']} deals actually measured, so their
ranges are wide. The extension tool (<code>search worlds --ids … --from-world {t0['worlds']}</code>) can measure
chosen scenarios at 640 or more deals to check them.</li>
</ul>

<h2>How the measurement works</h2>
<p><b>Re-using the saved deals.</b> Every decision was solved exactly in 320 deals. To see what a bot with N
deals would have done, we pick N of the 320 at random, let the bot choose with those, and repeat this
{t0['splits']} times. No new solving is needed, so any N up to 320 can be checked.</p>
<p><b>A fair referee.</b> To score a choice we need to know which card is truly best, and we only have 320
deals to judge with. If the same deals are used both to choose and to judge, the judge flatters the choice:
near 320 the "loss" drops to zero simply because the player and the judge saw the same deals (the grey
crosses in the charts show this). So each repetition splits the 320 deals in two halves. The bot's N-deal
choice and a careful 160-deal choice are both made from the first half, and both are scored on the second
half, which neither has seen. On average this measures exactly how much worse the N-deal choice is than the
160-deal choice, with no flattering. The fitted curve then adds back what the 160-deal choice itself still
loses, which gives the total loss against a perfect bot.</p>
<p><b>How sure we are.</b> The ranges come from re-drawing the set of decisions 1,000 times (keeping
the same mix of positions and match situations) and redoing every step. They cover both which
decisions happened to be picked and which deals happened to be drawn.</p>
<p><b>Checks.</b> (1) <i>Held out:</i> the curve is fitted only on N up to 80, and must predict N from 90 to
150. (2) <i>Extrapolation:</i> the whole recipe is rerun using only the first 160 deals of each decision
(so its own measurements stop at 80), and it must predict what the full 320 deals show, up to N = 150 — this
is the honest test of predicting beyond the measured range.</p>
{img(images, "overview-exponents.png", "Each group's own exponent against the shared one")}
<p class=sub>Each group's own exponent is uncertain because each group has only about 50–100 decisions.
{b_note}</p>

{features}

{rollout}

{''.join(sections)}

<h2>Method notes (for the record)</h2>
<ul>
<li>Choices are made exactly as the PIMC bot makes them (largest rounded fixed-point mean, ties to the first
card); losses are exact integer differences in units of 2<sup>−32</sup>; only the statistics are floating point.</li>
<li>Split-half reading: for split s, a seeded permutation of the M = {t0['worlds']} deals; the first N choose,
the first K = {t0['k']} make the reference choice, the last {t0['worlds'] - t0['k']} score both. Its expectation
is exactly L(N) − L(K), whatever the shape of the curve. The fitted model is a(N<sup>−b</sup> − K<sup>−b</sup>).</li>
<li>Fit: weighted least squares on log scale over N &le; K/2, each point weighted by its share of the log-N axis
divided by its squared relative standard error; b shared over the eight positions (profile search), a per
group; other groups get their own a with the trick's b. N for a threshold θ is (a/θ)<sup>1/b</sup>.</li>
<li>Uncertainty: stratified bootstrap of decisions within position × situation cells; every fit redone per
replicate; ranges are the 2.5% and 97.5% points.</li>
<li>Decisions are an equal mix of positions and match situations, not their natural frequencies; "all"
is the plain average over that mix.</li>
<li>Hand features (the 5 of trumps, robber, voids) are worked out from what the deciding player can see:
the robber flag comes from the public record of who burned a card (or must burn one now), never from hidden
cards. Records saved before these features existed get them by replaying each decision from its decision file
(<code>--decisions</code>), with no new solving. The comparisons between two groups are ratios of their N,
with ranges from the same bootstrap re-draws (paired).</li>
<li>Reproduce: <code>search curves --worlds exact-t3-n320.jsonl --decisions decisions-t3.jsonl --json
curves-t3.json</code> (likewise the other tricks); <code>search score --yardstick exact-t2-n320.jsonl --n 320 --method rollout-t2-l2-n64.jsonl:8 …:64
--json score-t2.json</code>; then <code>uv run python/tools/ncurves_report.py --out docs/m3-ncurves --score
score-t2.json curves-t2.json curves-t3.json curves-t4.json</code>. The text tables are in
<code>curves-t*.txt</code> and <code>score-t2.txt</code> next to this page.
Extend chosen decisions: <code>search worlds … --ids 3,17,42 --from-world 320 --to 640</code>, then pass both
worlds files to <code>search curves</code>.</li>
</ul>
</main></body></html>
"""
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "index.html"), "w") as f:
        f.write(page)
    print(f"wrote {a.out}/index.html and {len(images)} charts")


if __name__ == "__main__":
    main()
