"""Can anything the student can see find the decisions where it loses? (path-forward.md, session 0b)

Reads the JSON of `search student-check --sub-deals ... --json`, fits a gradient-boosted risk model on
the student's own values (margins, spread, candidates, place, stratum) with 5-fold cross-validation
(every decision is scored by a model that never saw it), and prints the router curve: search the
f% of decisions with the highest predicted loss with top-3 + N deals, keep the student's card for
the rest. Compared with the margin router and the oracle router (true loss).

    python tools/router_fit.py student-check.json [--seeds 5]
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

ONE = 2.0**32
ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
j = json.load(open(a.path))
pd = j["sub_deals"]["per_decision"]
grid = j["sub_deals_grid"]
reps = j["sub_deals_repeats"]
n = len(pd)
worlds = 320


def feats(d):
    v = d["values"] + [d["values"][-1]] * 3
    return [d["place"], d["stratum"], d["n_cands"], (v[0] - v[1]) / ONE, (v[1] - v[2]) / ONE,
            (v[0] - v[-1]) / ONE, v[0] / ONE, v[1] / ONE]


X = np.array([feats(d) for d in pd])
loss = np.array([d["loss"] for d in pd], dtype=float) / ONE
margin = X[:, 3]
F = [0, 5, 10, 20, 30, 50, 75, 100]
rng = np.random.default_rng(1)
print(f"{n} decisions, {int((loss > 0).sum())} with a loss; student {100*loss.mean():.4f}%")

pred = np.zeros((a.seeds, n))
for s in range(a.seeds):
    for tr, te in KFold(5, shuffle=True, random_state=s).split(X):
        m = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=s)
        m.fit(X[tr], loss[tr] * 100)
        pred[s, te] = m.predict(X[te])
rho = [np.corrcoef(p, loss)[0, 1] for p in pred]
print(f"correlation of predicted and true loss (cross-validated): {np.mean(rho):.3f}")
print(f"margin vs true loss: {np.corrcoef(-margin, loss)[0, 1]:.3f}")
# the share of the true loss in the predicted-riskiest 10/20/50%
for f in (10, 20, 50):
    k = int(np.ceil(n * f / 100))
    held = [loss[np.argsort(-p, kind="stable")[:k]].sum() / loss.sum() for p in pred]
    mh = loss[np.argsort(margin, kind="stable")[:k]].sum() / loss.sum()
    print(f"predicted-riskiest {f}% hold {100*np.mean(held):.1f}% of the loss (margin-lowest {100*mh:.1f}%)")

for gi, N in enumerate(grid):
    den = reps * (worlds - N) * worlds * ONE
    top3 = np.array([int(d["top3_corrected"][gi]) for d in pd]) / den
    print(f"\n=== {N} deals: loss % of money chance when the f% are searched (top-3 + {N} deals) ===")
    print(f"{'f':>4} {'model':>8} {'margin':>8} {'oracle':>8}   (boot 95% of model)")
    for f in F:
        k = int(np.ceil(n * f / 100))

        def cost(order):
            on = np.zeros(n, bool)
            on[order[:k]] = True
            return np.where(on, top3, loss)

        res = [cost(np.argsort(-p, kind="stable")).mean() for p in pred]
        mo = cost(np.argsort(margin, kind="stable")).mean()
        oo = cost(np.argsort(-loss, kind="stable")).mean()
        # bootstrap over decisions of the seed-averaged model router
        on_avg = np.mean([np.isin(np.arange(n), np.argsort(-p, kind="stable")[:k]) for p in pred], axis=0) > 0.5
        v = np.where(on_avg, top3, loss)
        bs = np.sort([v[rng.integers(0, n, n)].mean() for _ in range(1000)])
        print(f"{f:>3}% {100*np.mean(res):8.4f} {100*mo:8.4f} {100*oo:8.4f}   [{100*bs[50]:.4f}, {100*bs[949]:.4f}]")
