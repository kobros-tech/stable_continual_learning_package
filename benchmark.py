# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Validation-tuned benchmark: ordinary SGD, strategy_b, its relaxation, and diagonal degree control.

Run: python benchmark.py
All selection uses validation views (splits 100-103 cut from the training part); test numbers come
from 10 independent splits x 2 seeds and are compared pairwise. The degree-control study follows
the selection rule of the PR (acquisition floor, Pareto front, accuracy - forgetting) for
single-degree candidates only; the PR's combination step and Split-MNIST study are not repeated.
"""

import numpy as np
from scipy import stats
from sklearn.datasets import load_digits

import stable_continual_learning as S

ETA, MEM = 3.0, 60
METRICS = ("acc", "forgetting", "plasticity", "newest", "old")
STRATEGY_B = {"strategy": "strategy_b"}  # exact: n_ref=3, tol=1e-10, beta=0
RELAXED = [
    {"strategy": "strategy_b", "n_ref": 50, "tol": tol, "beta": beta}
    for tol in (0.3, 0.1, 0.03, 0.01)
    for beta in (0.0, 0.1, 0.3)
]
LR_SCALES = (1.0, 0.75, 0.5, 0.35, 0.25)
FACTORS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75)  # degree-control factors below 1
FLOOR = 0.5  # acquisition floor, fixed in advance

FUNCTIONAL_LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
OLD_ACCURACY_MARGIN = 0.01
FORGETTING_MARGIN = 0.01


def load(split_seed, val=False):
    d = load_digits()
    order = np.random.default_rng(split_seed).permutation(len(d.data))
    cut = int(0.75 * len(order))
    tr, te = order[:cut], order[cut:]
    x = d.data / 16.0
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    x = (x - mu) / sd
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(64, 16)))
    Phi, deg = S.poly_features(x @ Q)
    if val:
        k = int(0.8 * len(tr))
        tr, te = tr[:k], tr[k:]
    return Phi[tr], d.target[tr], Phi[te], d.target[te], deg


def evaluate(data, cfg, seeds, mem, scale=1.0, control=None):
    rs = [
        S.run(*data, seed, [ETA * scale] * 4, mem=mem, control=control, **cfg)
        for seed in seeds
    ]
    out = {k: float(np.mean([r[k] for r in rs])) for k in METRICS}
    out["immediate"] = np.mean(
        [r["immediate"] for r in rs], axis=0
    ).tolist()  # per experience
    return out


def table(configs, splits, seeds, mem, validation):
    """configs: name -> (cfg, lr_scale, control); returns name -> metric -> per-split list."""
    out = {n: {k: [] for k in (*METRICS, "immediate")} for n in configs}
    for split in splits:
        data = load(split, validation)
        for name, (cfg, scale, control) in configs.items():
            for key, value in evaluate(data, cfg, seeds, mem, scale, control).items():
                out[name][key].append(value)
    return out


def paired_ci(a, b):
    diff = np.asarray(a) - np.asarray(b)
    half = stats.t.ppf(0.975, len(diff) - 1) * stats.sem(diff)
    return diff.mean(), diff.mean() - half, diff.mean() + half


def pareto(m):
    """Names whose (acc, forgetting) no other candidate beats on both."""
    return [
        n
        for n, a in m.items()
        if not any(
            b["acc"] >= a["acc"]
            and b["forgetting"] <= a["forgetting"]
            and (b["acc"] > a["acc"] or b["forgetting"] < a["forgetting"])
            for o, b in m.items()
            if o != n
        )
    ]


def select(m, floor=FLOOR):
    """PR rule: every experience's seed-mean accuracy on arrival >= floor; then Pareto front of
    (acc, forgetting), then max acc - forgetting. None if nothing is feasible."""
    feasible = {n: v for n, v in m.items() if v["min_immediate"] >= floor}
    if not feasible:
        return None
    return max(
        pareto(feasible), key=lambda n: feasible[n]["acc"] - feasible[n]["forgetting"]
    )


def controls():
    """All-ones baseline plus one degree at a time with each factor below 1."""
    out = {"m[1,1,1,1]": None}
    for g in range(S.N_GROUPS):
        for f in FACTORS:
            v = [1.0] * S.N_GROUPS
            v[g] = f
            out["m[" + ",".join(f"{x:g}" for x in v) + "]"] = v
    return out


def mean_row(res):
    """Means over splits; min_immediate = min over experiences of the mean (over seeds and splits)
    accuracy right after that experience, the quantity the acquisition floor constrains."""
    out = {k: float(np.mean(v)) for k, v in res.items() if k != "immediate"}
    out["min_immediate"] = float(np.mean(res["immediate"], axis=0).min())
    return out

def functional_regularization_study():
    """Tune functional regularization on validation data, then test the selected value."""
    for mem in (60, 100):
        configs = {
            f"lambda={lam:g}": (
                {"functional_lambda": lam},
                1.0,
                None,
            )
            for lam in FUNCTIONAL_LAMBDAS
        }

        validation = table(
            configs,
            (100, 101, 102, 103),
            (1, 2),
            mem,
            True,
        )
        means = {name: mean_row(result) for name, result in validation.items()}
        baseline = means["lambda=0"]

        feasible = {
            name: result
            for name, result in means.items()
            if result["old"] >= baseline["old"] - OLD_ACCURACY_MARGIN
            and result["forgetting"]
            <= baseline["forgetting"] + FORGETTING_MARGIN
        }

        print(f"\n[functional regularization, memory {mem}/class]")
        print("lambda       acc    forget  plasticity  newest  old")
        for name, result in means.items():
            print(
                f"{name:12s} {result['acc']:.3f}  "
                f"{result['forgetting']:.3f}   "
                f"{result['plasticity']:.3f}     "
                f"{result['newest']:.3f}  {result['old']:.3f}"
            )

        if not feasible:
            print("No feasible validation candidate.")
            continue

        selected = max(feasible, key=lambda name: feasible[name]["newest"])
        print(f"Selected on validation: {selected}")

        test = table(
            {
                "baseline": ({"functional_lambda": 0.0}, 1.0, None),
                "regularized": (configs[selected][0], 1.0, None),
            },
            range(10),
            (1, 2),
            mem,
            False,
        )

        print("[held-out test: regularized minus baseline]")
        for metric in ("acc", "newest", "old", "forgetting"):
            diff = paired_ci(
                test["regularized"][metric],
                test["baseline"][metric],
            )
            print(
                f"{metric:10s} {diff[0]:+.3f} "
                f"[{diff[1]:+.3f}, {diff[2]:+.3f}]"
            )


def main():
    # 1. tune only the relaxation on validation views
    cand = {"ordinary": ({}, 1.0, None), "strategy_b": (STRATEGY_B, 1.0, None)}
    cand.update({f"relaxed|{c['tol']}|{c['beta']}": (c, 1.0, None) for c in RELAXED})
    val = table(cand, (100, 101, 102, 103), (1, 2), MEM, True)
    base_forget = np.mean(val["ordinary"]["forgetting"])
    eligible = [
        (np.mean(val[n]["acc"]), n)
        for n in cand
        if n.startswith("relaxed|")
        if np.mean(val[n]["forgetting"]) <= 0.7 * base_forget
    ]
    if not eligible:
        raise RuntimeError(
            "No relaxed configuration met the predeclared validation forgetting target."
        )
    name = max(eligible)[1]
    relaxed = cand[name][0]
    print(
        f"[validation] relaxed selected: tol={relaxed['tol']} beta={relaxed['beta']} "
        f"n_ref={relaxed['n_ref']}; ordinary forgetting={base_forget:.3f}"
    )

    # 2. test: ordinary, learning-rate controls, exact strategy_b, relaxed
    for mem in (60, 100):
        configs = {"ordinary": ({}, 1.0, None)}
        configs.update({f"lr x{s}": ({}, s, None) for s in LR_SCALES[1:]})
        configs.update(
            {"strategy_b": (STRATEGY_B, 1.0, None), "relaxed": (relaxed, 1.0, None)}
        )
        res = table(configs, range(10), (1, 2), mem, False)
        means = {n: mean_row(r) for n, r in res.items()}
        lr = sorted(
            [
                (means[f"lr x{s}"]["forgetting"], means[f"lr x{s}"]["acc"])
                for s in LR_SCALES[1:]
            ]
            + [(means["ordinary"]["forgetting"], means["ordinary"]["acc"])]
        )
        print(f"\n[test, memory {mem}/class, 10 splits x 2 seeds]")
        print("method       acc    forget  plasticity  newest  old")
        for n, r in means.items():
            print(
                f"{n:12s} {r['acc']:.3f}  {r['forgetting']:.3f}   {r['plasticity']:.3f}     "
                f"{r['newest']:.3f}  {r['old']:.3f}"
            )
        for a, b in (
            ("strategy_b", "ordinary"),
            ("relaxed", "ordinary"),
            ("relaxed", "strategy_b"),
        ):
            da, df = (paired_ci(res[a][k], res[b][k]) for k in ("acc", "forgetting"))
            print(
                f"{a} vs {b}: d_acc={da[0]:+.3f} [{da[1]:+.3f},{da[2]:+.3f}] "
                f"d_forget={df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}]"
            )
        da, df = (
            paired_ci(res["relaxed"][k], res["ordinary"][k])
            for k in ("acc", "forgetting")
        )
        frontier = means["relaxed"]["acc"] - np.interp(
            means["relaxed"]["forgetting"], [x for x, _ in lr], [y for _, y in lr]
        )
        print(
            f"relaxed: learning-rate frontier={frontier:+.3f} verdict="
            f"{'VALUE' if da[1] > -0.01 and df[2] < 0 else 'no value'} "
            "(acc CI lower bound > -0.01 and forgetting CI upper bound < 0)"
        )

    # 3. degree control on top of each base update: validation selection, then paired test
    bases = {"ordinary": {}, "strategy_b": STRATEGY_B, "relaxed": relaxed}
    print(f"\n[degree control, memory {MEM}/class, acquisition floor {FLOOR}]")
    for bname, cfg in bases.items():
        grid = controls()
        v = {
            n: mean_row(r)
            for n, r in table(
                {n: (cfg, 1.0, c) for n, c in grid.items()},
                (100, 101, 102, 103),
                (1, 2),
                MEM,
                True,
            ).items()
        }
        feasible = sum(m["min_immediate"] >= FLOOR for m in v.values())
        pick = select(v)
        print(
            f"{bname}: {feasible}/{len(v)} validation candidates feasible; selected {pick}"
        )
        if pick is None or grid[pick] is None:
            continue
        res = table(
            {"base": (cfg, 1.0, None), "ctrl": (cfg, 1.0, grid[pick])},
            range(10),
            (1, 2),
            MEM,
            False,
        )
        da, df = (
            paired_ci(res["ctrl"][k], res["base"][k]) for k in ("acc", "forgetting")
        )
        print(
            f"  test: d_acc={da[0]:+.3f} [{da[1]:+.3f},{da[2]:+.3f}] "
            f"d_forget={df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}] "
            f"min arrival acc {mean_row(res['base'])['min_immediate']:.3f} -> "
            f"{mean_row(res['ctrl'])['min_immediate']:.3f}"
        )

    functional_regularization_study()


if __name__ == "__main__":
    main()
