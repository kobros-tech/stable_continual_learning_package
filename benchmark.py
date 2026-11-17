# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Validation-tuned benchmark: ordinary SGD, strategy_b, its relaxation, and diagonal degree control.

Run: python benchmark.py
All selection uses validation views (splits 100-103 cut class-stratified from the training part,
standardised with statistics of the fitting rows only); test numbers come from 10 independent splits x 2 seeds and
are compared pairwise. The degree-control study follows
the selection rule of the PR (acquisition floor, Pareto front, accuracy - forgetting) for
single-degree candidates only; the PR's combination step and Split-MNIST study are not repeated.
"""

import numpy as np
from scipy import stats

import stable_cl as S

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


def load(split_seed, val=False):
    r"""Data for one split: train/test, or fit/validation when ``val`` is true.

    Thin wrapper around :func:`stable_cl.experiment.load_digits_features`.  In validation mode the
    training part is cut 80/20, stratified by class, into fitting and validation rows *before*
    the standardisation statistics are computed, and the statistics use the fitting rows only, so validation examples
    never influence preprocessing.

    Parameters
    ----------
    split_seed : int
        Seed of the split permutation (100-103 for validation, 0-9 for test).
    val : bool
        Validation view instead of the test split.

    Returns
    -------
    tuple
        ``(Phi_fit, y_fit, Phi_eval, y_eval, deg)``.
    """
    return S.load_digits_features(split_seed, validation=val)


def evaluate(data, cfg, seeds, mem, scale=1.0, control=None):
    r"""Average :func:`stable_cl.experiment.run` over training seeds on one split.

    For each seed in ``seeds`` runs the five experiences with base rates
    :math:`\eta = \text{ETA}\cdot\text{scale}` for every degree group, then returns the seed-mean
    of each metric in ``METRICS`` and of the arrival accuracies :math:`A_{t,t}` (``immediate``).

    Parameters
    ----------
    data : tuple
        Output of :func:`load`.
    cfg : dict
        Keyword arguments for ``run`` (``strategy``, ``n_ref``, ``tol``, ``beta``).
    seeds : iterable of int
        Training seeds.
    mem : int
        Replay memory per class.
    scale : float
        Multiplier of the base rate ``ETA`` (learning-rate controls).
    control : list of float, optional
        Degree-control factors.

    Returns
    -------
    dict
        Metric name -> float; ``immediate`` -> list of 5 floats.
    """
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
    """Evaluate every configuration on every split.

    Parameters
    ----------
    configs : dict
        ``name -> (cfg, lr_scale, control)``, see :func:`evaluate`.
    splits : iterable of int
        Split seeds; each split is loaded once and shared by all configurations (paired design).
    seeds : iterable of int
        Training seeds averaged within each split; must be non-empty.  Materialised once, so a
        generator is safe.
    mem : int
        Replay memory per class.
    validation : bool
        Use validation views (:func:`load` with ``val=True``).

    Returns
    -------
    dict
        ``name -> metric -> list`` with one entry per split (each already a mean over seeds).

    Raises
    ------
    ValueError
        If ``seeds`` is empty.
    """
    seeds = tuple(seeds)  # an iterator would be exhausted by the first evaluate() call
    if not seeds:
        raise ValueError("seeds must contain at least one training seed")
    out = {n: {k: [] for k in (*METRICS, "immediate")} for n in configs}
    for split in splits:
        data = load(split, validation)
        for name, (cfg, scale, control) in configs.items():
            for key, value in evaluate(data, cfg, seeds, mem, scale, control).items():
                out[name][key].append(value)
    return out


def paired_ci(a, b):
    r"""Paired 95 % confidence interval for the mean difference ``a - b`` across splits.

    With per-split differences :math:`\delta_s = a_s - b_s`, :math:`s = 1..n`,

    .. math::

        \bar\delta \pm t_{0.975,\,n-1}\,\frac{s_\delta}{\sqrt{n}} .

    Each :math:`a_s, b_s` is already the mean over the training seeds of split :math:`s`, so the
    interval reflects variability *between splits* after seed-averaging; it does not separately
    estimate training-seed variability.

    Parameters
    ----------
    a, b : array_like, shape (n,)
        Per-split values of two methods, aligned by split.

    Returns
    -------
    tuple of float
        ``(mean, lower, upper)``.
    """
    diff = np.asarray(a) - np.asarray(b)
    half = stats.t.ppf(0.975, len(diff) - 1) * stats.sem(diff)
    return diff.mean(), diff.mean() - half, diff.mean() + half


def pareto(m):
    r"""Names of the Pareto-optimal candidates in (accuracy, forgetting).

    Candidate :math:`b` dominates :math:`a` iff
    :math:`\text{acc}_b \ge \text{acc}_a`, :math:`\text{forget}_b \le \text{forget}_a`, and at least
    one inequality is strict.  The result is the set of non-dominated candidates.

    Parameters
    ----------
    m : dict
        ``name -> {"acc": float, "forgetting": float, ...}``.

    Returns
    -------
    list of str
        Non-dominated names, in the iteration order of ``m``.
    """
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
    r"""Validation selection rule of the PR for degree control.

    1. Feasible: :math:`\min_t \overline{A_{t,t}} \ge \text{floor}` (``min_immediate``, the mean
       over seeds and splits taken before the minimum over experiences).
    2. Among feasible candidates keep the Pareto front of (accuracy, forgetting), see :func:`pareto`.
    3. Return the front member maximising :math:`\text{acc} - \text{forgetting}`.

    Parameters
    ----------
    m : dict
        ``name -> row`` as returned by :func:`mean_row` (needs ``acc``, ``forgetting``,
        ``min_immediate``).
    floor : float
        Acquisition floor, fixed in advance.

    Returns
    -------
    str or None
        Selected name, or ``None`` if no candidate is feasible.
    """
    feasible = {n: v for n, v in m.items() if v["min_immediate"] >= floor}
    if not feasible:
        return None
    return max(
        pareto(feasible), key=lambda n: feasible[n]["acc"] - feasible[n]["forgetting"]
    )


def controls():
    """Candidate degree-control vectors: the all-ones baseline plus one degree at a time.

    Returns
    -------
    dict
        Name ``"m[...]"`` -> list of 4 factors (``None`` for the all-ones baseline); one entry
        for every degree group and every factor in ``FACTORS``, with the other factors 1.
    """
    out = {"m[1,1,1,1]": None}
    for g in range(S.N_GROUPS):
        for f in FACTORS:
            v = [1.0] * S.N_GROUPS
            v[g] = f
            out["m[" + ",".join(f"{x:g}" for x in v) + "]"] = v
    return out


def mean_row(res):
    r"""Summarise per-split results.

    Returns the mean over splits of each metric and ``min_immediate``
    :math:`= \min_t \overline{A_{t,t}}`, the minimum over experiences of the arrival accuracy
    averaged over seeds and splits (the quantity the acquisition floor constrains).

    Parameters
    ----------
    res : dict
        One entry of :func:`table`'s output: metric -> per-split list.

    Returns
    -------
    dict
        Metric -> float, plus ``min_immediate``.
    """
    out = {k: float(np.mean(v)) for k, v in res.items() if k != "immediate"}
    out["min_immediate"] = float(np.mean(res["immediate"], axis=0).min())
    return out


def main():
    """Run the complete benchmark and print the tables.

    1. Tune the relaxation on validation views: among the 12 (tol, beta) candidates whose
       validation forgetting is at most 70 % of ordinary SGD's, take the highest accuracy.
    2. Test, for memory 60 and 100: ordinary SGD, smaller-learning-rate controls, exact
       ``strategy_b`` and the selected relaxation, with paired 95 % intervals over 10 splits.
    3. Degree control on top of each base update: select on validation (see :func:`select`),
       then compare with and without the selected control on the test splits.
    """
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


if __name__ == "__main__":
    main()
