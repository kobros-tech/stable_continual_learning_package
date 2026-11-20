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

import argparse
import json
import platform
import subprocess
import sys

import numpy as np
import scipy
import sklearn
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


def provenance():
    """Metadata stored next to the raw results so a number can be traced to code and versions.

    Returns
    -------
    dict
        Git commit (``None`` outside a repository), dirty flag, Python/NumPy/SciPy/scikit-learn
        versions, and the protocol constants of this module.
    """

    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "protocol": {
            "ETA": ETA,
            "MEM": MEM,
            "FLOOR": FLOOR,
            "LR_SCALES": list(LR_SCALES),
            "FACTORS": list(FACTORS),
            "validation_splits": [100, 101, 102, 103],
            "test_splits": list(range(10)),
            "training_seeds": [1, 2],
        },
    }


def _print_test_section(mem, res):
    """Print one memory size of the test study: method table, paired comparisons, frontier."""
    means = {n: mean_row(r) for n, r in res.items()}
    lr_names = [f"lr x{s}" for s in LR_SCALES[1:]]
    lr = sorted(
        [(means[n]["forgetting"], means[n]["acc"]) for n in (*lr_names, "ordinary")]
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
        paired_ci(res["relaxed"][k], res["ordinary"][k]) for k in ("acc", "forgetting")
    )
    frontier = means["relaxed"]["acc"] - np.interp(
        means["relaxed"]["forgetting"], [x for x, _ in lr], [y for _, y in lr]
    )
    acc_ok, forget_ok = da[1] > -0.01, df[2] < 0
    yes = lambda ok: "yes" if ok else "no"
    print(
        f"relaxed: learning-rate frontier={frontier:+.3f} "
        f"verdict={'VALUE' if acc_ok and forget_ok else 'no value'} "
        f"(acc CI lower bound {da[1]:+.3f} > -0.01: {yes(acc_ok)}; "
        f"forgetting CI upper bound {df[2]:+.3f} < 0: {yes(forget_ok)})"
    )


def report(raw):
    """Print every comparison of the benchmark from a raw-results dictionary.

    This is the only code that prints results: :func:`main` builds the raw dictionary and then
    calls this function, so ``python benchmark.py --report FILE`` reproduces the console output
    of the run that wrote ``FILE`` (same lines, same order) without recomputing any model.

    Parameters
    ----------
    raw : dict
        Contents of the JSON written by ``python benchmark.py --out FILE``: the validation
        selection of the relaxation (``relaxation``), per-split test metrics for each memory size
        (``test_mem60``, ``test_mem100``), and the degree-control study (``degree_control``:
        for each base update its validation rows, the selected candidate or ``None``, and the
        per-split test metrics when a candidate was selected).
    """
    rel = raw["relaxation"]
    print(
        f"[validation] relaxed selected: tol={rel['config']['tol']} beta={rel['config']['beta']} "
        f"n_ref={rel['config']['n_ref']}; ordinary forgetting={rel['ordinary_forgetting']:.3f}"
    )
    for mem in (60, 100):
        if f"test_mem{mem}" in raw:
            _print_test_section(mem, raw[f"test_mem{mem}"])
    dc = raw.get("degree_control")
    if dc:
        print(f"\n[degree control, memory {MEM}/class, acquisition floor {FLOOR}]")
        for bname, entry in dc.items():
            v = entry["validation"]
            feasible = sum(m["min_immediate"] >= FLOOR for m in v.values())
            print(
                f"{bname}: {feasible}/{len(v)} validation candidates feasible; "
                f"selected {entry['selected']}"
            )
            res = entry["test"]
            if res is None:
                continue
            da, df = (
                paired_ci(res["ctrl"][k], res["base"][k]) for k in ("acc", "forgetting")
            )
            print(
                f"  test: d_acc={da[0]:+.3f} [{da[1]:+.3f},{da[2]:+.3f}] "
                f"d_forget={df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}] "
                f"min arrival acc {mean_row(res['base'])['min_immediate']:.3f} -> "
                f"{mean_row(res['ctrl'])['min_immediate']:.3f}"
            )


def main(out=None):
    """Run the complete benchmark, then print the tables with :func:`report`.

    1. Tune the relaxation on validation views: among the 12 (tol, beta) candidates whose
       validation forgetting is at most 70 % of ordinary SGD's, take the highest accuracy.
    2. Test, for memory 60 and 100: ordinary SGD, smaller-learning-rate controls, exact
       ``strategy_b`` and the selected relaxation, with paired 95 % intervals over 10 splits.
    3. Degree control on top of each base update: select on validation (see :func:`select`),
       then compare with and without the selected control on the test splits.

    Parameters
    ----------
    out : str, optional
        Path of a JSON file for the raw per-split results and :func:`provenance`.  Everything
        printed is produced by :func:`report` from that same dictionary, so the file is enough
        to reproduce the printed output.
    """
    raw = {"provenance": provenance()}

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
    raw["relaxation"] = {
        "selected": name,
        "config": relaxed,
        "ordinary_forgetting": float(base_forget),
        "validation": val,
    }

    # 2. test: ordinary, learning-rate controls, exact strategy_b, relaxed
    for mem in (60, 100):
        configs = {"ordinary": ({}, 1.0, None)}
        configs.update({f"lr x{s}": ({}, s, None) for s in LR_SCALES[1:]})
        configs.update(
            {"strategy_b": (STRATEGY_B, 1.0, None), "relaxed": (relaxed, 1.0, None)}
        )
        raw[f"test_mem{mem}"] = table(configs, range(10), (1, 2), mem, False)

    # 3. degree control on top of each base update: validation selection, then paired test
    bases = {"ordinary": {}, "strategy_b": STRATEGY_B, "relaxed": relaxed}
    raw["degree_control"] = {}
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
        pick = select(v)
        test = None
        if pick is not None and grid[pick] is not None:
            test = table(
                {"base": (cfg, 1.0, None), "ctrl": (cfg, 1.0, grid[pick])},
                range(10),
                (1, 2),
                MEM,
                False,
            )
        raw["degree_control"][bname] = {
            "validation": v,
            "selected": pick,
            "test": test,
        }

    report(raw)
    if out is not None:
        with open(out, "w") as fh:
            json.dump(raw, fh, indent=1)
        print(f"raw results written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out", help="write raw per-split results and provenance to this JSON file"
    )
    ap.add_argument(
        "--report", help="print paired comparisons from a raw JSON file and exit"
    )
    args = ap.parse_args()
    if args.report:
        with open(args.report) as fh:
            report(json.load(fh))
        sys.exit(0)
    main(args.out)
