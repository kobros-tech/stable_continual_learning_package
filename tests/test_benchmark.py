# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Benchmark statistics and selection rules (the full benchmark itself is not run here)."""

import numpy as np
import pytest
from scipy import stats

import benchmark as B
from stable_cl import load_digits_features


def test_paired_ci_matches_the_t_interval_of_the_differences():
    a = np.array([0.50, 0.55, 0.61, 0.48, 0.52])
    b = np.array([0.45, 0.50, 0.60, 0.40, 0.55])
    mean, lo, hi = B.paired_ci(a, b)
    diff = a - b
    half = stats.t.ppf(0.975, 4) * diff.std(ddof=1) / np.sqrt(5)
    assert np.isclose(mean, diff.mean())
    assert np.isclose(lo, diff.mean() - half) and np.isclose(hi, diff.mean() + half)


def test_paired_ci_of_identical_methods_is_degenerate_at_zero():
    a = np.array([0.4, 0.5, 0.6])
    mean, lo, hi = B.paired_ci(a, a)
    assert mean == lo == hi == 0.0


def test_pareto_keeps_exactly_the_non_dominated_candidates():
    m = {
        "a": {"acc": 0.8, "forgetting": 0.2},
        "b": {"acc": 0.7, "forgetting": 0.1},  # lower forgetting: not dominated
        "c": {"acc": 0.6, "forgetting": 0.3},  # dominated by a
        "d": {"acc": 0.8, "forgetting": 0.2},  # tie with a: neither dominates
    }
    assert B.pareto(m) == ["a", "b", "d"]


def row(acc, forgetting, floor_stat):
    return {"acc": acc, "forgetting": forgetting, "min_immediate": floor_stat}


def test_select_applies_floor_then_pareto_then_acc_minus_forgetting():
    m = {
        "infeasible_but_best": row(0.99, 0.0, 0.4),
        "x": row(0.80, 0.30, 0.6),  # acc - forgetting = 0.50
        "y": row(0.75, 0.10, 0.6),  # 0.65, on the front
        "z": row(0.70, 0.20, 0.6),  # dominated by y
    }
    assert B.select(m) == "y"


def test_select_returns_none_when_nothing_meets_the_floor():
    assert B.select({"a": row(0.9, 0.1, 0.49), "b": row(0.8, 0.0, 0.1)}) is None


def test_controls_grid_has_baseline_plus_one_degree_at_a_time():
    grid = B.controls()
    assert len(grid) == 1 + 4 * len(B.FACTORS) == 25
    assert grid["m[1,1,1,1]"] is None
    assert grid["m[1,0,1,1]"] == [1.0, 0.0, 1.0, 1.0]
    assert all(sum(f != 1.0 for f in v) == 1 for v in grid.values() if v is not None)


def test_mean_row_takes_the_minimum_after_averaging_over_splits():
    res = {
        "acc": [0.6, 0.8],
        "forgetting": [0.1, 0.3],
        # split 1 is worst on experience 0, split 2 on experience 1:
        # mean of per-split minima would be 0.2; minimum of the means is 0.35
        "immediate": [[0.2, 0.9], [0.9, 0.5]],
    }
    r = B.mean_row(res)
    assert np.isclose(r["acc"], 0.7) and np.isclose(r["forgetting"], 0.2)
    assert np.isclose(r["min_immediate"], 0.55)  # means are [0.55, 0.7]
    assert "immediate" not in r


def test_load_delegates_to_the_shared_leakage_free_loader():
    a = B.load(101, val=True)
    b = load_digits_features(101, validation=True)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_evaluate_averages_seeds_and_reports_arrival_accuracy(toy):
    out = B.evaluate(toy, {}, (0, 1), 20)
    assert set(out) == {*B.METRICS, "immediate"} and len(out["immediate"]) == 5
    assert all(0.0 <= out[k] <= 1.0 for k in B.METRICS)


def test_table_returns_one_entry_per_split_for_every_configuration(monkeypatch, toy):
    monkeypatch.setattr(B, "load", lambda split, validation: toy)
    cfgs = {"ordinary": ({}, 1.0, None), "sb": ({"strategy": "strategy_b"}, 1.0, None)}
    out = B.table(cfgs, (0, 1, 2), (0,), 20, False)
    assert set(out) == set(cfgs)
    assert all(len(out[n]["acc"]) == 3 and len(out[n]["immediate"]) == 3 for n in cfgs)


def test_table_accepts_a_generator_of_seeds_for_every_configuration_and_split(
    monkeypatch, toy
):
    monkeypatch.setattr(B, "load", lambda split, validation: toy)
    cfgs = {"ordinary": ({}, 1.0, None), "sb": ({"strategy": "strategy_b"}, 1.0, None)}
    from_tuple = B.table(cfgs, (0, 1), (0, 1), 20, False)
    from_generator = B.table(cfgs, (0, 1), (s for s in (0, 1)), 20, False)
    assert (
        from_generator == from_tuple
    )  # later splits/configs must still see both seeds


def test_table_rejects_empty_seeds(monkeypatch, toy):
    monkeypatch.setattr(B, "load", lambda split, validation: toy)
    with pytest.raises(ValueError, match="seeds"):
        B.table({"ordinary": ({}, 1.0, None)}, (0,), iter(()), 20, False)
