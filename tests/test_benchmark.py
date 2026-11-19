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


def test_provenance_records_versions_and_the_protocol_constants():
    p = B.provenance()
    assert {"commit", "dirty", "python", "numpy", "scipy", "scikit-learn"} <= set(p)
    assert p["protocol"]["MEM"] == B.MEM and p["protocol"]["FLOOR"] == B.FLOOR
    assert p["protocol"]["test_splits"] == list(range(10))


def _methods(acc, forget, imm=None):
    return {
        "acc": acc,
        "forgetting": forget,
        "plasticity": acc,
        "newest": acc,
        "old": acc,
        "immediate": imm or [[0.9] * 5] * len(acc),
    }


def _test_section():
    names = [
        "ordinary",
        *[f"lr x{s}" for s in B.LR_SCALES[1:]],
        "strategy_b",
        "relaxed",
    ]
    base = [0.50, 0.60, 0.55]
    res = {}
    for i, n in enumerate(names):
        res[n] = _methods(
            [a + 0.01 * i for a in base],
            [0.30 - 0.02 * i, 0.40 - 0.02 * i, 0.35 - 0.02 * i],
        )
    return res


def _raw(selected="m[1,0.1,1,1]"):
    ctrl = {
        "base": _methods([0.5, 0.6, 0.55], [0.3, 0.4, 0.35]),
        "ctrl": _methods([0.5, 0.6, 0.56], [0.25, 0.4, 0.3]),
    }
    row = lambda floor_stat: {
        "acc": 0.7,
        "forgetting": 0.1,
        "min_immediate": floor_stat,
    }
    return {
        "provenance": {},
        "relaxation": {
            "selected": "relaxed|0.03|0.3",
            "config": {"strategy": "strategy_b", "n_ref": 50, "tol": 0.03, "beta": 0.3},
            "ordinary_forgetting": 0.053,
            "validation": {},
        },
        "test_mem60": _test_section(),
        "test_mem100": _test_section(),
        "degree_control": {
            "ordinary": {
                "validation": {"a": row(0.6), "b": row(0.4)},
                "selected": selected,
                "test": ctrl,
            },
            "strategy_b": {
                "validation": {"a": row(0.1), "b": row(0.2)},
                "selected": None,
                "test": None,
            },
        },
    }


def _capture(raw):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        B.report(raw)
    return buf.getvalue()


def test_report_reproduces_every_section_the_benchmark_prints():
    out = _capture(_raw())
    assert "[validation] relaxed selected: tol=0.03 beta=0.3 n_ref=50" in out
    for mem in (
        60,
        100,
    ):  # both memory sizes: method table, comparisons, frontier, verdict
        assert f"[test, memory {mem}/class" in out
    assert out.count("learning-rate frontier=") == 2 and out.count("verdict=") == 2
    assert out.count("strategy_b vs ordinary:") == 2
    assert "lr x0.25 " in out  # the learning-rate control rows
    assert "[degree control, memory 60/class" in out
    assert "ordinary: 1/2 validation candidates feasible; selected m[1,0.1,1,1]" in out
    assert "  test: d_acc=" in out and "min arrival acc" in out
    assert "strategy_b: 0/2 validation candidates feasible; selected None" in out


def test_report_prints_no_degree_control_test_when_nothing_was_selected():
    raw = _raw(selected=None)
    raw["degree_control"]["ordinary"]["test"] = None
    assert "  test: d_acc=" not in _capture(raw)


def test_report_paired_comparison_matches_paired_ci():
    raw = _raw()
    res = raw["test_mem60"]
    d = B.paired_ci(res["strategy_b"]["acc"], res["ordinary"]["acc"])
    assert f"strategy_b vs ordinary: d_acc={d[0]:+.3f}" in _capture(raw)


def test_report_round_trips_through_json():
    import json

    raw = _raw()
    assert _capture(json.loads(json.dumps(raw))) == _capture(raw)


def test_verdict_explanation_states_the_actual_bounds_and_which_criteria_hold():
    import re

    raw = _raw()
    ok = _capture(raw)
    assert re.search(
        r"verdict=VALUE \(acc CI lower bound [+-]\d\.\d{3} > -0\.01: yes; "
        r"forgetting CI upper bound [+-]\d\.\d{3} < 0: yes\)",
        ok,
    )
    # relaxed accuracy 0.05 below ordinary on every split: the accuracy criterion fails,
    # the forgetting criterion still holds, and the text must say so.
    raw["test_mem60"]["relaxed"]["acc"] = [
        a - 0.05 for a in raw["test_mem60"]["ordinary"]["acc"]
    ]
    bad = _capture(raw)
    assert re.search(
        r"verdict=no value \(acc CI lower bound -0\.050 > -0\.01: no; "
        r"forgetting CI upper bound [+-]\d\.\d{3} < 0: yes\)",
        bad,
    )


def test_main_output_without_the_footer_equals_report_exactly(monkeypatch, tmp_path):
    import contextlib
    import io
    import json

    def fake_table(configs, splits, seeds, mem, validation):
        n = len(list(splits))
        return {
            name: {
                "acc": [0.8] * n,
                "forgetting": [0.05 if name == "ordinary" else 0.02] * n,
                "plasticity": [0.8] * n,
                "newest": [0.5] * n,
                "old": [0.8] * n,
                "immediate": [[0.9] * 5] * n,
            }
            for name in configs
        }

    monkeypatch.setattr(B, "table", fake_table)
    out = tmp_path / "raw.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        B.main(str(out))
    lines = buf.getvalue().split("\n")
    assert lines[-2].startswith("raw results written to")  # footer is the last line
    live = "\n".join(lines[:-2]) + "\n"
    assert live == _capture(json.loads(out.read_text()))  # exact: blank lines included
