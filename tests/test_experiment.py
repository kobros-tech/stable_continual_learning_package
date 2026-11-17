# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Incremental protocol, metric definitions and leakage-free data preparation."""

import numpy as np
import pytest

import stable_cl
from stable_cl import load_digits_features, run
from stable_cl.experiment import TASKS


@pytest.mark.parametrize("kw", [{}, {"strategy": "strategy_b"}])
def test_all_ones_control_reproduces_the_uncontrolled_run_exactly(toy, kw):
    a = run(*toy, 0, 3.0, mem=20, **kw)["acc_matrix"]
    b = run(*toy, 0, 3.0, mem=20, control=[1.0] * 4, **kw)["acc_matrix"]
    assert np.array_equal(a, b, equal_nan=True)


def test_all_zero_control_is_a_no_op_with_no_null_basis(toy):
    r = run(*toy, 0, 3.0, mem=20, control=[0.0] * 4, strategy="strategy_b")
    assert r["free_dim"] == [None] * 5 and r["null_dim"] == [None] * 5
    assert r["effective_rates"] == [0.0] * 4


def test_run_is_deterministic_for_a_seed(toy):
    a = run(*toy, 3, 3.0, mem=20)["acc_matrix"]
    b = run(*toy, 3, 3.0, mem=20)["acc_matrix"]
    assert np.array_equal(a, b, equal_nan=True)


def test_metrics_follow_their_definitions_from_the_accuracy_matrix(toy):
    r = run(*toy, 1, 3.0, mem=20)
    A = r["acc_matrix"]
    assert np.isnan(A[np.triu_indices(5, 1)]).all()  # no entry for unseen experiences
    assert not np.isnan(A[np.tril_indices(5)]).any()
    assert np.isclose(r["acc"], A[4].mean())
    assert np.isclose(r["newest"], A[4, 4])
    assert np.isclose(r["old"], A[4, :4].mean())
    assert np.isclose(r["plasticity"], np.diag(A).mean())
    forgetting = np.mean(
        [max(A[t, j] for t in range(j, 4)) - A[4, j] for j in range(4)]
    )
    assert np.isclose(r["forgetting"], forgetting)
    assert r["immediate"] == np.diag(A).tolist()


def test_effective_rates_report_the_per_degree_rates(toy):
    r = run(*toy, 0, [3.0, 2.0, 1.0, 0.5], mem=20, control=[1, 0.5, 0, 1])
    assert r["effective_rates"] == [3.0, 1.0, 0.0, 0.5]


def test_tasks_partition_the_ten_classes():
    assert sorted(c for t in TASKS for c in t) == list(range(10))


# ----------------------------------------------------------------- data preparation / leakage
def _patched_digits(monkeypatch, scale_rows):
    """Make load_digits return data whose rows ``scale_rows`` are multiplied by 1000."""
    import sklearn.datasets as ds

    real = ds.load_digits()

    class Fake:
        data = real.data.copy()
        target = real.target

    Fake.data[scale_rows] *= 1000.0
    monkeypatch.setattr(ds, "load_digits", lambda: Fake)


def _indices(split_seed, validation):
    order = np.random.default_rng(split_seed).permutation(1797)
    fit, ev = order[: int(0.75 * 1797)], order[int(0.75 * 1797) :]
    if validation:
        k = int(0.8 * len(fit))
        fit, ev = fit[:k], fit[k:]
    return fit, ev


@pytest.mark.parametrize("validation", [False, True])
def test_evaluation_rows_do_not_influence_preprocessing(monkeypatch, validation):
    clean = load_digits_features(5, validation)
    _, ev = _indices(5, validation)
    _patched_digits(monkeypatch, ev)
    dirty = load_digits_features(5, validation)
    assert np.array_equal(clean[0], dirty[0])  # fitting features untouched
    assert not np.array_equal(
        clean[2], dirty[2]
    )  # the perturbation did reach the eval rows


def test_validation_view_is_cut_from_the_training_part_only():
    fit, ev = _indices(3, True)
    tr, te = _indices(3, False)
    assert set(fit) | set(ev) == set(tr) and not set(fit) & set(ev)
    assert not (set(fit) | set(ev)) & set(te)
    Phi_f, y_f, Phi_e, y_e, deg = load_digits_features(3, True)
    assert (len(y_f), len(y_e)) == (len(fit), len(ev)) == (1077, 270)
    assert Phi_f.shape[1] == Phi_e.shape[1] == len(deg) == 969


def test_test_protocol_sizes():
    _, y_tr, _, y_te, _ = load_digits_features(0)
    assert (len(y_tr), len(y_te)) == (1347, 450)


def test_package_exports_every_documented_name():
    for name in stable_cl.__all__:
        assert hasattr(stable_cl, name)
