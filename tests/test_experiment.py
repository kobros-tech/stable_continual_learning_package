# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Incremental protocol, metric definitions and leakage-free data preparation."""

import numpy as np
import pytest

import stable_cl
from stable_cl import load_digits_features, run, stratified_cut
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


def _labels():
    """Digit labels of the whole dataset."""
    from sklearn.datasets import load_digits

    return load_digits().target


def _indices(split_seed, validation):
    """Fit and evaluation indices of a split, recomputed independently of the loader."""
    order = np.random.default_rng(split_seed).permutation(1797)
    fit, ev = order[: int(0.75 * 1797)], order[int(0.75 * 1797) :]
    if validation:
        fit, ev = stratified_cut(_labels(), fit, 0.8)
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
    assert (len(y_f), len(y_e)) == (len(fit), len(ev)) and len(fit) + len(ev) == 1347
    assert Phi_f.shape[1] == Phi_e.shape[1] == len(deg) == 969


def test_validation_view_is_class_stratified():
    y = _labels()
    tr, _ = _indices(103, False)
    _, y_f, _, y_e, _ = load_digits_features(103, True)
    n_tr = np.bincount(y[tr], minlength=10)
    assert np.array_equal(np.bincount(y_f, minlength=10), (0.8 * n_tr).astype(int))
    assert np.array_equal(
        np.bincount(y_f, minlength=10) + np.bincount(y_e, minlength=10), n_tr
    )


def test_test_protocol_is_not_stratified_and_keeps_its_sizes():
    _, y_tr, _, y_te, _ = load_digits_features(0)
    assert (len(y_tr), len(y_te)) == (1347, 450)
    tr, te = _indices(
        0, False
    )  # plain permutation cut, unchanged by the validation fix
    assert np.array_equal(y_tr, _labels()[tr]) and np.array_equal(y_te, _labels()[te])


# ------------------------------------------------------------------ stratified_cut
def test_stratified_cut_takes_the_floor_of_the_fraction_of_every_class():
    y = np.repeat([0, 1, 2], [10, 7, 3])
    idx = np.random.default_rng(0).permutation(len(y))
    first, second = stratified_cut(y, idx, 0.8)
    assert np.bincount(y[first], minlength=3).tolist() == [8, 5, 2]
    assert np.bincount(y[second], minlength=3).tolist() == [2, 2, 1]


def test_stratified_cut_partitions_preserves_order_and_is_deterministic():
    y = np.repeat(np.arange(4), 25)
    idx = np.random.default_rng(1).permutation(len(y))[:80]
    first, second = stratified_cut(y, idx, 0.5)
    assert sorted(np.concatenate([first, second])) == sorted(idx)
    assert not set(first) & set(second)
    pos = {v: i for i, v in enumerate(idx)}
    assert all(pos[a] < pos[b] for a, b in zip(first, first[1:]))  # relative order kept
    assert all(
        np.array_equal(u, v)
        for u, v in zip(stratified_cut(y, idx, 0.5), (first, second))
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5, np.nan])
def test_stratified_cut_rejects_invalid_fractions(fraction):
    with pytest.raises(ValueError):
        stratified_cut(np.zeros(5, int), np.arange(5), fraction)


# ------------------------------------------------------------------ replay-memory availability
def test_memory_larger_than_a_class_raises_instead_of_resampling(toy):
    with pytest.raises(ValueError, match="replay memory of 41"):
        run(*toy, 0, 3.0, mem=41)  # the toy has exactly 40 examples per class
    run(*toy, 0, 3.0, mem=40)  # exactly enough is fine


def test_memory_error_names_every_short_class_with_its_count():
    y = np.repeat(np.arange(10), 30)
    y[:5] = 3  # class 3 has 35, every other class 30 or fewer
    X = np.random.default_rng(0).normal(size=(len(y), 3))
    deg = np.zeros(3, int)
    with pytest.raises(ValueError) as e:
        run(X, y, X, y, deg, 0, 3.0, mem=33)
    msg = str(e.value)
    assert "0: 25" in msg and "3: 35" not in msg and "9: 30" in msg


def test_previous_memory_100_failure_is_now_a_clear_error():
    """Validation split 103 has only 96 fitting examples of digit 8 (even after stratifying)."""
    data = load_digits_features(103, validation=True)
    counts = np.bincount(data[1], minlength=10)
    assert counts[8] == 96 and counts.min() == 96
    with pytest.raises(ValueError, match=r"8: 96"):
        run(*data, 0, 3.0, mem=100)


def test_memory_100_is_feasible_on_the_stratified_validation_split_100():
    data = load_digits_features(100, validation=True)
    assert np.bincount(data[1], minlength=10).min() >= 100
    assert 0.0 <= run(*data, 0, 3.0, mem=100)["acc"] <= 1.0


@pytest.mark.parametrize("mem", [-1])
def test_negative_memory_is_rejected(toy, mem):
    with pytest.raises(ValueError):
        run(*toy, 0, 3.0, mem=mem)


@pytest.mark.parametrize("mem", [2.5, True, "20"])
def test_non_integer_memory_is_rejected(toy, mem):
    with pytest.raises(TypeError):
        run(*toy, 0, 3.0, mem=mem)


def test_package_exports_every_documented_name():
    for name in stable_cl.__all__:
        assert hasattr(stable_cl, name)


# ------------------------------------------------------------------ demo
def test_demo_accepts_a_generator_and_consumes_it_once(capsys):
    stable_cl.demo(seed for seed in (3,))
    out = capsys.readouterr().out
    assert "training seeds [3]" in out and "nan" not in out.lower()
    assert (
        out.count("acc=") == 3
    )  # one line per configuration, none computed from zero seeds


def test_demo_rejects_empty_seeds_before_doing_any_work():
    with pytest.raises(ValueError, match="at least one"):
        stable_cl.demo([])
    with pytest.raises(ValueError, match="at least one"):
        stable_cl.demo(seed for seed in ())
