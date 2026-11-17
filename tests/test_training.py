# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Softmax, the cross-entropy gradient and whole-experience invariants of train_task."""

import numpy as np
import pytest

from stable_cl import softmax, train_task


def test_softmax_rows_are_probabilities_and_stable():
    z = np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]])
    p = softmax(z)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1.0) and (p >= 0).all()
    assert np.allclose(p[1], 1 / 3)


def test_softmax_is_shift_invariant():
    z = np.random.default_rng(0).normal(size=(4, 5))
    assert np.allclose(softmax(z), softmax(z + 7.0))


class FirstRng:
    """Deterministic stand-in for a Generator: always picks the first ``size`` pool members."""

    def choice(self, pool, size, replace):
        return pool[:size]


def test_one_ordinary_step_is_minus_d_times_the_cross_entropy_gradient():
    rng = np.random.default_rng(0)
    p, n_seen = 6, 3
    X = rng.normal(size=(12, p))
    y = np.repeat([0, 1, 2], 4)
    W0 = rng.normal(size=(p, 3)) * 0.3
    eta = [1.0, 0.5, 2.0, 0.25]
    deg = np.array([0, 1, 1, 2, 3, 3])
    # bs=12, three classes of 4: one batch containing every example exactly once
    W = W0.copy()
    train_task(W, X, y, X[:0], y[:0], deg, eta, FirstRng(), n_seen, epochs=1, bs=12)

    def mean_ce(V):
        P = softmax(X @ V)
        return -np.mean(np.log(P[np.arange(len(y)), y]))

    num = np.zeros_like(W0)  # central finite differences of the mean cross-entropy
    h = 1e-6
    for i in range(p):
        for k in range(3):
            E = np.zeros_like(W0)
            E[i, k] = h
            num[i, k] = (mean_ce(W0 + E) - mean_ce(W0 - E)) / (2 * h)
    d = np.array(eta)[deg]
    assert np.allclose(W0 - W, d[:, None] * num, atol=1e-7)


def test_unseen_columns_and_frozen_rows_never_change_and_reference_logits_are_kept(
    toy, rng
):
    Phi, y, _, _, tdeg = toy
    cur = np.flatnonzero(np.isin(y, (2, 3)))
    old = np.flatnonzero(np.isin(y, (0, 1)))[::2]
    ref = np.concatenate([np.flatnonzero(y[old] == c)[:3] for c in (0, 1)])
    W0 = rng.normal(size=(Phi.shape[1], 10)) * 0.1
    for strategy in ("ordinary", "strategy_b"):
        W = W0.copy()
        train_task(
            W,
            Phi[cur],
            y[cur],
            Phi[old],
            y[old],
            tdeg,
            3.0,
            np.random.default_rng(1),
            4,
            control=[1, 0.5, 0, 1],
            strategy=strategy,
        )
        moved = np.abs(Phi[old][ref] @ (W - W0)[:, :4]).max()
        assert not (W - W0)[tdeg == 2].any()  # frozen degree
        assert not (W - W0)[:, 4:].any()  # classes not yet seen
        assert (moved < 1e-9) if strategy == "strategy_b" else moved > 1e-3


def test_first_experience_is_ordinary_even_for_strategy_b(toy):
    Phi, y, _, _, tdeg = toy
    cur = np.flatnonzero(np.isin(y, (0, 1)))
    out = []
    for strategy in ("ordinary", "strategy_b"):
        W = np.zeros((Phi.shape[1], 10))
        info = {}
        train_task(
            W,
            Phi[cur],
            y[cur],
            Phi[:0],
            y[:0],
            tdeg,
            3.0,
            np.random.default_rng(1),
            2,
            strategy=strategy,
            info=info,
        )
        out.append(W)
        assert info == {"null_dim": None, "free_dim": None}
    assert np.array_equal(out[0], out[1])


def test_all_zero_effective_rate_is_a_no_op(toy):
    Phi, y, _, _, tdeg = toy
    W = np.zeros((Phi.shape[1], 10))
    cur = np.flatnonzero(np.isin(y, (0, 1)))
    out = train_task(
        W,
        Phi[cur],
        y[cur],
        Phi[:0],
        y[:0],
        tdeg,
        3.0,
        np.random.default_rng(0),
        2,
        control=[0.0] * 4,
    )
    assert out is W and not W.any()


def test_info_reports_null_and_free_dimensions(toy):
    Phi, y, _, _, tdeg = toy
    cur = np.flatnonzero(np.isin(y, (2, 3)))
    old = np.flatnonzero(np.isin(y, (0, 1)))
    info = {}
    train_task(
        np.zeros((Phi.shape[1], 10)),
        Phi[cur],
        y[cur],
        Phi[old],
        y[old],
        tdeg,
        3.0,
        np.random.default_rng(0),
        4,
        strategy="strategy_b",
        info=info,
    )
    p = Phi.shape[1]
    assert (
        info["null_dim"] == p - 6
    )  # 6 reference rows (3 per old class), full row rank
    assert info["free_dim"] == p - 6


def test_invalid_strategy_and_beta_are_rejected(toy):
    Phi, y, _, _, tdeg = toy
    args = (
        np.zeros((Phi.shape[1], 10)),
        Phi[:5],
        y[:5],
        Phi[:0],
        y[:0],
        tdeg,
        3.0,
        np.random.default_rng(0),
        2,
    )
    with pytest.raises(ValueError):
        train_task(*args, strategy="nope")
    with pytest.raises(ValueError):
        train_task(*args, beta=1.5)
