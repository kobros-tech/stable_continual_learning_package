# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Null-space mathematics of strategy_b (math.tex, Propositions 1 and 2)."""

import numpy as np
import pytest

from stable_cl import diagonal_rates, free_dim, null_basis, null_step


@pytest.fixture
def setup(rng, deg, rates):
    """Random R (8 reference rows) and G, with degree 2 frozen."""
    eta, m = rates
    d = diagonal_rates(eta, deg, m)
    R = rng.normal(size=(8, len(deg)))
    G = rng.normal(size=(len(deg), 4))
    return R, G, d, null_basis(R, d)


def test_null_basis_is_orthonormal_and_annihilated_by_A(setup):
    R, _, d, N = setup
    assert np.allclose(N.T @ N, np.eye(N.shape[1]), atol=1e-9)
    assert np.abs((R * np.sqrt(d)) @ N).max() < 1e-9


def test_exact_step_leaves_reference_logits_unchanged(setup):
    R, G, d, N = setup
    assert np.abs(R @ null_step(G, N, d, 0.0)).max() < 1e-9


def test_frozen_rows_do_not_move(setup, deg):
    _, G, d, N = setup
    assert not null_step(G, N, d, 0.0)[deg == 2].any()
    assert not null_step(G, N, d, 0.7)[deg == 2].any()


def test_exact_step_matches_pseudoinverse_formula(setup, deg):
    R, G, d, N = setup
    root = np.sqrt(d)[:, None]
    A = R * np.sqrt(d)
    expected = root * ((np.eye(len(deg)) - np.linalg.pinv(A) @ A) @ (root * G))
    assert np.allclose(null_step(G, N, d, 0.0), expected)


def test_beta_one_is_ordinary_rate_weighted_sgd(setup):
    _, G, d, N = setup
    assert np.allclose(null_step(G, N, d, 1.0), d[:, None] * G)


def test_reference_logits_move_by_beta_times_their_sgd_change(setup):
    R, G, d, N = setup
    b = 0.3
    assert np.allclose(R @ null_step(G, N, d, b), b * R @ (d[:, None] * G))


def test_step_is_a_descent_direction_with_the_stated_value(setup):
    _, G, d, N = setup
    root, b = np.sqrt(d)[:, None], 0.3
    rg = root * G
    Prg = N @ (N.T @ rg)
    value = b * np.sum(rg**2) + (1 - b) * np.sum(Prg**2)
    assert np.isclose(np.sum(G * null_step(G, N, d, b)), value)
    assert np.sum(G * null_step(G, N, d, 0.0)) >= -1e-12


def test_free_dim_equals_support_size_minus_reference_rank(setup):
    R, _, d, N = setup
    S = d > 0
    assert free_dim(N, d) == S.sum() - np.linalg.matrix_rank(R[:, S])


def test_exact_step_is_the_constrained_minimiser(setup, rng):
    _, G, d, N = setup
    S, root = d > 0, np.sqrt(d)[:, None]
    step = null_step(G, N, d, 0.0)
    Z = root * (N @ rng.normal(size=(N.shape[1], 4)))  # feasible perturbation of dW
    f = lambda U: np.sum(G * U) + 0.5 * np.sum(U[S] ** 2 / d[S][:, None])
    assert np.isclose(f(-step + Z) - f(-step), 0.5 * np.sum(Z[S] ** 2 / d[S][:, None]))


def test_relaxed_logit_change_respects_the_spectral_bound(setup):
    R, G, d, _ = setup
    b, tol = 0.3, 0.3
    Nt = null_basis(R, d, tol)
    tau = tol * np.linalg.svd(R * np.sqrt(d), compute_uv=False)[0]
    rg = np.sqrt(d)[:, None] * G
    gap = np.linalg.norm(R @ null_step(G, Nt, d, b) - b * R @ (d[:, None] * G))
    assert gap <= (1 - b) * tau * np.linalg.norm(rg) + 1e-9


def test_larger_tolerance_gives_larger_basis(setup):
    R, _, d, _ = setup
    dims = [null_basis(R, d, tol).shape[1] for tol in (1e-10, 0.03, 0.3, 3.0)]
    assert dims == sorted(dims)


def test_null_basis_edge_cases():
    assert null_basis(np.zeros((3, 5)), np.ones(5)).shape == (5, 5)
    assert null_basis(np.eye(5), np.ones(5)).shape == (5, 0)
    assert null_basis(np.zeros((0, 5)), np.ones(5)).shape == (5, 5)


def test_free_dim_is_zero_for_empty_basis():
    assert free_dim(np.zeros((5, 0)), np.ones(5)) == 0


def test_beta_outside_unit_interval_is_rejected(setup):
    _, G, d, N = setup
    for beta in (-0.1, 1.1):
        with pytest.raises(ValueError):
            null_step(G, N, d, beta)
