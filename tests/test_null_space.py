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


# ------------------------------------------------------------------ boundary behaviour
def test_tol_zero_is_strictest_and_never_larger_than_the_default(setup):
    R, _, d, N = setup
    N0 = null_basis(R, d, 0.0)
    assert N0.shape[1] <= N.shape[1]
    assert np.allclose(N0.T @ N0, np.eye(N0.shape[1]), atol=1e-9)
    # for a full-row-rank R with exact zeros removed, tol=0 still annihilates A
    assert np.abs((R * np.sqrt(d)) @ N0).max() < 1e-9


def test_tol_zero_on_exactly_rank_deficient_input_still_gives_an_orthonormal_null_basis():
    R = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])  # rank 1
    N = null_basis(R, np.ones(4), 0.0)
    assert N.shape[1] >= 2 and np.abs(R @ N).max() < 1e-12
    assert np.allclose(N.T @ N, np.eye(N.shape[1]))


def test_repeated_singular_values_give_the_correct_projector(rng):
    p = 7
    Q2, _ = np.linalg.qr(rng.normal(size=(p, p)))
    Q1, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = (
        Q1 @ np.hstack([np.eye(3), np.zeros((3, p - 3))]) @ Q2
    )  # three equal singular values
    d = np.ones(p)
    N = null_basis(R, d)
    expected_P = np.eye(p) - Q2[:3].T @ Q2[:3]  # basis columns are not unique; P is
    assert N.shape == (p, p - 3)
    assert np.allclose(N @ N.T, expected_P, atol=1e-10)
    G = rng.normal(size=(p, 2))
    assert np.abs(R @ null_step(G, N, d, 0.0)).max() < 1e-10


def test_very_small_singular_values_are_treated_as_null_according_to_tol():
    R = np.diag([1.0, 1e-12, 0.0])[:2]  # sigma = (1, 1e-12) on a 3-dimensional space
    d = np.ones(3)
    assert null_basis(R, d, 1e-10).shape[1] == 2  # 1e-12 <= 1e-10: counted as null
    assert null_basis(R, d, 1e-14).shape[1] == 1  # 1e-12 > 1e-14: a real constraint
    G = np.random.default_rng(0).normal(size=(3, 2))
    N = null_basis(R, d, 1e-10)
    bound = 1e-10 * np.linalg.norm(G)
    assert (
        np.linalg.norm(R @ null_step(G, N, d, 0.0)) <= bound
    )  # ||A N|| <= tol * max(s1, 1)


def test_zero_effective_rates_with_reference_data_give_a_free_basis_and_a_zero_step(
    rng,
):
    R, G, d = rng.normal(size=(4, 7)), rng.normal(size=(7, 3)), np.zeros(7)
    N = null_basis(R, d)
    assert N.shape == (7, 7) and np.allclose(N @ N.T, np.eye(7))  # P = I
    assert not null_step(G, N, d, 0.0).any() and not null_step(G, N, d, 0.4).any()
    assert free_dim(N, d) == 0


@pytest.mark.parametrize("tol", [-1e-3, np.nan, np.inf, -np.inf, [1e-3, 1e-3]])
def test_invalid_tolerance_is_rejected(tol):
    with pytest.raises(ValueError):
        null_basis(np.eye(3), np.ones(3), tol)


@pytest.mark.parametrize("tol", [True, "1e-3", 1e-3 + 0j])
def test_non_real_tolerance_is_rejected(tol):
    with pytest.raises(TypeError):
        null_basis(np.eye(3), np.ones(3), tol)


@pytest.mark.parametrize(
    "R, d",
    [
        (np.ones(3), np.ones(3)),  # R not 2-D
        (np.ones((2, 3)), np.ones(4)),  # d has the wrong length
        (np.ones((2, 3)), np.ones((3, 1))),  # d not 1-D
        (np.ones((2, 3)), np.array([1.0, -1.0, 1.0])),  # negative rate
        (np.ones((2, 3)), np.array([1.0, np.nan, 1.0])),  # non-finite rate
        (np.array([[1.0, np.nan, 0.0]]), np.ones(3)),  # non-finite reference
        (np.ones((2, 0)), np.ones(0)),  # no features
    ],
)
def test_incompatible_null_basis_inputs_are_rejected(R, d):
    with pytest.raises(ValueError):
        null_basis(R, d)


def test_null_step_and_free_dim_reject_incompatible_shapes(setup):
    _, G, d, N = setup
    with pytest.raises(ValueError):
        null_step(G[:-1], N, d)  # G has the wrong number of rows
    with pytest.raises(ValueError):
        null_step(G, N[:-1], d)  # N has the wrong number of rows
    with pytest.raises(ValueError):
        null_step(G, N, d[:, None])  # d not 1-D
    with pytest.raises(ValueError):
        free_dim(N[:-1], d)
