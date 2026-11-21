# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The public steps and the achieved spectral bound: shapes, identities, validation."""

import numpy as np
import pytest

from stable_cl import (
    diagonal_rates,
    null_basis,
    ordinary_step,
    relaxed_step,
    spectral_bound,
)


def _orthonormal(rng, p, q):
    return np.linalg.qr(rng.normal(size=(p, q)))[0] if q else np.zeros((p, 0))


@pytest.fixture
def problem(rng, deg, rates):
    eta, m = rates
    d = diagonal_rates(eta, deg, m)
    R = rng.normal(size=(8, len(deg)))
    G = rng.normal(size=(len(deg), 4))
    return R, G, d


@pytest.mark.parametrize("q", [0, 3, 40])
def test_ordinary_step_is_d_times_g_and_equals_relaxed_at_beta_one(rng, problem, q):
    _, G, d = problem
    S = ordinary_step(G, d)
    assert np.array_equal(S, d[:, None] * G)
    N = _orthonormal(rng, len(d), q)  # any basis: beta = 1 ignores it
    assert np.allclose(relaxed_step(G, N, d, 1.0), S, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 1.0])
def test_relaxed_step_matches_the_explicit_matrix_formula(problem, beta):
    R, G, d = problem
    N = null_basis(R, d, 0.03)
    Dh, P = np.diag(np.sqrt(d)), N @ N.T
    expected = Dh @ (P + beta * (np.eye(len(d)) - P)) @ Dh @ G
    assert np.allclose(relaxed_step(G, N, d, beta), expected, rtol=1e-10, atol=1e-12)


def test_steps_return_float64_of_the_right_shape_and_leave_inputs_alone(problem):
    R, G, d = problem
    N = null_basis(R, d)
    G32, G_before = G.astype(np.float32), G.astype(np.float32).copy()
    for S in (ordinary_step(G32, d), relaxed_step(G32, N, d, 0.4)):
        assert S.shape == G.shape and S.dtype == np.float64
    assert np.array_equal(G32, G_before)


def test_ordinary_step_rejects_invalid_inputs(problem):
    _, G, d = problem
    bad_G = G.copy()
    bad_G[0, 0] = np.nan
    bad_d = d.copy()
    bad_d[0] = -1.0
    for args in [(G[:-1], d), (G, d[:, None]), (bad_G, d), (G, bad_d), (G[:, 0], d)]:
        with pytest.raises(ValueError):
            ordinary_step(*args)


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf, [0.1, 0.2]])
def test_beta_outside_the_unit_interval_is_rejected(problem, bad):
    R, G, d = problem
    with pytest.raises(ValueError):
        relaxed_step(G, null_basis(R, d), d, bad)


@pytest.mark.parametrize("bad", [True, "0.5", 0.5 + 0j])
def test_non_real_beta_is_rejected(problem, bad):
    R, G, d = problem
    with pytest.raises(TypeError):
        relaxed_step(G, null_basis(R, d), d, bad)


@pytest.mark.parametrize("tol", [1e-10, 0.03, 0.3])
def test_spectral_bound_is_the_largest_singular_value_of_A_N(problem, tol):
    R, _, d = problem
    N = null_basis(R, d, tol)
    A = R * np.sqrt(d)
    s = np.linalg.svd(A, compute_uv=False)
    k = int((s > tol * max(s[0], 1.0)).sum())
    expected = float(s[k]) if k < len(s) else 0.0  # sigma_{k+1}: first discarded value
    got = spectral_bound(R, d, N)
    assert isinstance(got, float)
    assert np.isclose(got, expected, rtol=1e-9, atol=1e-12)
    assert np.isclose(got, np.linalg.svd(A @ N, compute_uv=False)[0], atol=1e-12)


def test_spectral_bound_of_an_exact_basis_is_round_off(problem):
    R, _, d = problem
    assert spectral_bound(R, d, null_basis(R, d, 1e-10)) < 1e-9


def test_spectral_bound_is_zero_without_rows_or_columns(problem):
    R, _, d = problem
    assert spectral_bound(R, d, np.zeros((len(d), 0))) == 0.0
    assert spectral_bound(np.zeros((0, len(d))), d, np.eye(len(d))) == 0.0


def test_spectral_bound_rejects_invalid_inputs(problem):
    R, _, d = problem
    N = null_basis(R, d)
    bad_N = N.copy()
    bad_N[0, 0] = np.inf
    cases = [
        (R[:, :0], d[:0], N[:0]),  # no columns
        (R, d[:-1], N),
        (R, d, N[:-1]),
        (R, d, bad_N),
        (R, d[:, None], N),
        (R[0], d, N),  # R not 2-D
    ]
    for args in cases:
        with pytest.raises(ValueError):
            spectral_bound(*args)


@pytest.mark.parametrize(
    "dtype, value",
    [(np.int64, 2**40), (np.int32, 2**20), (np.uint8, 200), (np.int8, 100)],
)
def test_integer_inputs_are_multiplied_in_float64_and_cannot_overflow(dtype, value):
    G = np.full((3, 2), value, dtype=dtype)
    d = np.full(3, value, dtype=dtype)
    expected = float(value) ** 2  # beyond the range of `dtype`: integer maths wraps
    out = ordinary_step(G, d)
    assert out.dtype == np.float64 and out.shape == G.shape
    assert np.array_equal(out, np.full((3, 2), expected))


def test_integer_inputs_agree_with_their_float_versions():
    rng = np.random.default_rng(5)
    G = rng.integers(-1000, 1000, size=(6, 3))
    d = rng.integers(0, 50, size=6)
    assert np.array_equal(
        ordinary_step(G, d), d.astype(float)[:, None] * G.astype(float)
    )


@pytest.mark.parametrize("bad", [np.ones((3, 2), dtype=bool), np.ones((3, 2)) + 0j])
def test_ordinary_step_rejects_boolean_and_complex_gradients(bad):
    with pytest.raises(TypeError):
        ordinary_step(bad, np.ones(3))


@pytest.mark.parametrize("bad", [np.ones(3, dtype=bool), np.ones(3) + 0j])
def test_ordinary_step_rejects_boolean_and_complex_rates(bad):
    with pytest.raises(TypeError):
        ordinary_step(np.ones((3, 2)), bad)
