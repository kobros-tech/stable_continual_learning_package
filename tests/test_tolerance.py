# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""S3: the relative tolerance rule ``sigma_i > tol * sigma_1`` of ``null_basis``.

The earlier rule was ``sigma_i > tol * max(sigma_1, 1)``.  ``legacy_null_basis`` below is a copy of
that rule, kept only as a reference for the comparisons; it is not part of the package.  The new
rule must (1) agree with it when ``sigma_1 >= 1``, (2) give a null space contained in the legacy
one when ``sigma_1 < 1``, (3) not depend on the units of ``d`` or ``R``, and (4) keep its edge-case
conventions.  Everything here is a floating-point test, not a proof.
"""

import numpy as np
import pytest

from stable_cl import null_basis, ordinary_step, relaxed_step, spectral_bound

RELATIVE_SLACK = (
    1e-12  # round-off slack on ||A N|| <= tol * sigma_1, in units of sigma_1
)
TOLS = [1e-10, 1e-3, 0.03, 0.3, 0.9]


def legacy_null_basis(R, d, tol=1e-10):
    """The pre-S3 rule, threshold ``tol * max(sigma_1, 1)`` (reference only)."""
    R = np.asarray(R, dtype=float)
    if R.shape[0] == 0:
        return np.eye(R.shape[1])
    _, s, vt = np.linalg.svd(R * np.sqrt(d), full_matrices=True)
    return vt[int((s > tol * max(float(s[0]), 1.0)).sum()) :].T


def make(seed, scale=1.0, p=16, rows=6, frozen=0):
    rng = np.random.default_rng(seed)
    R = scale * rng.normal(size=(rows, p))
    d = rng.uniform(0.2, 3.0, size=p)
    d[:frozen] = 0.0
    return R, d, rng.normal(size=(p, 3))


def sigma1(R, d):
    return float(np.linalg.svd(R * np.sqrt(d), compute_uv=False)[0])


def proj(N):
    return N @ N.T


# ------------------------------------------------------------ sigma_1 >= 1: same as legacy
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("target", [1.0, 2.0, 50.0])
@pytest.mark.parametrize("frozen", [0, 5])
@pytest.mark.parametrize("seed", range(3))
def test_projector_equals_the_legacy_rule_when_sigma1_is_at_least_one(
    seed, frozen, target, tol
):
    R, d, _ = make(seed, frozen=frozen)
    R = R * (target / sigma1(R, d))
    N, N_old = null_basis(R, d, tol), legacy_null_basis(R, d, tol)
    assert N.shape == N_old.shape
    assert np.allclose(proj(N), proj(N_old), atol=1e-12)


@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("scale", [3.0, 100.0, 1e4])
@pytest.mark.parametrize("seed", range(3))
def test_basis_is_identical_to_the_legacy_basis_for_unscaled_sigma1_above_one(
    seed, scale, tol
):
    R, d, _ = make(seed, scale=scale)
    assert sigma1(R, d) >= 1.0
    # same expression `tol * sigma_1` on the same SVD: not merely close, equal
    assert np.array_equal(null_basis(R, d, tol), legacy_null_basis(R, d, tol))


# ------------------------------------------------------------ sigma_1 < 1: legacy contains new
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("scale", [1e-3, 1e-2, 0.1])
@pytest.mark.parametrize("frozen", [0, 5])
@pytest.mark.parametrize("seed", range(3))
def test_legacy_null_space_contains_the_new_one_when_sigma1_is_below_one(
    seed, frozen, scale, tol
):
    R, d, _ = make(seed, scale=scale, frozen=frozen)
    assert sigma1(R, d) < 1.0
    N, N_old = null_basis(R, d, tol), legacy_null_basis(R, d, tol)
    assert N.shape[1] <= N_old.shape[1]
    # every new basis vector lies in the span of the legacy basis: P_old N = N
    assert np.allclose(proj(N_old) @ N, N, atol=1e-9)


def test_the_two_rules_differ_when_sigma1_is_below_one_and_tol_is_large():
    R, d, _ = make(0, scale=0.01)
    p = R.shape[1]
    N_old, N = legacy_null_basis(R, d, 0.3), null_basis(R, d, 0.3)
    assert N_old.shape[1] == p  # legacy: tol * 1 exceeds every singular value, P = I
    assert N.shape[1] < p  # relative rule: the reference directions are constraints
    assert N.shape[1] >= p - R.shape[0]  # rank(A) <= number of reference rows


# ------------------------------------------------------------ scale invariance
@pytest.mark.parametrize("tol", [1e-10, 0.03, 0.3])
@pytest.mark.parametrize("c", [1e-22, 1e-8, 1e-3, 7.0, 1e6, 1e12])
@pytest.mark.parametrize("frozen", [0, 5])
@pytest.mark.parametrize("seed", range(3))
def test_projector_is_invariant_under_a_positive_rescaling_of_the_rates(
    seed, frozen, c, tol
):
    R, d, _ = make(seed, frozen=frozen)
    N, N_c = null_basis(R, d, tol), null_basis(R, c * d, tol)
    assert N.shape == N_c.shape
    assert np.allclose(proj(N), proj(N_c), atol=1e-9)


@pytest.mark.parametrize("tol", [1e-10, 0.03, 0.3])
@pytest.mark.parametrize("c", [1e-20, 1e-3, 1e3, 1e20])
@pytest.mark.parametrize("seed", range(3))
def test_projector_is_invariant_under_a_positive_rescaling_of_the_features(
    seed, c, tol
):
    R, d, _ = make(seed, frozen=3)
    N, N_c = null_basis(R, d, tol), null_basis(c * R, d, tol)
    assert N.shape == N_c.shape
    assert np.allclose(proj(N), proj(N_c), atol=1e-9)


# ------------------------------------------------------------ the d = 1e-22 hazard
def test_tiny_rates_are_not_treated_as_a_free_space():
    # Hazard of the legacy rule: with d = 1e-22, sigma_1 ~ 1e-11 < 1e-10 * max(sigma_1, 1) = 1e-10,
    # so every direction counted as null (P = I) and the "exact" step did not protect R at all.
    R, _, G = make(0, p=12, rows=4)
    d = np.full(12, 1e-22)
    assert sigma1(R, d) < 1e-10  # the premise of the hazard
    N_old = legacy_null_basis(R, d)
    assert N_old.shape == (12, 12)  # legacy: nothing constrained
    N = null_basis(R, d)
    assert N.shape == (12, 8)  # rank(R) = 4 directions constrained
    assert np.allclose(proj(N), proj(null_basis(R, np.ones(12))), atol=1e-9)

    sgd_change = np.linalg.norm(R @ ordinary_step(G, d))
    assert sgd_change > 0.0
    legacy_change = np.linalg.norm(R @ relaxed_step(G, N_old, d, 0.0))
    exact_change = np.linalg.norm(R @ relaxed_step(G, N, d, 0.0))
    assert legacy_change > 0.5 * sgd_change  # legacy: references move as under SGD
    assert exact_change <= 1e-9 * sgd_change  # relative rule: references are preserved


@pytest.mark.parametrize("tol", [1e-10, 0.03, 0.3])
@pytest.mark.parametrize("scale", [1e-30, 1e-20, 1e-6, 1.0, 1e6])
@pytest.mark.parametrize("seed", range(3))
def test_spectral_norm_of_A_N_is_at_most_tol_times_sigma1_at_every_scale(
    seed, scale, tol
):
    R, d, _ = make(seed, scale=scale, frozen=4)
    N = null_basis(R, d, tol)
    s1 = sigma1(R, d)
    # explicit, relative round-off slack: an absolute slack would swamp tol * sigma_1 at tiny scale
    assert spectral_bound(R, d, N) <= tol * s1 + RELATIVE_SLACK * s1


# ------------------------------------------------------------ edge cases
@pytest.mark.parametrize("tol", [0.0, 1e-10, 0.3, 0.999999])
def test_no_reference_rows_zero_rates_and_zero_sigma1_give_the_identity(tol):
    rng = np.random.default_rng(0)
    p = 7
    cases = {
        "m = 0": (np.zeros((0, p)), rng.uniform(0.5, 2.0, p)),
        "d = 0": (rng.normal(size=(4, p)), np.zeros(p)),
        "sigma_1 = 0 (R = 0)": (np.zeros((4, p)), rng.uniform(0.5, 2.0, p)),
        "R only on frozen rates": (
            np.hstack([rng.normal(size=(4, 3)), np.zeros((4, p - 3))]),
            np.hstack([np.zeros(3), np.ones(p - 3)]),
        ),
    }
    for name, (R, d) in cases.items():
        N = null_basis(R, d, tol)
        assert np.array_equal(N, np.eye(p)), name


def test_zero_rates_give_the_zero_step_and_zero_spectral_bound():
    R, G = np.random.default_rng(1).normal(size=(4, 6)), np.ones((6, 2))
    d = np.zeros(6)
    N = null_basis(R, d, 0.3)
    assert spectral_bound(R, d, N) == 0.0
    assert not relaxed_step(G, N, d, 0.0).any()


@pytest.mark.parametrize("tol", [0.0, 1e-10, 0.3, 0.999999])
@pytest.mark.parametrize("rate", [1.0, 1e-22, 1e22])
@pytest.mark.parametrize("scale", [1.0, 1e-30])
def test_full_column_rank_with_equal_singular_values_gives_an_empty_basis(
    tol, rate, scale
):
    # all singular values equal sigma_1 > tol * sigma_1, at every scale of R and d
    N = null_basis(scale * np.eye(5), np.full(5, rate), tol)
    assert N.shape == (5, 0)


# ------------------------------------------------------------ tol validation on [0, 1)
VALID_TOLS = [0, 0.0, 1e-300, 0.5, np.float64(0.25), np.nextafter(1.0, 0.0)]


@pytest.mark.parametrize("tol", VALID_TOLS)
def test_tol_inside_the_half_open_unit_interval_is_accepted(tol):
    N = null_basis(np.random.default_rng(0).normal(size=(3, 6)), np.ones(6), tol)
    assert np.allclose(N.T @ N, np.eye(N.shape[1]), atol=1e-12)


@pytest.mark.parametrize(
    "tol", [1, 1.0, np.nextafter(1.0, 2.0), 2.0, 10.0, -1e-300, -0.5, np.nan, np.inf]
)
def test_tol_outside_the_half_open_unit_interval_is_rejected(tol):
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        null_basis(np.eye(3), np.ones(3), tol)
