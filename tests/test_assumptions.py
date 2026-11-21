# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""M6: numerical checks of the assumptions the Lean theorems take from the implementation.

Lean proves statements about the exact mathematical update.  Everything here is a floating-point
test of what ``null_basis`` returns; none of it is a proof.  Each test names the assumption
(A2-A5, see ``docs/traceability.md``) or the proposition it checks.

Tolerance convention (identical in ``null_basis``, ``math.tex`` and these tests):
``tau = tol * sigma_1`` with ``sigma_1`` the largest singular value of ``A = R D^{1/2}``
(relative, so invariant under rescaling of ``R`` or ``d``; ``tol`` lies in ``[0, 1)``).
"""

import numpy as np
import pytest

from stable_cl import diagonal_rates, null_basis, relaxed_step

NULL_RESIDUAL_TOL = 1e-9  # A2: ||A N|| on exact-null fixtures
ORTHO_TOL = 1e-9  # A3: ||N^T N - I||
COMPLETENESS_TOL = 1e-9  # A5: ||N N^T V - V||
SLACK = 1e-9  # floating-point slack on the residual inequalities
RELATIVE_SLACK = (
    1e-12  # round-off slack on A4, in units of sigma_1 (about 4500 machine epsilons)
)


def make_case(seed, p=24, rows=9, scale=1.0, rank=None, frozen=0):
    """Random reference matrix ``R`` (optionally rank-deficient), rates ``d`` and gradient ``G``.

    ``frozen`` zeroes the rate of that many leading coordinates, so ``d_j = 0`` occurs.
    ``scale`` multiplies ``R`` exactly once, also when ``rank`` is given: ``R[:rank]`` below is
    taken from the already scaled matrix, so the product is ``scale`` times the unit-scale one
    (pinned by ``test_make_case_applies_scale_exactly_once``). Do not scale it again there.
    """
    rng = np.random.default_rng(seed)
    R = scale * rng.normal(size=(rows, p))
    if rank is not None:
        R = R[:rank].T @ rng.normal(size=(rank, rows))
        R = R.T  # rows x p, rank <= `rank`
    d = rng.uniform(0.2, 3.0, size=p)
    d[:frozen] = 0.0
    G = rng.normal(size=(p, 3))
    return R, d, G


def tau_of(R, d, tol):
    A = R * np.sqrt(d)
    sigma1 = np.linalg.svd(A, compute_uv=False)[0] if A.size else 0.0
    return tol * float(sigma1)


def sigma1_of(R, d):
    A = R * np.sqrt(d)
    return float(np.linalg.svd(A, compute_uv=False)[0]) if A.size else 0.0


def independent_null_space(A, rel=1e-10):
    """Orthonormal basis of null(A) computed by a different route (QR of A^T), not ``null_basis``."""
    Q, Rr = np.linalg.qr(A.T, mode="complete")
    diag = np.abs(np.diag(Rr)) if Rr.size else np.array([])
    rank = int((diag > rel * diag.max(initial=0.0)).sum())
    return Q[:, rank:]


CASES = [
    {"seed": s, "scale": sc, "rank": rk, "frozen": fz}
    for s in range(4)
    for sc in (0.01, 1.0, 100.0)
    for rk, fz in ((None, 0), (None, 5), (4, 0), (4, 6))
]


# ------------------------------------------------------------- A3 and A2 (exact basis)
@pytest.mark.parametrize("case", CASES)
def test_a3_exact_basis_is_orthonormal(case):
    R, d, _ = make_case(**case)
    N = null_basis(R, d)
    assert np.abs(N.T @ N - np.eye(N.shape[1])).max() <= ORTHO_TOL


@pytest.mark.parametrize("case", CASES)
def test_a2_exact_basis_is_annihilated_by_A(case):
    R, d, _ = make_case(**case)
    N = null_basis(R, d)
    A = R * np.sqrt(d)
    # relative to the size of A so that scale=100 is not judged by an absolute residual only
    assert np.abs(A @ N).max() <= NULL_RESIDUAL_TOL * max(np.abs(A).max(), 1.0) * 10


# ------------------------------------------------------------- A5 (completeness)
@pytest.mark.parametrize("case", CASES)
def test_a5_projector_fixes_every_vector_of_an_independently_built_null_space(case):
    R, d, _ = make_case(**case)
    A = R * np.sqrt(d)
    N = null_basis(R, d)
    M = independent_null_space(
        A
    )  # orthonormal basis of null(A) from QR, not from the SVD
    assert N.shape[1] == M.shape[1]
    V = M @ np.random.default_rng(1).normal(size=(M.shape[1], 5))
    assert np.abs(N @ (N.T @ V) - V).max() <= COMPLETENESS_TOL * max(np.abs(V).max(), 1)


def test_a5_holds_for_nearly_degenerate_singular_values():
    # singular values 1, 1e-6, 1e-12: the cutoff 1e-10 must separate 1e-6 (constraint) from
    # 1e-12 (null); the basis is complete for the null space *at that cutoff*.
    rng = np.random.default_rng(3)
    p = 6
    Q1, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q2, _ = np.linalg.qr(rng.normal(size=(p, p)))
    R = Q1 @ np.diag([1.0, 1e-6, 1e-12]) @ Q2[:3]
    N = null_basis(R, np.ones(p), 1e-10)
    assert N.shape[1] == p - 2
    V = Q2[
        2:
    ].T  # right singular vectors for sigma = 1e-12 and the exact null directions
    assert np.abs(N @ (N.T @ V) - V).max() <= 1e-9


# ------------------------------------------------------------- A4 (spectral bound), direct
@pytest.mark.parametrize("tol", [1e-10, 1e-3, 0.03, 0.3])
@pytest.mark.parametrize("case", CASES)
def test_a4_spectral_norm_of_A_N_is_at_most_tau(case, tol):
    R, d, _ = make_case(**case)
    A = R * np.sqrt(d)
    N = null_basis(R, d, tol)
    if N.shape[1] == 0:
        return  # empty basis: A N is the empty product, bound holds trivially
    # explicit round-off slack, relative to sigma_1 so that it cannot swamp a small tau
    bound = tau_of(R, d, tol) + RELATIVE_SLACK * sigma1_of(R, d)
    assert np.linalg.norm(A @ N, 2) <= bound


@pytest.mark.parametrize("tol", [1e-3, 0.03, 0.3])
@pytest.mark.parametrize("case", CASES)
def test_cutoff_consistency_links_the_svd_threshold_to_a4(case, tol):
    # Every right singular direction with sigma <= tau is in the basis and none above it, so
    # ||A N||_2 = sigma_{k+1} <= tau holds by construction of the threshold, not by luck.
    R, d, _ = make_case(**case)
    A = R * np.sqrt(d)
    s = np.linalg.svd(A, compute_uv=False)
    tau = tau_of(R, d, tol)
    k = int((s > tau).sum())  # constraints kept
    N = null_basis(R, d, tol)
    assert N.shape[1] == A.shape[1] - k
    if N.shape[1]:
        sigma_next = s[k] if k < len(s) else 0.0
        assert np.linalg.norm(A @ N, 2) <= sigma_next + RELATIVE_SLACK * sigma1_of(R, d)
        assert sigma_next <= tau


# ------------------------------------------------------------- Prop 2(c): residual bound
@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("tol", [0.03, 0.3])
@pytest.mark.parametrize("case", CASES)
def test_relaxed_residual_bound_over_a_beta_grid(case, tol, beta):
    R, d, G = make_case(**case)
    N = null_basis(R, d, tol)
    root = np.sqrt(d)[:, None]
    u = root * G
    dW = -relaxed_step(G, N, d, beta)
    gap = np.linalg.norm(R @ dW + beta * (R * np.sqrt(d)) @ u)
    assert gap <= (1 - beta) * tau_of(R, d, tol) * np.linalg.norm(u) + SLACK


@pytest.mark.parametrize("tol", [1e-10, 1e-3, 0.03, 0.3])
@pytest.mark.parametrize("case", CASES)
def test_strict_step_approximate_preservation_is_the_beta_zero_case(case, tol):
    # ||R dW||_F <= tau ||D^{1/2} G||_F : the beta = 0 instance of Prop 2(c).
    R, d, G = make_case(**case)
    N = null_basis(R, d, tol)
    u = np.sqrt(d)[:, None] * G
    assert (
        np.linalg.norm(R @ relaxed_step(G, N, d, 0.0))
        <= tau_of(R, d, tol) * np.linalg.norm(u) + SLACK
    )


# ------------------------------------------------------------- Prop 1(d), 2(a): descent
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("case", CASES)
def test_descent_value_matches_the_closed_form_and_is_nonnegative(case, beta):
    R, d, G = make_case(**case)
    N = null_basis(R, d)
    u = np.sqrt(d)[:, None] * G
    Pu = N @ (N.T @ u)
    value = beta * np.sum(u**2) + (1 - beta) * np.sum(Pu**2)
    inner = np.sum(G * relaxed_step(G, N, d, beta))  # <G, S> = -<G, dW>
    assert np.isclose(inner, value, rtol=1e-9, atol=1e-9)
    assert inner >= -SLACK


# ------------------------------------------------------------- Prop 1(c): constrained minimiser
@pytest.mark.parametrize("case", CASES)
def test_strict_step_beats_random_feasible_perturbations(case):
    R, d, G = make_case(**case)
    N = null_basis(R, d)
    S = d > 0
    root = np.sqrt(d)[:, None]
    step = relaxed_step(G, N, d, 0.0)  # the update is -step

    def J(Z):
        return np.sum(G * Z) + 0.5 * np.sum(Z[S] ** 2 / d[S][:, None])

    rng = np.random.default_rng(7)
    best = J(-step)
    for _ in range(5):
        Z = root * (
            N @ rng.normal(size=(N.shape[1], G.shape[1]))
        )  # feasible: Z_S free, R Z ~ 0
        assert J(-step + Z) >= best - 1e-9


# ------------------------------------------------------------- M2 and degenerate cases
@pytest.mark.parametrize("beta", [0.0, 0.4, 1.0])
def test_frozen_rows_are_exactly_zero(beta):
    R, d, G = make_case(2, frozen=7)
    N = null_basis(R, d)
    assert not relaxed_step(G, N, d, beta)[:7].any()


def test_empty_null_space_gives_the_zero_strict_step_and_sgd_for_beta_one():
    rng = np.random.default_rng(0)
    p = 5
    R, d, G = np.eye(p), np.ones(p), rng.normal(size=(p, 2))
    N = null_basis(R, d)
    assert N.shape == (p, 0)
    assert not relaxed_step(G, N, d, 0.0).any()
    assert np.allclose(relaxed_step(G, N, d, 1.0), d[:, None] * G)


def test_trivial_reference_gives_the_identity_projector():
    rng = np.random.default_rng(0)
    p = 5
    R, d, G = np.zeros((3, p)), rng.uniform(0.5, 2, p), rng.normal(size=(p, 2))
    N = null_basis(R, d)
    assert np.allclose(N @ N.T, np.eye(p))
    assert np.allclose(relaxed_step(G, N, d, 0.0), d[:, None] * G)


def test_zero_gradient_gives_a_zero_step():
    R, d, _ = make_case(0)
    N = null_basis(R, d)
    for beta in (0.0, 0.5, 1.0):
        assert not relaxed_step(np.zeros((len(d), 3)), N, d, beta).any()


def test_tau_is_relative_to_sigma1_also_below_one():
    # sigma_1 < 1: tau = tol * sigma_1 (smaller than tol), and rescaling R leaves N N^T unchanged.
    R, d, _ = make_case(0, scale=0.01)
    sigma1 = sigma1_of(R, d)
    assert sigma1 < 1 and np.isclose(tau_of(R, d, 0.3), 0.3 * sigma1)
    assert tau_of(R, d, 0.3) < 0.3
    N, N_big = null_basis(R, d, 0.3), null_basis(1e4 * R, d, 0.3)
    assert N.shape == N_big.shape and np.allclose(N @ N.T, N_big @ N_big.T, atol=1e-9)


@pytest.mark.parametrize("rank", [None, 4])
@pytest.mark.parametrize("frozen", [0, 6])
@pytest.mark.parametrize("scale", [0.01, 100.0])
def test_make_case_applies_scale_exactly_once(scale, rank, frozen):
    # Guards the fixture: rescaled cases must be `scale` times the unit-scale case, with the same
    # rank, rates and gradient (not identical to it, and not scaled a second time).
    R1, d1, G1 = make_case(1, scale=1.0, rank=rank, frozen=frozen)
    Rs, ds, Gs = make_case(1, scale=scale, rank=rank, frozen=frozen)
    assert np.allclose(Rs, scale * R1, rtol=1e-12, atol=0.0)
    assert np.array_equal(ds, d1) and np.array_equal(Gs, G1)
    if rank is not None:
        assert np.linalg.matrix_rank(Rs) == rank


@pytest.mark.parametrize("target", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("tol", [0.03, 0.3])
@pytest.mark.parametrize("beta", [0.0, 0.5])
@pytest.mark.parametrize("seed", range(3))
def test_relaxed_residual_bound_with_sigma1_around_one(seed, beta, tol, target):
    # The scaled cases above have sigma_1 from about 0.07 to 3500. Here R is rescaled so that
    # sigma_1 = 0.5, 1 and 2, around the point where the earlier max(sigma_1, 1) rule changed branch.
    R, d, G = make_case(seed, rank=4, frozen=5)
    R = R * (target / np.linalg.svd(R * np.sqrt(d), compute_uv=False)[0])
    assert np.isclose(np.linalg.svd(R * np.sqrt(d), compute_uv=False)[0], target)
    N = null_basis(R, d, tol)
    u = np.sqrt(d)[:, None] * G
    dW = -relaxed_step(G, N, d, beta)
    gap = np.linalg.norm(R @ dW + beta * (R * np.sqrt(d)) @ u)
    assert gap <= (1 - beta) * tau_of(R, d, tol) * np.linalg.norm(u) + SLACK


def test_diagonal_rates_are_the_default_rate_vector_for_these_cases():
    # guards the fixtures: rates built by `diagonal_rates` also satisfy the checks above
    deg = np.repeat([0, 1, 2, 3], [1, 5, 14, 20])
    d = diagonal_rates([3.0, 2.0, 0.75, 1.5], deg, [1.0, 0.5, 0.0, 1.0])
    R = np.random.default_rng(0).normal(size=(8, len(deg)))
    N = null_basis(R, d)
    A = R * np.sqrt(d)
    assert np.linalg.norm(A @ N, 2) <= 1e-9
