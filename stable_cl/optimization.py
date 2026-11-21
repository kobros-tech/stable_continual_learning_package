# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

r"""Diagonal rates and the two steps of the update: ``ordinary`` and ``relaxed``.

One minibatch gradient :math:`G \in \mathbb{R}^{p \times K}` is turned into a weight update using a
diagonal rate matrix :math:`D = \operatorname{diag}(d)`.  The vector :math:`d` holds the
*effective* rate of every feature: it already contains every learning-rate factor, and no further
rate is applied to a step.  With group base rates :math:`\eta` and degree-control factors :math:`m`
it is built by :func:`diagonal_rates` as :math:`d_j = \eta_{g(j)}\, m_{g(j)}`.

.. math::

    \text{ordinary:}\quad \Delta W = -D\,G,
    \qquad
    \text{relaxed:}\quad
    \Delta W = -D^{1/2}\bigl[P + \beta (I - P)\bigr] D^{1/2} G,
    \quad P = N N^\top .

Here :math:`R` holds the features of the stored reference examples and the columns of :math:`N` form
an orthonormal basis of (the numerical) null space of :math:`A = R D^{1/2}`.  The special cases are
:math:`\beta = 1` (ordinary rate-weighted SGD, for every :math:`N`), :math:`\beta = 0` (exact
reference preservation when :math:`A N = 0`) and :math:`0 < \beta < 1` (relaxed).  The relaxed step
is not unconditionally reference-preserving: with an approximate null basis the reference logits
change by at most the residual bound of ``math.tex`` (Proposition 2), under the assumptions stated
there and in ``docs/traceability.md``.
"""

from __future__ import annotations

import numpy as np


def _check_matrix_and_rates(M, d, what):
    r"""Validate a ``(p, q)`` matrix against a rate vector of length ``p``.

    Parameters
    ----------
    M : ndarray
        Matrix whose first dimension must equal ``len(d)``.
    d : ndarray
        Diagonal rates.
    what : str
        Name of ``M`` used in error messages.

    Raises
    ------
    ValueError
        If ``M`` is not 2-D, ``d`` is not 1-D of matching length, or either contains
        non-finite values or ``d`` has negative entries.
    """
    if np.ndim(M) != 2:
        raise ValueError(f"{what} must be 2-D, got {np.ndim(M)}-D")
    if np.ndim(d) != 1 or len(d) != np.shape(M)[0]:
        raise ValueError(
            f"d must be 1-D with len(d) == {what}.shape[0] == {np.shape(M)[0]}"
        )
    if (
        not np.isfinite(M).all()
        or not np.isfinite(d).all()
        or (np.asarray(d) < 0).any()
    ):
        raise ValueError(f"{what} must be finite and d finite and non-negative")


def _real(v, what):
    r"""Convert ``v`` to a float64 array, rejecting non-real input *before* any cast.

    Booleans (which NumPy would silently cast to 0/1), complex numbers (whose imaginary part a cast
    would discard) and strings are not valid rates or control factors.

    Parameters
    ----------
    v : array_like
        Candidate numbers.
    what : str
        Name used in error messages.

    Returns
    -------
    ndarray of float64
        Same shape as ``v``.

    Raises
    ------
    ValueError
        If ``v`` is a ragged (non-rectangular) array.
    TypeError
        If ``v`` contains booleans, complex numbers or non-numbers.
    """
    try:
        flat, a = np.ravel(np.asarray(v, dtype=object)), np.asarray(v)
    except ValueError as e:
        raise ValueError(f"{what} must be a regular array of numbers") from e
    if any(isinstance(x, (bool, np.bool_)) for x in flat) or a.dtype.kind not in "iuf":
        raise TypeError(
            f"{what} must be real numbers (no booleans, complex or strings)"
        )
    return a.astype(float)


def diagonal_rates(base_rates, group, control=None):
    r"""Diagonal of the rate matrix :math:`D` (the matrix itself is never formed).

    .. math::

        d_j = \eta_{g(j)}\, m_{g(j)},
        \qquad D = \operatorname{diag}(d),
        \qquad D^{1/2} = \operatorname{diag}(\sqrt{d_j}),

    where :math:`\eta \in \mathbb{R}^{G}_{\ge 0}` are base rates per feature group (at least one
    positive), :math:`m \in [0, 1]^{G}` are degree-control factors (:math:`m = \mathbf{1}` when
    ``control`` is ``None``) and :math:`g(j)` is the group of feature :math:`j`.  The number of
    groups :math:`G` is ``len(base_rates)``; nothing here knows what a group means (for example
    polynomial degree).

    Conventions, all enforced:

    * ``control=None`` and ``control=[1] * G`` give exactly :math:`d_j = \eta_{g(j)}`.
    * :math:`m_g = 0` freezes every feature of group :math:`g`.
    * All-zero effective rates are allowed and describe a no-op experience.
    * The map is *prefix-stable*: ``d`` depends on ``group[j]`` only, so features added by
      expansion take the rate of their own group and old features keep theirs.
    * Neither ``base_rates`` nor ``control`` is broadcast: a scalar is rejected.

    Parameters
    ----------
    base_rates : array_like, shape (G,)
        Base rates per group, finite and non-negative, at least one positive.  If a global
        learning rate applies, it must already be a factor of these rates.
    group : ndarray of int, shape (p,)
        Group index of each feature, values in ``0..G-1``.
    control : array_like, shape (G,), optional
        Degree-control factors in ``[0, 1]``.

    Returns
    -------
    ndarray, shape (p,)
        The vector ``d`` with :math:`d_j = \eta_{g(j)} m_{g(j)} \ge 0`.

    Raises
    ------
    TypeError
        If ``base_rates`` or ``control`` contains booleans, complex numbers or strings.
    ValueError
        If shapes, finiteness or ranges are violated, a scalar is passed for ``base_rates`` or
        ``control``, or ``group`` is not a 1-D integer array with values in ``0..G-1``.
    """
    e = _real(base_rates, "base rates")
    if (
        e.ndim != 1
        or e.size < 1
        or not np.isfinite(e).all()
        or (e < 0).any()
        or not (e > 0).any()
    ):
        raise ValueError(
            "base rates: a 1-D array of G >= 1 finite non-negative values, one positive"
        )
    n_groups = e.size
    if control is not None:
        m = _real(control, "degree-control factors")
        if (
            m.shape != (n_groups,)
            or not np.isfinite(m).all()
            or ((m < 0) | (m > 1)).any()
        ):
            raise ValueError(
                f"degree-control factors: {n_groups} finite values in [0, 1]"
            )
        e = e * m
    group = np.asarray(group)
    if (
        group.ndim != 1
        or group.dtype.kind not in "iu"
        or (group.size and (group.min() < 0 or group.max() >= n_groups))
    ):
        raise ValueError(
            f"group must be a 1-D integer array with values in 0..{n_groups - 1}"
        )
    return e[group]


def null_basis(R, d, tol=1e-10):
    r"""Orthonormal basis of the (numerical) null space of :math:`A = R D^{1/2}`.

    With the full SVD :math:`A = U \Sigma V^\top`, singular values
    :math:`\sigma_1 \ge \sigma_2 \ge \dots`, let

    .. math::

        k = \#\{\, i : \sigma_i > \text{tol}\cdot\max(\sigma_1, 1) \,\}.

    The returned :math:`N` has the right-singular vectors :math:`v_i`, :math:`i > k`, as columns
    (all of :math:`\operatorname{null}(A)` is included, because the full SVD is used).  The
    threshold :math:`\text{tol}\cdot\max(\sigma_1, 1)` is inherited from the earlier
    ``polynomial_cl`` implementation; it is not invariant under a rescaling of ``d`` when
    :math:`\sigma_1 < 1`.  With ``tol`` around ``1e-10`` the result is :math:`\operatorname{null}(A)` up to round-off
    (*exact*); with ``tol > 0`` large it additionally keeps directions that :math:`A` only weakly
    sees (*relaxed*), and then :math:`\|A N\|_2 \le \text{tol}\cdot\max(\sigma_1, 1)`.

    ``tol = 0`` is the strictest setting: every singular value that is not *exactly* zero counts
    as non-null, so round-off-level singular values of a rank-deficient :math:`A` are kept out
    of the basis and the result can be smaller than for ``tol = 1e-10``.  Directions with
    singular values below the threshold (very small :math:`\sigma_i`) are treated as null, which
    is what "numerical null space" means.  Repeated singular values are harmless: the
    individual columns of :math:`N` are then not unique, but :math:`P = N N^\top` is.

    Edge cases: with no reference rows (``R.shape[0] == 0``) the whole space is free and the
    identity is returned; if :math:`A` has full column rank the result has zero columns; if
    every rate is zero then :math:`A = 0`, all singular values vanish and a full orthonormal
    basis (:math:`P = I`) is returned.

    Parameters
    ----------
    R : ndarray, shape (m_R, p)
        Features of the reference examples, one per row.
    d : ndarray, shape (p,)
        Diagonal rates from :func:`diagonal_rates`, :math:`d \ge 0`.
    tol : float, default 1e-10
        Relative spectral tolerance.

    Returns
    -------
    ndarray, shape (p, p - k)
        Matrix :math:`N` with :math:`N^\top N = I`.

    Raises
    ------
    ValueError
        If ``tol`` is negative, NaN or infinite; if ``R`` is not a finite 2-D array with at
        least one column; or if ``d`` is not a finite non-negative vector of length ``R.shape[1]``.
    TypeError
        If ``tol`` is not a real number.
    """
    tol = _real(tol, "tol")
    if tol.ndim != 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite non-negative scalar")
    tol = float(tol)
    R, d = np.asarray(R), np.asarray(d)
    if np.ndim(R) == 2 and R.shape[1] < 1:
        raise ValueError("R must have at least one column")
    _check_matrix_and_rates(R.T, d, "R.T")
    if R.shape[0] == 0:
        return np.eye(R.shape[1])
    _, s, vt = np.linalg.svd(R * np.sqrt(d), full_matrices=True)
    return vt[int((s > tol * max(float(s[0]), 1.0)).sum()) :].T


def ordinary_step(G, d):
    r"""Ordinary rate-weighted step (returned **unsigned**).

    .. math::

        S = D\,G, \qquad \Delta W = -S .

    The caller *subtracts* the returned array from the weights.  Up to round-off this equals
    :func:`relaxed_step` with ``beta=1`` for every basis ``N``.

    Both inputs are converted to ``float64`` before they are multiplied, so integer arrays cannot
    overflow or wrap around.

    Parameters
    ----------
    G : ndarray, shape (p, K)
        Minibatch gradient.
    d : ndarray, shape (p,)
        Effective diagonal rates, :math:`d \ge 0`, learning rate included.

    Returns
    -------
    ndarray of float64, shape (p, K)
        The step :math:`S`.

    Raises
    ------
    TypeError
        If ``G`` or ``d`` contains booleans, complex numbers or non-numbers.
    ValueError
        If ``G`` is not a finite 2-D array, ``d`` is not a finite non-negative vector of length
        ``G.shape[0]``.
    """
    G, d = _real(G, "G"), _real(d, "d")
    _check_matrix_and_rates(G, d, "G")
    return d[:, None] * G


def relaxed_step(G, N, d, beta=0.0):
    r"""Rate-weighted, relaxed null-space step (returned **unsigned**).

    With :math:`P = N N^\top`, :math:`u = D^{1/2} G`, the function returns

    .. math::

        S = D^{1/2}\bigl[P + \beta (I - P)\bigr] D^{1/2} G
          = D^{1/2}\bigl[\beta\, u + (1 - \beta)\, N N^\top u\bigr],

    and the weight update is :math:`\Delta W = -S`: the caller *subtracts* the returned array from
    the weights.

    Special cases: :math:`\beta = 1` gives :math:`S = D G` (ordinary rate-weighted SGD) for every
    ``N``; :math:`\beta = 0` with an exact null basis gives :math:`R\,S = 0`, so the logits of the
    reference inputs are unchanged; rows with :math:`d_j = 0` of :math:`S` are zero.

    Parameters
    ----------
    G : ndarray, shape (p, K)
        Minibatch gradient.
    N : ndarray, shape (p, q)
        Orthonormal basis from :func:`null_basis` (:math:`N^\top N = I`).  Finiteness is
        checked; orthonormality is *assumed* and not verified, because checking it on every
        minibatch would be wasteful and the basis is built once per experience.
    d : ndarray, shape (p,)
        Effective diagonal rates, :math:`d \ge 0`, learning rate included.
    beta : float, default 0.0
        Relaxation in ``[0, 1]``.

    Returns
    -------
    ndarray, shape (p, K)
        The step :math:`S` (positive direction; :math:`\Delta W = -S`).

    Raises
    ------
    TypeError
        If ``beta`` is a boolean, complex number or string.
    ValueError
        If ``beta`` is not a scalar in ``[0, 1]``, the shapes of ``G`` (p, K), ``N`` (p, q) and
        ``d`` (p,) are incompatible, or ``N`` contains non-finite values.
    """
    beta = _real(beta, "beta")
    if beta.ndim != 0 or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be a scalar in [0, 1]")
    beta = float(beta)
    _check_matrix_and_rates(G, d, "G")
    if np.ndim(N) != 2 or np.shape(N)[0] != len(d):
        raise ValueError("N must be 2-D with N.shape[0] == len(d)")
    if not np.isfinite(N).all():
        raise ValueError("N must be finite")
    root = np.sqrt(d)[:, None]
    rg = root * G
    return root * (beta * rg + (1.0 - beta) * (N @ (N.T @ rg)))


def spectral_bound(R, d, N):
    r"""Achieved spectral norm :math:`\|A N\|_2` for :math:`A = R D^{1/2}`.

    This is the *computed* quantity, not the threshold used to build ``N``.  It is the
    :math:`\tau` that the relaxed residual bound (``math.tex``, Proposition 2) actually needs:
    ``N`` satisfies the spectral assumption A4 for any :math:`\tau \ge \|A N\|_2`.

    Parameters
    ----------
    R : ndarray, shape (m_R, p)
        Reference features, one example per row.
    d : ndarray, shape (p,)
        Effective diagonal rates, :math:`d \ge 0`.
    N : ndarray, shape (p, q)
        Null-space basis from :func:`null_basis`.

    Returns
    -------
    float
        :math:`\|A N\|_2`, or ``0.0`` if ``R`` has no rows or ``N`` has no columns.

    Raises
    ------
    ValueError
        If ``R`` is not a finite 2-D array with at least one column, ``d`` is not a finite
        non-negative vector of length ``R.shape[1]``, or ``N`` is not a finite 2-D array with
        ``N.shape[0] == len(d)``.
    """
    R, d, N = np.asarray(R), np.asarray(d), np.asarray(N)
    if np.ndim(R) == 2 and R.shape[1] < 1:
        raise ValueError("R must have at least one column")
    _check_matrix_and_rates(R.T, d, "R.T")
    _check_matrix_and_rates(N, d, "N")
    if R.shape[0] == 0 or N.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm((R * np.sqrt(d)) @ N, 2))


def free_dim(N, d):
    r"""Number of feature directions that can actually move under the exact step.

    .. math::

        r = \operatorname{rank}\bigl(D^{1/2} N\bigr),

    the dimension of the space containing the columns of :math:`\Delta W` (``math.tex``,
    Proposition 1(e)).  For an exact basis :math:`r = |S| - \operatorname{rank}(R_S)` with
    :math:`S = \{j : d_j > 0\}`; ``r == 0`` means the step cannot learn anything.

    Parameters
    ----------
    N : ndarray, shape (p, q)
        Null-space basis from :func:`null_basis`.
    d : ndarray, shape (p,)
        Diagonal rates.

    Returns
    -------
    int
        :math:`r`, or 0 if ``N`` has no columns.

    Raises
    ------
    ValueError
        If ``N`` is not 2-D or ``N.shape[0] != len(d)``.
    """
    _check_matrix_and_rates(N, d, "N")
    return int(np.linalg.matrix_rank(np.sqrt(d)[:, None] * N)) if N.size else 0
