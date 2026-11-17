# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

r"""Diagonal degree-wise learning rates and the (relaxed) null-space step.

One minibatch gradient :math:`G \in \mathbb{R}^{p \times K}` is turned into a weight update using a
diagonal rate matrix :math:`D = \operatorname{diag}(d)`, :math:`d_j = \eta_{\deg(j)} m_{\deg(j)}`:

.. math::

    \text{ordinary:}\quad \Delta W = -D\,G,
    \qquad
    \text{strategy\_b:}\quad
    \Delta W = -D^{1/2}\bigl[P + \beta (I - P)\bigr] D^{1/2} G,
    \quad P = N N^\top .

Here :math:`R` holds the features of the stored reference examples of the old classes and the
columns of :math:`N` form an orthonormal basis of (the numerical) null space of
:math:`A = R D^{1/2}`.  With :math:`\beta = 0` and :math:`\text{tol} \to 0` this is exactly
``strategy_b`` of the ``polynomial_cl`` ``diagonal_control`` PR: reference logits :math:`R\,\Delta W`
do not change.  :math:`\text{tol} > 0` and :math:`\beta > 0` are relaxations specific to this
package.  See ``math.tex`` for the proofs.
"""

from __future__ import annotations

import numpy as np

from .features import N_GROUPS


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


def diagonal_rates(eta, deg, control=None):
    r"""Diagonal of the rate matrix :math:`D` (the matrix itself is never formed).

    .. math::

        d_j = \eta_{\deg(j)}\, m_{\deg(j)},
        \qquad D = \operatorname{diag}(d),
        \qquad D^{1/2} = \operatorname{diag}(\sqrt{d_j}),

    where :math:`\eta \in \mathbb{R}^{4}_{\ge 0}` are base rates per degree group (at least one
    positive) and :math:`m \in [0, 1]^{4}` are degree-control factors (:math:`m = \mathbf{1}` when
    ``control`` is ``None``).

    Conventions, all enforced:

    * ``control=None`` and ``control=[1, 1, 1, 1]`` give exactly :math:`d_j = \eta_{\deg(j)}`.
    * :math:`m_g = 0` freezes every feature of degree :math:`g`.
    * All-zero effective rates are allowed and describe a no-op experience.
    * The map is *prefix-stable*: ``d`` depends on ``deg[j]`` only, so features added by expansion
      take the rate of their own degree and old features keep theirs.
    * ``control`` is never broadcast: a scalar is rejected.

    Parameters
    ----------
    eta : float or array_like, shape (4,)
        Base rates, finite and non-negative, at least one positive.  A scalar is repeated for all
        four degree groups.
    deg : ndarray of int, shape (p,)
        Degree of each feature, values in ``0..3`` (see :func:`stable_cl.features.poly_features`).
    control : array_like, shape (4,), optional
        Degree-control factors in ``[0, 1]``.

    Returns
    -------
    ndarray, shape (p,)
        The vector ``d`` with :math:`d_j = \eta_{\deg(j)} m_{\deg(j)} \ge 0`.

    Raises
    ------
    TypeError
        If ``eta`` or ``control`` contains booleans, complex numbers or strings.
    ValueError
        If shapes, finiteness or ranges are violated, or ``deg`` is not a 1-D integer array
        with values in ``0..3``.
    """
    e = _real(eta, "learning rates")
    if e.ndim == 0:
        e = np.full(N_GROUPS, float(e))
    if (
        e.shape != (N_GROUPS,)
        or not np.isfinite(e).all()
        or (e < 0).any()
        or not (e > 0).any()
    ):
        raise ValueError(
            f"learning rates: {N_GROUPS} finite non-negative values, one positive"
        )
    if control is not None:
        m = _real(control, "degree-control factors")
        if (
            m.shape != (N_GROUPS,)
            or not np.isfinite(m).all()
            or ((m < 0) | (m > 1)).any()
        ):
            raise ValueError(
                f"degree-control factors: {N_GROUPS} finite values in [0, 1]"
            )
        e = e * m
    deg = np.asarray(deg)
    if (
        deg.ndim != 1
        or deg.dtype.kind not in "iu"
        or (deg.size and (deg.min() < 0 or deg.max() >= N_GROUPS))
    ):
        raise ValueError(
            f"deg must be a 1-D integer array with values in 0..{N_GROUPS - 1}"
        )
    return e[deg]


def null_basis(R, d, tol=1e-10):
    r"""Orthonormal basis of the (numerical) null space of :math:`A = R D^{1/2}`.

    With the full SVD :math:`A = U \Sigma V^\top`, singular values
    :math:`\sigma_1 \ge \sigma_2 \ge \dots`, let

    .. math::

        k = \#\{\, i : \sigma_i > \text{tol}\cdot\max(\sigma_1, 1) \,\}.

    The returned :math:`N` has the right-singular vectors :math:`v_i`, :math:`i > k`, as columns
    (all of :math:`\operatorname{null}(A)` is included, because the full SVD is used).  The
    threshold :math:`\text{tol}\cdot\max(\sigma_1, 1)` is the convention of the ``polynomial_cl``
    PR.  With ``tol`` around ``1e-10`` the result is :math:`\operatorname{null}(A)` up to round-off
    (*exact*); with ``tol > 0`` large it additionally keeps directions that :math:`A` only weakly
    sees (*relaxed*), and then :math:`\|A N\|_2 \le \text{tol}\cdot\max(\sigma_1, 1)`.

    Edge cases: with no reference rows (``R.shape[0] == 0``) the whole space is free and the
    identity is returned; if :math:`A` has full column rank the result has zero columns.

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
    """
    if R.shape[0] == 0:
        return np.eye(R.shape[1])
    _, s, vt = np.linalg.svd(R * np.sqrt(d), full_matrices=True)
    return vt[int((s > tol * max(float(s[0]), 1.0)).sum()) :].T


def null_step(G, N, d, beta=0.0):
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
        Orthonormal basis from :func:`null_basis` (:math:`N^\top N = I`).
    d : ndarray, shape (p,)
        Diagonal rates, :math:`d \ge 0`.
    beta : float, default 0.0
        Relaxation in ``[0, 1]``.

    Returns
    -------
    ndarray, shape (p, K)
        The step :math:`S` (positive direction; :math:`\Delta W = -S`).

    Raises
    ------
    ValueError
        If ``beta`` is outside ``[0, 1]``.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    root = np.sqrt(d)[:, None]
    rg = root * G
    return root * (beta * rg + (1.0 - beta) * (N @ (N.T @ rg)))


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
    """
    return int(np.linalg.matrix_rank(np.sqrt(d)[:, None] * N)) if N.size else 0
