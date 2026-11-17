# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

r"""Softmax cross-entropy training of one experience with replay and optional null-space steps."""

from __future__ import annotations

import numpy as np

from .features import require_int
from .optimization import diagonal_rates, free_dim, null_basis, null_step

STRATEGIES = ("ordinary", "strategy_b")


def softmax(z):
    r"""Row-wise softmax, computed stably.

    .. math::

        \operatorname{softmax}(z)_{ik} = \frac{e^{z_{ik} - \max_l z_{il}}}{\sum_l e^{z_{il} - \max_l z_{il}}}.

    Subtracting the row maximum changes nothing mathematically and prevents overflow.

    Parameters
    ----------
    z : ndarray, shape (m, K)
        Logits.

    Returns
    -------
    ndarray, shape (m, K)
        Probabilities; every row is non-negative and sums to 1.
    """
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _check_task_arguments(W, X, y, Xm, ym, deg, n_seen, epochs, bs, n_ref):
    r"""Validate the data, shapes and counts passed to :func:`train_task`.

    Checks, in order: ``W`` is 2-D with ``p`` rows and ``K`` columns; ``1 <= n_seen <= K``;
    ``epochs >= 1``, ``bs >= 1``, ``n_ref >= 0``; ``X`` and ``Xm`` are 2-D with ``p`` columns;
    ``y`` and ``ym`` are 1-D integer arrays matching the number of rows of ``X`` and ``Xm``;
    ``X`` has at least one row; every label lies in ``[0, n_seen)``; ``deg`` covers all ``p``
    features.

    Parameters
    ----------
    W, X, y, Xm, ym, deg, n_seen, epochs, bs, n_ref
        As in :func:`train_task`.

    Returns
    -------
    tuple of int
        ``(n_seen, epochs, bs, n_ref)`` as Python integers.

    Raises
    ------
    TypeError
        If a count is not an integer or a label array is not of integer dtype.
    ValueError
        If any shape, count or label-range condition above is violated.
    """
    if np.ndim(W) != 2:
        raise ValueError(f"W must be 2-D (features x classes), got {np.ndim(W)}-D")
    p, K = W.shape
    n_seen = require_int(n_seen, "n_seen", 1)
    if n_seen > K:
        raise ValueError(f"n_seen={n_seen} exceeds the {K} classifier columns")
    epochs, bs = require_int(epochs, "epochs", 1), require_int(bs, "bs", 1)
    n_ref = require_int(n_ref, "n_ref", 0)
    for name, A in (("X", X), ("Xm", Xm)):
        if np.ndim(A) != 2 or A.shape[1] != p:
            raise ValueError(f"{name} must be 2-D with {p} columns (the rows of W)")
    for name, labels, A in (("y", y, X), ("ym", ym, Xm)):
        if np.ndim(labels) != 1 or len(labels) != len(A):
            raise ValueError(f"{name} must be 1-D with one label per row of its data")
        if np.asarray(labels).dtype.kind not in "iu":
            raise TypeError(f"{name} must be an integer array")
        if len(labels) and (labels.min() < 0 or labels.max() >= n_seen):
            raise ValueError(f"{name} contains labels outside [0, n_seen={n_seen})")
    if len(X) == 0:
        raise ValueError("X must contain at least one example")
    if np.ndim(deg) != 1 or len(deg) < p:
        raise ValueError(f"deg must be 1-D with at least {p} entries")
    return n_seen, epochs, bs, n_ref


def train_task(
    W,
    X,
    y,
    Xm,
    ym,
    deg,
    eta,
    rng,
    n_seen,
    *,
    control=None,
    strategy="ordinary",
    epochs=3,
    bs=128,
    n_ref=3,
    tol=1e-10,
    beta=0.0,
    info=None,
):
    r"""Train one experience **in place** with class-balanced minibatch SGD.

    Per minibatch :math:`B` the gradient of the mean cross-entropy over the seen classes
    (columns ``0..n_seen-1`` of ``W``) is

    .. math::

        G = \frac{1}{|B|} \sum_{i \in B} \phi(x_i)\,\bigl(p_i - e_{y_i}\bigr)^\top,
        \qquad p_i = \operatorname{softmax}\bigl(W_{:, :n_{\text{seen}}}^\top \phi(x_i)\bigr),

    and the seen columns are updated by

    .. math::

        W_{:, :n_{\text{seen}}} \leftarrow W_{:, :n_{\text{seen}}} - S,
        \qquad
        S = \begin{cases}
            D\,G & \text{ordinary (or no replay data)}, \\
            D^{1/2}[P + \beta(I - P)] D^{1/2} G & \text{strategy\_b},
        \end{cases}

    with :math:`D = \operatorname{diag}(d)` from :func:`~stable_cl.optimization.diagonal_rates`
    and :math:`P = N N^\top` from the null basis of :math:`R D^{1/2}`
    (:func:`~stable_cl.optimization.null_basis`).  :math:`R` is formed from the first ``n_ref``
    stored examples of every old class, and :math:`N` is built once per experience from the rate
    vector actually used.  The defaults of ``strategy_b`` (``beta=0``, ``tol=1e-10``) are the exact
    step of the ``polynomial_cl`` PR; ``tol`` and ``beta`` larger than that are relaxations.

    Minibatches are class-balanced over the current data and the replay memory: each class
    contributes ``max(1, bs // n_classes)`` examples (with replacement only when the class has
    fewer), and one epoch is ``ceil(len(data) / bs)`` such batches.  The first experience has no
    replay data and is therefore ordinary even for ``strategy="strategy_b"``.  If every effective
    rate is zero the experience is a valid no-op.

    Parameters
    ----------
    W : ndarray, shape (p, K)
        Classifier weights; modified in place and also returned.
    X, y : ndarray, shapes (n, p), (n,)
        Features and labels of the current experience (already cut to the active feature count).
    Xm, ym : ndarray, shapes (m, p), (m,)
        Replay memory features and labels (``m == 0`` for the first experience).
    deg : ndarray of int
        Degree of every feature; only ``deg[:p]`` is used.
    eta : float or array_like, shape (4,)
        Base rates per degree group.
    rng : numpy.random.Generator
        Source of minibatch randomness.
    n_seen : int
        Number of classes seen so far (columns of ``W`` that are trained).
    control : array_like, shape (4,), optional
        Degree-control factors in ``[0, 1]``.
    strategy : {"ordinary", "strategy_b"}
        Update rule.
    epochs, bs : int
        Epochs over the experience and nominal batch size.
    n_ref : int
        Reference examples per old class used to build :math:`R` (``>= 0``; with ``0`` there
        are no constraints and ``strategy_b`` reduces to the ordinary step).
    tol, beta : float
        Spectral tolerance and relaxation of the null-space step.
    info : dict, optional
        If given, filled with ``null_dim`` (columns of :math:`N`) and ``free_dim``
        (:func:`~stable_cl.optimization.free_dim`), both ``None`` when no null basis is built.

    Returns
    -------
    ndarray, shape (p, K)
        ``W`` itself.

    Raises
    ------
    ValueError
        If ``strategy`` is unknown, ``beta`` is outside ``[0, 1]``, ``epochs`` or ``bs`` is
        below 1, ``n_ref`` is negative, ``n_seen`` is not in ``[1, W.shape[1]]``, ``X`` / ``Xm``
        do not have ``W.shape[0]`` columns, ``X`` is empty, a label is outside
        ``[0, n_seen)``, or ``deg`` is shorter than ``W.shape[0]``.  Rate and control errors
        are those of :func:`~stable_cl.optimization.diagonal_rates`.
    TypeError
        If a count is not an integer or a label array is not of integer dtype.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    n_seen, epochs, bs, n_ref = _check_task_arguments(
        W, X, y, Xm, ym, deg, n_seen, epochs, bs, n_ref
    )
    count = W.shape[0]
    d = diagonal_rates(eta, deg[:count], control)
    if not d.any():
        return W
    Xa, ya = np.concatenate([X, Xm]), np.concatenate([y, ym])
    pools = [np.flatnonzero(ya == c) for c in np.unique(ya)]
    per = max(1, bs // len(pools))
    ref = [np.flatnonzero(ym == c)[:n_ref] for c in np.unique(ym)]
    ref = np.concatenate(ref) if ref else np.empty(0, int)
    N = null_basis(Xm[ref], d, tol) if strategy == "strategy_b" and len(ref) else None
    if info is not None:
        info.update(
            null_dim=None if N is None else N.shape[1],
            free_dim=None if N is None else free_dim(N, d),
        )
    for _ in range(epochs):
        for _ in range(-(-len(ya) // bs)):
            b = np.concatenate(
                [rng.choice(p, per, replace=len(p) < per) for p in pools]
            )
            xb = Xa[b]
            P = softmax(xb @ W[:, :n_seen])
            P[np.arange(len(b)), ya[b]] -= 1.0
            G = xb.T @ P / len(b)
            W[:, :n_seen] -= d[:, None] * G if N is None else null_step(G, N, d, beta)
    return W
