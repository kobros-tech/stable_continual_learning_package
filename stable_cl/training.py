# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

r"""Softmax cross-entropy training of one experience with replay and optional null-space steps."""

from __future__ import annotations

import numpy as np

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
        Reference examples per old class used to build :math:`R`.
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
        If ``strategy`` is unknown or ``beta`` is outside ``[0, 1]``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
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
