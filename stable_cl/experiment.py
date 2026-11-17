# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

r"""Class-incremental digits experiment: data preparation, five experiences, metrics, demo.

Run the demo with ``python -m stable_cl``.
"""

from __future__ import annotations

import time

import numpy as np

from .features import N_GROUPS, expand, poly_features
from .optimization import diagonal_rates
from .training import train_task

TASKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
DEGREES = (1, 2, 3, 3, 3)  # active polynomial degree in each experience


def run(Phi_tr, y_tr, Phi_te, y_te, deg, seed, eta, *, mem=100, **kw):
    r"""Run the five class-incremental experiences and report accuracy and forgetting metrics.

    Experience :math:`t` introduces the classes ``TASKS[t]`` and activates the features of degree
    ``<= DEGREES[t]``; the classifier is expanded with zero rows when that count grows
    (:func:`~stable_cl.features.expand`; the logits on a probe batch are asserted unchanged).  The
    replay memory holds ``mem`` random training examples per class.  After each experience the
    accuracy :math:`A_{t,j}` on the test examples of experience :math:`j \le t` is measured
    among the :math:`2t + 2` classes seen so far.  With five experiences (:math:`t, j = 0..4`):

    .. math::

        \text{acc} = \tfrac15 \sum_{j} A_{4,j}, \qquad
        \text{forgetting} = \tfrac14 \sum_{j<4} \Bigl(\max_{j \le t \le 3} A_{t,j} - A_{4,j}\Bigr),

    .. math::

        \text{plasticity} = \tfrac15 \sum_t A_{t,t}, \qquad
        \text{newest} = A_{4,4}, \qquad
        \text{old} = \tfrac14 \sum_{j<4} A_{4,j}.

    Low forgetting is meaningful only alongside plasticity.

    Parameters
    ----------
    Phi_tr, y_tr : ndarray
        Training features (all ``p`` columns) and integer labels ``0..9``.
    Phi_te, y_te : ndarray
        Evaluation features and labels (test or validation).
    deg : ndarray of int, shape (p,)
        Degree of every feature.
    seed : int
        Seed of the generator used for the memory and the minibatches.
    eta : float or array_like, shape (4,)
        Base rates per degree group.
    mem : int, default 100
        Replay memory size per class.
    **kw
        Forwarded to :func:`~stable_cl.training.train_task` (``control``, ``strategy``, ``tol``,
        ``beta``, ``n_ref``, ``epochs``, ``bs``).

    Returns
    -------
    dict
        ``acc``, ``forgetting``, ``newest``, ``plasticity``, ``old`` (floats), ``immediate``
        (:math:`A_{t,t}` as a list), ``effective_rates`` (:math:`d` per degree group),
        ``null_dim`` and ``free_dim`` (per experience, ``None`` without a null basis) and
        ``acc_matrix`` (5x5 array, ``nan`` for :math:`j > t`).
    """
    rng = np.random.default_rng(seed)
    memory = {
        c: rng.choice(np.flatnonzero(y_tr == c), mem, replace=False) for c in range(10)
    }
    W = np.zeros((int((deg <= DEGREES[0]).sum()), 10))
    acc = np.full((5, 5), np.nan)
    null_dim, free = [], []
    for t, classes in enumerate(TASKS):
        count = int((deg <= DEGREES[t]).sum())
        if count > len(W):
            probe = Phi_te[:64, :count]
            before = probe[:, : len(W)] @ W
            W = expand(W, count)
            assert np.allclose(probe @ W, before, atol=1e-9)
        cur = np.flatnonzero(np.isin(y_tr, classes))
        old = (
            np.concatenate([memory[c] for c in range(2 * t)]) if t else np.empty(0, int)
        )
        info = {}
        train_task(
            W,
            Phi_tr[cur, :count],
            y_tr[cur],
            Phi_tr[old, :count],
            y_tr[old],
            deg,
            eta,
            rng,
            2 * t + 2,
            info=info,
            **kw,
        )
        null_dim.append(info.get("null_dim"))
        free.append(info.get("free_dim"))
        for j in range(t + 1):
            rows = np.flatnonzero(np.isin(y_te, TASKS[j]))
            pred = (Phi_te[rows, :count] @ W[:, : 2 * t + 2]).argmax(1)
            acc[t, j] = float(np.mean(pred == y_te[rows]))
    return {
        "acc": float(np.nanmean(acc[4])),
        "forgetting": float(np.mean([acc[j:4, j].max() - acc[4, j] for j in range(4)])),
        "newest": float(acc[4, 4]),
        "plasticity": float(np.mean(np.diag(acc))),
        "old": float(np.mean(acc[4, :4])),
        "immediate": np.diag(acc).tolist(),
        "effective_rates": diagonal_rates(
            eta, np.arange(N_GROUPS), kw.get("control")
        ).tolist(),
        "null_dim": null_dim,
        "free_dim": free,
        "acc_matrix": acc,
    }


def load_digits_features(split_seed=0, validation=False):
    r"""Split the digits, standardise with *fitting-set* statistics only, and expand polynomially.

    The 1797 examples are permuted by ``split_seed`` and cut 75 % / 25 % into ``tr`` / ``te``.
    Let :math:`(\text{fit}, \text{eval})` be

    * ``(tr, te)`` when ``validation=False`` (test protocol), and
    * ``(tr[:k], tr[k:])`` with :math:`k = \lfloor 0.8\,|tr| \rfloor` when ``validation=True``
      (a validation view cut from the training part; the test part is not used).

    The pixels, scaled by :math:`1/16`, are standardised with statistics computed on the *fit*
    rows only,

    .. math::

        \tilde{x} = \frac{x - \mu_{\text{fit}}}{\sigma_{\text{fit}} + 10^{-6}},

    and then applied unchanged to the evaluation rows, so evaluation examples never influence the
    preprocessing.  The result is projected to 16 dimensions by a fixed random orthonormal matrix
    (seed 0) and passed through :func:`~stable_cl.features.poly_features` (degree 3).

    Parameters
    ----------
    split_seed : int, default 0
        Seed of the permutation defining the split.
    validation : bool, default False
        Return a fit/validation view of the training part instead of train/test.

    Returns
    -------
    tuple
        ``(Phi_fit, y_fit, Phi_eval, y_eval, deg)`` with features of shape ``(n, 969)``.
    """
    from sklearn.datasets import load_digits

    d = load_digits()
    order = np.random.default_rng(split_seed).permutation(len(d.data))
    cut = int(0.75 * len(order))
    fit, ev = order[:cut], order[cut:]
    if validation:
        k = int(0.8 * len(fit))
        fit, ev = fit[:k], fit[k:]
    x = d.data / 16.0
    mu, sd = x[fit].mean(0), x[fit].std(0) + 1e-6
    x = (x - mu) / sd
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(64, 16)))
    Phi, deg = poly_features(x @ Q)
    return Phi[fit], d.target[fit], Phi[ev], d.target[ev], deg


CONFIGS = {
    "ordinary SGD": {},
    "strategy_b (beta=0, tol=1e-10, n_ref=3)": {"strategy": "strategy_b"},
    "relaxed (tol=.03, beta=.3, n_ref=50)": {
        "strategy": "strategy_b",
        "n_ref": 50,
        "tol": 0.03,
        "beta": 0.3,
    },
}


def demo(seeds=(3, 4, 5, 6, 7)):
    r"""Print ordinary SGD, exact ``strategy_b`` and the relaxation on one fixed split.

    Uses split seed 0, memory 100 per class and :math:`\eta = 3` for every degree group; reports
    the seed-mean of the metrics defined in :func:`run`.

    Parameters
    ----------
    seeds : iterable of int
        Training seeds to average over.
    """
    data, eta = load_digits_features(), [3.0] * 4
    print(
        f"fixed split seed 0; training seeds {list(seeds)}; memory 100/class; eta={eta}"
    )
    print("Low forgetting is meaningful only alongside plasticity.")
    for name, kw in CONFIGS.items():
        t0 = time.perf_counter()
        rs = [run(*data, seed, eta, **kw) for seed in seeds]
        a = {
            k: np.mean([r[k] for r in rs])
            for k in ("acc", "forgetting", "plasticity", "newest", "old")
        }
        print(
            f"{name:42s} acc={a['acc']:.3f} forget={a['forgetting']:.3f} plast={a['plasticity']:.3f} "
            f"newest={a['newest']:.3f} old={a['old']:.3f} time={(time.perf_counter() - t0) / len(seeds):.1f}s"
        )
