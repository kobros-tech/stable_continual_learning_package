# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Class-incremental polynomial classifier: ordinary SGD vs strategy_b, with diagonal degree control.

One minibatch gradient G (p x K) is turned into a step with D = diag(d),
d_j = eta[deg j] * m[deg j] (base rate eta, degree-control factor m in [0, 1]):

    ordinary:    dW = -D G
    strategy_b:  dW = -D^(1/2) [P + beta (I - P)] D^(1/2) G,   P = N N^T,  N spans null(R D^(1/2))

R holds the features of the first n_ref stored examples of every old class.  With beta = 0 and
tol -> 0 this is exactly the strategy_b of the polynomial_cl diagonal_control PR.  tol > 0 and
beta > 0 are relaxations that belong to this package; they are not strategy_b.  See math.tex.
"""

from __future__ import annotations

import sys
import time
from itertools import combinations_with_replacement
from math import comb, sqrt

import numpy as np

TASKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
DEGREES = (1, 2, 3, 3, 3)  # active polynomial degree in each experience
N_GROUPS = 4  # degree groups 0..3
STRATEGIES = ("ordinary", "strategy_b")


def poly_features(x, degree=3, chunk=256):
    """Degree-balanced polynomial features (ordered by degree); returns features and degrees."""
    n = x.shape[1]
    exps = [np.zeros(n, int)]
    for d in range(1, degree + 1):
        exps += [
            np.bincount(c, minlength=n)
            for c in combinations_with_replacement(range(n), d)
        ]
    E = np.array(exps)
    deg = E.sum(1)
    scale = np.array([1.0 if d == 0 else 1 / sqrt(comb(n + d - 1, d)) for d in deg])
    Phi = np.concatenate(
        [
            np.prod(x[i : i + chunk, None, :] ** E[None], axis=2) * scale
            for i in range(0, len(x), chunk)
        ]
    )
    return Phi, deg


def expand(W, count):
    """Append zero rows: new features start at 0, so no logit changes."""
    return np.vstack([W, np.zeros((count - len(W), W.shape[1]), W.dtype)])


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def functional_distillation_gradient(X, W, W_teacher, n_old, n_seen):
    """Gradient of KL(teacher || current) on old reference examples.

    The teacher distribution covers old classes only. It is padded with
    zeros for new classes, while the student distribution covers all
    classes seen in the current experience.
    """
    if not 0 <= n_old <= n_seen <= W.shape[1]:
        raise ValueError("require 0 <= n_old <= n_seen <= number of classes")
    if W_teacher.shape != W.shape:
        raise ValueError("teacher and current weights must have the same shape")
    if X.ndim != 2 or X.shape[1] != W.shape[0]:
        raise ValueError("X and W have incompatible feature dimensions")

    if len(X) == 0 or n_old == 0:
        return np.zeros((W.shape[0], n_seen), dtype=W.dtype)

    teacher = softmax(X @ W_teacher[:, :n_old])
    student = softmax(X @ W[:, :n_seen])

    target = np.zeros_like(student)
    target[:, :n_old] = teacher

    return X.T @ (student - target) / len(X)

# ----------------------------------------------------------------------------- diagonal rates
def _real(v, what):
    """float64 array; booleans, complex numbers and non-numbers are rejected before any cast."""
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
    """d_j = eta[deg_j] * m[deg_j], the diagonal of D (D is never formed).

    eta: scalar or N_GROUPS finite non-negative base rates, at least one positive.
    control: None (= all ones) or N_GROUPS factors in [0, 1]; 0 freezes a degree group.
    The result may be all zero (a no-op); it is prefix-stable: features added by expansion
    take the rate of their own degree and old features keep theirs.
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


# ----------------------------------------------------------------------------- null-space step
def null_basis(R, d, tol=1e-10):
    """Orthonormal N spanning the right-singular directions of A = R diag(sqrt d) whose singular
    value is <= tol * max(sigma_max, 1) (the PR's convention; all of null(A) is included).
    tol ~ 1e-10 gives null(A)."""
    if R.shape[0] == 0:
        return np.eye(R.shape[1])
    _, s, vt = np.linalg.svd(R * np.sqrt(d), full_matrices=True)
    return vt[int((s > tol * max(float(s[0]), 1.0)).sum()) :].T


def null_step(G, N, d, beta=0.0):
    """Step D^(1/2) [P + beta (I - P)] D^(1/2) G (the update is minus this); beta=1 is D G."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    root = np.sqrt(d)[:, None]
    rg = root * G
    return root * (beta * rg + (1.0 - beta) * (N @ (N.T @ rg)))


def free_dim(N, d):
    """Number of feature directions that can move: rank(D^(1/2) N)."""
    return int(np.linalg.matrix_rank(np.sqrt(d)[:, None] * N)) if N.size else 0


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
    n_old=0,
    functional_lambda=0.0,
    info=None,
):
    """Train one experience in place. Defaults of strategy_b are those of the PR (exact, beta=0).

    Class-balanced minibatches over current + replay data; the first experience (no replay) is
    ordinary. If every effective rate is zero the experience is a valid no-op.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if not np.isfinite(functional_lambda) or functional_lambda < 0:
        raise ValueError("functional_lambda must be finite and non-negative")
    if not 0 <= n_old <= n_seen <= W.shape[1]:
        raise ValueError("require 0 <= n_old <= n_seen <= number of classes")
    count = W.shape[0]
    W_teacher = (
        W.copy()
        if functional_lambda > 0 and n_old > 0
        else None
    )
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

            if W_teacher is not None:
                old_batch = b >= len(X)
                if old_batch.any():
                    G += functional_lambda * functional_distillation_gradient(
                        xb[old_batch],
                        W,
                        W_teacher,
                        n_old,
                        n_seen,
                    )

            W[:, :n_seen] -= (
                d[:, None] * G
                if N is None
                else null_step(G, N, d, beta)
            )
    return W


# ----------------------------------------------------------------------------- experiment
def run(Phi_tr, y_tr, Phi_te, y_te, deg, seed, eta, *, mem=100, **kw):
    """Five class-incremental experiences; kw goes to train_task (control, strategy, tol, ...)."""
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
            n_old=2 * t,
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


def load_digits_features(split_seed=0):
    """Split before standardisation; use train statistics and a fixed random projection."""
    from sklearn.datasets import load_digits

    d = load_digits()
    order = np.random.default_rng(split_seed).permutation(len(d.data))
    cut = int(0.75 * len(order))
    tr, te = order[:cut], order[cut:]
    x = d.data / 16.0
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    x = (x - mu) / sd
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(64, 16)))
    Phi, deg = poly_features(x @ Q)
    return Phi[tr], d.target[tr], Phi[te], d.target[te], deg


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


# ----------------------------------------------------------------------------- checks
def _raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc:
        return True
    return False


def _toy(per=40, n=4):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), per)
    Phi, deg = poly_features(
        rng.normal(size=(10, n))[y] * 1.5 + rng.normal(size=(len(y), n))
    )
    return Phi, y, Phi, y, deg


def _check_pr_parity():
    """One step of the PR's fit_task (strategy_b) against null_step, if polynomial_cl is installed."""
    try:
        from polynomial_cl.continual.degree_rates import (
            apply_degree_control,
            degree_of_feature,
            step_scale_for,
        )
        from polynomial_cl.training.multiclass import fit_task, null_space_basis
    except ImportError:
        print("PR parity: skipped (polynomial_cl not importable)")
        return
    rng = np.random.default_rng(1)
    deg = degree_of_feature(4, 3)
    eta, m = [3.0, 2.0, 0.75, 1.5], [1, 0.5, 0.0, 1.0]
    d = diagonal_rates(eta, deg, m)
    X, y, R, W0 = (
        rng.normal(size=(30, len(deg))),
        rng.integers(0, 4, 30),
        rng.normal(size=(6, len(deg))),
        rng.normal(size=(len(deg), 4)),
    )
    base, mult = step_scale_for(apply_degree_control(np.array(eta), m), deg)
    W_pr = W0.copy()
    fit_task(
        W_pr,
        X,
        y,
        np.zeros((0, len(deg))),
        np.zeros(0, np.int64),
        epochs=1,
        learning_rate=base,
        batch_size=len(y),
        seed=0,
        null_basis=null_space_basis(R * np.sqrt(mult.astype(float))),
        group_scale=mult,
    )
    P = softmax(X @ W0)
    P[np.arange(len(y)), y] -= 1.0
    W_sm = W0 - null_step(X.T @ P / len(y), null_basis(R, d), d, 0.0)
    assert np.abs(W_pr - W_sm).max() < 1e-5  # the PR works in float32
    print("PR parity: strategy_b step matches polynomial_cl (float32 tolerance)")


def check():
    rng = np.random.default_rng(0)
    deg = np.repeat([0, 1, 2, 3], [1, 5, 14, 20])
    eta, m = [3.0, 2.0, 0.75, 1.5], [1.0, 0.5, 0.0, 1.0]
    # diagonal rates: definition, identity, freezing, prefix stability, validation
    d = diagonal_rates(eta, deg, m)
    assert np.array_equal(d, np.array(eta)[deg] * np.array(m)[deg])
    assert np.array_equal(diagonal_rates(eta, deg, [1] * 4), np.array(eta)[deg])
    assert np.array_equal(diagonal_rates(eta, deg[:20], m), d[:20])
    assert np.array_equal(diagonal_rates(2.0, deg), np.full(len(deg), 2.0))
    assert not diagonal_rates(
        eta, deg, [0] * 4
    ).any()  # all-zero effective rate is allowed
    for bad in ([1, 1, 1, -0.1], [1, 1, 1, np.nan], [1, 1, 1], [1, 1, 1, 1.5], 0.5):
        assert _raises(ValueError, diagonal_rates, eta, deg, bad)
    for bad in ([True, False, True, True], [1 + 0j, 1, 1, 1], ["a"] * 4):
        assert _raises(TypeError, diagonal_rates, bad, deg)  # as rates
        assert _raises(TypeError, diagonal_rates, eta, deg, bad)  # as control factors
    assert _raises(ValueError, diagonal_rates, [0, 0, 0, 0], deg)
    assert _raises(ValueError, diagonal_rates, eta, np.array([0.0, 1.0]))
    assert _raises(ValueError, diagonal_rates, eta, np.array([0, 4]))

    # strategy_b algebra (degree 2 frozen)
    R, G = rng.normal(size=(8, len(deg))), rng.normal(size=(len(deg), 4))
    root, S = np.sqrt(d)[:, None], d > 0
    N = null_basis(R, d)
    assert np.allclose(N.T @ N, np.eye(N.shape[1]), atol=1e-9)
    step = null_step(G, N, d, 0.0)
    assert np.abs(R @ step).max() < 1e-9  # reference logits unchanged
    assert not step[deg == 2].any()  # frozen rows do not move
    A = R * np.sqrt(d)  # independent pseudo-inverse form
    assert np.allclose(
        step, root * ((np.eye(len(deg)) - np.linalg.pinv(A) @ A) @ (root * G))
    )
    assert np.allclose(null_step(G, N, d, 1.0), d[:, None] * G)  # beta=1 is ordinary
    b = 0.3
    assert np.allclose(
        R @ null_step(G, N, d, b), b * R @ (d[:, None] * G)
    )  # exact null: beta * SGD
    rg, Prg = root * G, N @ (N.T @ (root * G))
    assert np.isclose(
        np.sum(G * null_step(G, N, d, b)), b * np.sum(rg**2) + (1 - b) * np.sum(Prg**2)
    )
    assert free_dim(N, d) == S.sum() - np.linalg.matrix_rank(R[:, S])
    Z = root * (N @ rng.normal(size=(N.shape[1], 4)))  # feasible perturbation of dW
    f = lambda U: np.sum(G * U) + 0.5 * np.sum(U[S] ** 2 / d[S][:, None])
    assert np.isclose(
        f(-step + Z) - f(-step), 0.5 * np.sum(Z[S] ** 2 / d[S][:, None])
    )  # argmin
    tol = 0.3  # relaxed: bound on logit change
    Nt = null_basis(R, d, tol)
    tau = tol * np.linalg.svd(R * np.sqrt(d), compute_uv=False)[0]
    assert (
        np.linalg.norm(R @ null_step(G, Nt, d, b) - b * R @ (d[:, None] * G))
        <= (1 - b) * tau * np.linalg.norm(rg) + 1e-9
    )
    assert null_basis(np.zeros((3, 5)), np.ones(5)).shape == (5, 5)
    assert null_basis(np.eye(5), np.ones(5)).shape == (5, 0)
    assert null_basis(np.zeros((0, 5)), np.ones(5)).shape == (5, 5)

    # whole-experience invariance on toy data: reference logits constant, frozen rows fixed
    Phi, y, _, _, tdeg = _toy()
    cur, old = (
        np.flatnonzero(np.isin(y, (2, 3))),
        np.flatnonzero(np.isin(y, (0, 1)))[::2],
    )
    ref = np.concatenate([np.flatnonzero(y[old] == c)[:3] for c in (0, 1)])
    W0 = rng.normal(size=(Phi.shape[1], 10)) * 0.1
    for strategy in STRATEGIES:
        W = W0.copy()
        train_task(
            W,
            Phi[cur],
            y[cur],
            Phi[old],
            y[old],
            tdeg,
            3.0,
            np.random.default_rng(1),
            4,
            control=[1, 0.5, 0, 1],
            strategy=strategy,
        )
        moved = np.abs(Phi[old][ref] @ (W - W0)[:, :4]).max()
        assert not (W - W0)[tdeg == 2].any() and not (W - W0)[:, 4:].any()
        assert (moved < 1e-9) if strategy == "strategy_b" else moved > 1e-3
    # all-ones control reproduces the uncontrolled run exactly; all-zero control is a no-op
    toy = _toy()
    for kw in ({}, {"strategy": "strategy_b"}):
        a = run(*toy, 0, 3.0, mem=20, **kw)["acc_matrix"]
        b_ = run(*toy, 0, 3.0, mem=20, control=[1.0] * 4, **kw)["acc_matrix"]
        assert np.array_equal(a, b_, equal_nan=True)
    assert (
        run(*toy, 0, 3.0, mem=20, control=[0.0] * 4, strategy="strategy_b")["free_dim"]
        == [None] * 5
    )
    _check_pr_parity()
    print("all checks passed")


if __name__ == "__main__":
    check() if "--check" in sys.argv else demo()
