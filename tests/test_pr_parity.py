# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""One strategy_b step against the polynomial_cl diagonal_control PR (skipped if it is not installed
or is a version without the diagonal-control API)."""

import numpy as np
import pytest

from stable_cl import diagonal_rates, null_basis, null_step, softmax

REQUIRED_API = {
    "polynomial_cl.continual.degree_rates": (
        "apply_degree_control",
        "degree_of_feature",
        "step_scale_for",
    ),
    "polynomial_cl.training.multiclass": ("fit_task", "null_space_basis"),
}


def _require_diagonal_control_api():
    """Import the PR functions, skipping (with the reason) if the installed package lacks them.

    A missing or older ``polynomial_cl`` is an environment condition, not a failure of this
    package.  Once every required name exists, any numerical disagreement still fails the test.
    """
    pytest.importorskip("polynomial_cl")
    found = {}
    for module, names in REQUIRED_API.items():
        mod = pytest.importorskip(module)
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            pytest.skip(
                f"installed polynomial_cl is not the diagonal_control PR: "
                f"{module} has no {', '.join(missing)}"
            )
        found.update({n: getattr(mod, n) for n in names})
    return found


def test_strategy_b_step_matches_polynomial_cl():
    api = _require_diagonal_control_api()
    apply_degree_control = api["apply_degree_control"]
    degree_of_feature = api["degree_of_feature"]
    step_scale_for = api["step_scale_for"]
    fit_task = api["fit_task"]
    null_space_basis = api["null_space_basis"]
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
