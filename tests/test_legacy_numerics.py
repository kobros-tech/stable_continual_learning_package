# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""The S1-head numerics are unchanged by the API change.

`tests/golden/legacy_numerics.json` was produced by the implementation as it stood at the end of S1
(functions `diagonal_rates`, `null_basis`, `null_step`, `free_dim`; threshold `tol * max(sigma_1, 1)`).
Inputs are stored in the file. `N` itself is sign- and rotation-ambiguous, so the projector `N N^T`
is compared. The threshold rule is what a later, separate PR changes; this file pins it until then.
"""

import json
import pathlib

import numpy as np
import pytest

from stable_cl import (
    diagonal_rates,
    free_dim,
    null_basis,
    ordinary_step,
    relaxed_step,
)

GOLDEN = pathlib.Path(__file__).parent / "golden" / "legacy_numerics.json"
CASES = json.loads(GOLDEN.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_new_api_reproduces_the_s1_head_numbers(case):
    R = np.array(case["R"]).reshape(len(case["R"]), len(case["group"]))
    G, group = np.array(case["G"]), np.array(case["group"])
    d = diagonal_rates(case["base_rates"], group, case["control"])
    assert np.array_equal(d, np.array(case["d"]))
    N = null_basis(R, d, case["tol"])
    assert N.shape[1] == case["null_dim"]
    assert np.allclose(N @ N.T, np.array(case["P"]), atol=1e-10)
    step = relaxed_step(G, N, d, case["beta"])
    assert np.allclose(step, np.array(case["step"]), rtol=1e-9, atol=1e-12)
    assert free_dim(N, d) == case["free_dim"]
    if case["beta"] == 1.0:
        expected = np.array(case["step"])
        assert np.allclose(ordinary_step(G, d), expected, rtol=1e-9, atol=1e-12)
