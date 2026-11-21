# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""M1 (Lean `expand_logits`): zero-padding the weights keeps every logit unchanged.

The check is plain numpy on arbitrary feature matrices. How a feature map is built, and that its old
features form a prefix of the new ones (assumption A1), is not part of this package; A1 is checked
where the feature map lives (`polynomial_cl_core`).
"""

import numpy as np
import pytest


def _zero_pad(W, count):
    """Test-local expansion `W -> [W; 0]` (the package has no expansion function)."""
    return np.vstack([W, np.zeros((count - W.shape[0], W.shape[1]))])


@pytest.mark.parametrize("old, mid, new", [(1, 17, 40), (5, 5, 9), (3, 4, 4)])
def test_zero_padding_preserves_logits(old, mid, new):
    rng = np.random.default_rng(old * 100 + new)
    Phi = rng.normal(size=(30, new))  # the first `old` columns are the old features
    W = rng.normal(size=(old, 10))
    before = Phi[:, :old] @ W
    for count in (mid, new):
        assert np.allclose(Phi[:, :count] @ _zero_pad(W, count), before, atol=1e-12)


def test_zero_padding_keeps_old_coefficients_and_starts_new_ones_at_zero():
    W = np.random.default_rng(0).normal(size=(5, 3))
    W2 = _zero_pad(W, 9)
    assert W2.shape == (9, 3)
    assert np.array_equal(W2[:5], W)
    assert not W2[5:].any()
