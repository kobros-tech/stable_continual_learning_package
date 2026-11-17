# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Zero-initialised expansion keeps every logit unchanged."""

import numpy as np

from stable_cl import expand, poly_features


def test_expansion_appends_zero_coefficients_and_keeps_old_ones():
    W = np.random.default_rng(0).normal(size=(5, 3))
    W2 = expand(W, 9)
    assert W2.shape == (9, 3)
    assert np.array_equal(W2[:5], W)
    assert not W2[5:].any()


def test_expansion_does_not_modify_its_input_and_keeps_dtype():
    W = np.ones((2, 2), dtype=np.float32)
    W2 = expand(W, 4)
    assert W.shape == (2, 2) and W2.dtype == np.float32


def test_expansion_to_same_size_is_a_copy_of_equal_value():
    W = np.random.default_rng(1).normal(size=(4, 2))
    assert np.array_equal(expand(W, 4), W)


def test_expansion_preserves_logits_on_real_features():
    x = np.random.default_rng(2).normal(size=(30, 4))
    Phi, deg = poly_features(x)
    old = int((deg <= 1).sum())
    W = np.random.default_rng(3).normal(size=(old, 10))
    before = Phi[:, :old] @ W
    for count in (int((deg <= 2).sum()), len(deg)):
        assert np.allclose(Phi[:, :count] @ expand(W, count), before, atol=1e-12)
