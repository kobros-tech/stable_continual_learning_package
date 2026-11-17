# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Shared fixtures: a synthetic degree vector for algebra tests and a small separable toy task."""

import numpy as np
import pytest

from stable_cl import poly_features


@pytest.fixture
def rng():
    """Deterministic generator for tests that need random matrices."""
    return np.random.default_rng(0)


@pytest.fixture
def deg():
    """Synthetic degree vector with 1, 5, 14, 20 features of degree 0..3 (p = 40)."""
    return np.repeat([0, 1, 2, 3], [1, 5, 14, 20])


@pytest.fixture
def rates():
    """Base rates and degree-control factors; degree 2 is frozen, degree 1 halved."""
    return [3.0, 2.0, 0.75, 1.5], [1.0, 0.5, 0.0, 1.0]


@pytest.fixture
def toy():
    """Ten Gaussian classes in 4 dimensions, degree-3 features; train set equals eval set."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 40)
    Phi, deg = poly_features(
        rng.normal(size=(10, 4))[y] * 1.5 + rng.normal(size=(len(y), 4))
    )
    return Phi, y, Phi, y, deg
