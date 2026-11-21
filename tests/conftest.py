# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Shared fixtures: a synthetic degree vector, rates and a deterministic generator for algebra tests."""

import numpy as np
import pytest


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
