# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Mathematical update for stable continual learning.

Modules
-------
features
    Polynomial feature map and zero-initialised expansion.
optimization
    Diagonal degree-wise rates and the (relaxed) null-space step.

Experiment, training and benchmark code is not part of this package. It was removed together with
its historical results; see the README for where the legacy source is preserved.
"""

from .features import N_GROUPS, expand, poly_features
from .optimization import diagonal_rates, free_dim, null_basis, null_step

__all__ = [
    "N_GROUPS",
    "diagonal_rates",
    "expand",
    "free_dim",
    "null_basis",
    "null_step",
    "poly_features",
]
