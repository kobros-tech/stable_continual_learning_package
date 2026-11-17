# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Stable continual learning with degree-balanced polynomial features.

Modules
-------
features
    Polynomial feature map and zero-initialised expansion.
optimization
    Diagonal degree-wise rates and the (relaxed) null-space step.
training
    Softmax cross-entropy training of one experience with replay.
experiment
    Data preparation, the five-experience protocol, metrics and the demo.
"""

from .experiment import (
    CONFIGS,
    DEGREES,
    TASKS,
    demo,
    load_digits_features,
    run,
    stratified_cut,
)
from .features import N_GROUPS, expand, poly_features
from .optimization import diagonal_rates, free_dim, null_basis, null_step
from .training import STRATEGIES, softmax, train_task

__all__ = [
    "CONFIGS",
    "DEGREES",
    "N_GROUPS",
    "STRATEGIES",
    "TASKS",
    "demo",
    "diagonal_rates",
    "expand",
    "free_dim",
    "load_digits_features",
    "null_basis",
    "null_step",
    "poly_features",
    "run",
    "softmax",
    "stratified_cut",
    "train_task",
]
