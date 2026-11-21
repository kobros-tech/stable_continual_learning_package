# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

r"""Mathematical update for stable continual learning.

The package implements one update with two named methods:

``ordinary``
    :func:`ordinary_step`, :math:`\Delta W = -D G`.
``relaxed``
    :func:`relaxed_step`, :math:`\Delta W = -D^{1/2}[P + \beta(I - P)]D^{1/2}G`, with the
    null-space basis from :func:`null_basis`.  :math:`\beta = 1` is ordinary, :math:`\beta = 0`
    is the exact step.

:func:`diagonal_rates` builds the effective per-feature rates :math:`d`, :func:`spectral_bound`
reports the achieved :math:`\|A N\|_2`, and :func:`free_dim` counts the directions that can move.
The package contains no feature maps, models, data or experiments.
"""

from .optimization import (
    diagonal_rates,
    free_dim,
    null_basis,
    ordinary_step,
    relaxed_step,
    spectral_bound,
)

__all__ = [
    "diagonal_rates",
    "free_dim",
    "null_basis",
    "ordinary_step",
    "relaxed_step",
    "spectral_bound",
]
