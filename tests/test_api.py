# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Public API surface: exactly the documented names; removed names stay removed."""

import importlib
import inspect

import pytest

import stable_cl

PUBLIC = [
    "diagonal_rates",
    "free_dim",
    "null_basis",
    "ordinary_step",
    "relaxed_step",
    "spectral_bound",
]


def test_public_names_are_exactly_the_documented_ones():
    assert sorted(stable_cl.__all__) == PUBLIC
    visible = [n for n in dir(stable_cl) if not n.startswith("_")]
    assert sorted(n for n in visible if n != "optimization") == PUBLIC


@pytest.mark.parametrize("name", PUBLIC)
def test_public_functions_are_documented_and_use_only_two_method_names(name):
    doc = inspect.getdoc(getattr(stable_cl, name))
    assert doc and len(doc) > 40
    assert "strategy_b" not in doc and "strategy\\_b" not in doc


REMOVED = [
    "null_step",
    "poly_features",
    "expand",
    "N_GROUPS",
    "train_task",
    "STRATEGIES",
    "run",
]


@pytest.mark.parametrize("name", REMOVED)
def test_removed_names_are_gone(name):
    assert not hasattr(stable_cl, name)


def test_the_polynomial_feature_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("stable_cl.features")


def test_the_update_module_has_no_polynomial_or_experiment_dependencies():
    source = inspect.getsource(importlib.import_module("stable_cl.optimization"))
    imports = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    banned = ("features", "scipy", "sklearn")
    assert not any(word in line for line in imports for word in banned)
    assert "N_GROUPS" not in source
