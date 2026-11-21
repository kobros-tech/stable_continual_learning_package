# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Diagonal rates d_j = eta_deg(j) * m_deg(j): definition, identities, validation."""

import numpy as np
import pytest

from stable_cl import diagonal_rates


def test_rate_is_base_rate_times_control_factor(deg, rates):
    eta, m = rates
    assert np.array_equal(
        diagonal_rates(eta, deg, m), np.array(eta)[deg] * np.array(m)[deg]
    )


def test_all_ones_control_reproduces_uncontrolled_rates(deg, rates):
    eta, _ = rates
    assert np.array_equal(diagonal_rates(eta, deg, [1] * 4), np.array(eta)[deg])
    assert np.array_equal(diagonal_rates(eta, deg, [1] * 4), diagonal_rates(eta, deg))


@pytest.mark.parametrize("bad", [2.0, 2, np.float64(2.0)])
def test_scalar_base_rate_is_rejected_and_never_broadcast(deg, bad):
    with pytest.raises(ValueError):
        diagonal_rates(bad, deg)


@pytest.mark.parametrize("n_groups", [1, 2, 3, 6])
def test_group_count_is_inferred_from_the_base_rates(n_groups):
    rng = np.random.default_rng(n_groups)
    eta = rng.uniform(0.5, 3.0, n_groups)
    m = rng.uniform(0.0, 1.0, n_groups)
    group = rng.integers(0, n_groups, 30)
    assert np.array_equal(diagonal_rates(eta, group, m), (eta * m)[group])
    assert np.array_equal(diagonal_rates(eta, group), eta[group])


def test_group_index_must_fit_the_inferred_group_count():
    with pytest.raises(ValueError):
        diagonal_rates([1.0, 2.0, 3.0], np.array([0, 3]))
    assert np.array_equal(diagonal_rates([1.0, 2.0, 3.0], np.array([2, 0])), [3.0, 1.0])


def test_control_length_must_match_the_group_count():
    with pytest.raises(ValueError):
        diagonal_rates([1.0, 2.0, 3.0], np.array([0, 1, 2]), [1, 1, 1, 1])
    with pytest.raises(ValueError):
        diagonal_rates([1.0, 2.0, 3.0, 4.0], np.array([0, 1, 2]), [1, 1, 1])


def test_boolean_group_index_is_rejected(rates):
    with pytest.raises(ValueError):
        diagonal_rates(rates[0], np.array([True, False]))


def test_zero_factor_freezes_a_degree(deg, rates):
    eta, m = rates
    d = diagonal_rates(eta, deg, m)
    assert not d[deg == 2].any() and d[deg == 3].all()


def test_rates_are_prefix_stable(deg, rates):
    eta, m = rates
    assert np.array_equal(
        diagonal_rates(eta, deg[:20], m), diagonal_rates(eta, deg, m)[:20]
    )


def test_all_zero_effective_rates_are_allowed(deg, rates):
    eta, _ = rates
    assert not diagonal_rates(eta, deg, [0] * 4).any()


@pytest.mark.parametrize(
    "bad", [[1, 1, 1, -0.1], [1, 1, 1, np.nan], [1, 1, 1], [1, 1, 1, 1.5], 0.5]
)
def test_invalid_control_factors_are_rejected(deg, rates, bad):
    with pytest.raises(ValueError):
        diagonal_rates(rates[0], deg, bad)


@pytest.mark.parametrize("bad", [[-1, 1, 1, 1], [1, 1, 1, -0.1], [1, -1e-9, 1, 1]])
def test_negative_base_rates_are_rejected(deg, bad):
    # Domain assumed by Lean's `rate_nonneg`: see docs/traceability.md, A0.
    with pytest.raises(ValueError):
        diagonal_rates(bad, deg)


@pytest.mark.parametrize(
    "bad", [[True, False, True, True], [1 + 0j, 1, 1, 1], ["a"] * 4]
)
def test_non_real_values_are_rejected_for_rates_and_factors(deg, rates, bad):
    with pytest.raises(TypeError):
        diagonal_rates(bad, deg)
    with pytest.raises(TypeError):
        diagonal_rates(rates[0], deg, bad)


def test_all_zero_base_rates_are_rejected(deg):
    with pytest.raises(ValueError):
        diagonal_rates([0, 0, 0, 0], deg)


@pytest.mark.parametrize(
    "bad", [np.array([0.0, 1.0]), np.array([0, 4]), np.array([[0, 1]])]
)
def test_invalid_degree_vectors_are_rejected(rates, bad):
    with pytest.raises(ValueError):
        diagonal_rates(rates[0], bad)
