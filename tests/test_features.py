# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Polynomial feature map: count, ordering, monomials, degree-balanced scaling."""

from math import comb, sqrt

import numpy as np

from stable_cl import poly_features


def test_feature_count_matches_binomial_formula():
    x = np.random.default_rng(0).normal(size=(5, 16))
    Phi, deg = poly_features(x)
    assert Phi.shape == (5, comb(16 + 3, 3)) == (5, 969)
    assert [int((deg == k).sum()) for k in range(4)] == [1, 16, 136, 816]


def test_features_are_ordered_by_degree():
    _, deg = poly_features(np.ones((1, 4)))
    assert np.all(np.diff(deg) >= 0)
    assert deg[0] == 0 and deg[-1] == 3


def test_lower_degree_features_are_a_prefix_of_higher_degree_features():
    x = np.random.default_rng(1).normal(size=(7, 4))
    Phi3, deg3 = poly_features(x, degree=3)
    Phi2, deg2 = poly_features(x, degree=2)
    assert np.allclose(Phi3[:, : Phi2.shape[1]], Phi2)
    assert np.array_equal(deg3[: len(deg2)], deg2)


def test_monomials_and_scaling_for_two_inputs_degree_two():
    x = np.array([[2.0, 3.0]])
    Phi, deg = poly_features(x, degree=2)
    s1, s2 = 1 / sqrt(comb(2, 1)), 1 / sqrt(comb(3, 2))
    expected = [1.0, 2 * s1, 3 * s1, 4 * s2, 6 * s2, 9 * s2]
    assert np.array_equal(deg, [0, 1, 1, 2, 2, 2])
    assert np.allclose(Phi[0], expected)


def test_constant_feature_is_one():
    Phi, _ = poly_features(np.random.default_rng(2).normal(size=(6, 3)))
    assert np.array_equal(Phi[:, 0], np.ones(6))


def test_chunking_does_not_change_the_result():
    x = np.random.default_rng(3).normal(size=(20, 4))
    a, _ = poly_features(x, chunk=256)
    b, _ = poly_features(x, chunk=3)
    assert np.array_equal(a, b)
