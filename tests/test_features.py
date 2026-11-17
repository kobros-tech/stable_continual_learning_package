# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Polynomial feature map: count, ordering, monomials, degree-balanced scaling."""

from math import comb, sqrt

import numpy as np
import pytest

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


def test_empty_input_gives_an_empty_feature_matrix_with_the_right_width():
    Phi, deg = poly_features(np.empty((0, 4)))
    full, full_deg = poly_features(np.ones((2, 4)))
    assert Phi.shape == (0, full.shape[1]) and Phi.dtype == np.float64
    assert np.array_equal(deg, full_deg)


def test_degree_zero_has_only_the_constant_feature():
    Phi, deg = poly_features(np.random.default_rng(0).normal(size=(3, 4)), degree=0)
    assert Phi.shape == (3, 1) and np.all(Phi == 1.0) and deg.tolist() == [0]


@pytest.mark.parametrize("chunk", [0, -1])
def test_non_positive_chunk_is_rejected(chunk):
    with pytest.raises(ValueError):
        poly_features(np.ones((2, 3)), chunk=chunk)


def test_negative_degree_is_rejected():
    with pytest.raises(ValueError):
        poly_features(np.ones((2, 3)), degree=-1)


@pytest.mark.parametrize("bad", [2.0, True, "3", None])
def test_non_integer_degree_or_chunk_is_rejected(bad):
    with pytest.raises(TypeError):
        poly_features(np.ones((2, 3)), degree=bad)
    with pytest.raises(TypeError):
        poly_features(np.ones((2, 3)), chunk=bad)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4), (4, 0)])
def test_wrong_input_dimensions_are_rejected(shape):
    with pytest.raises(ValueError):
        poly_features(np.ones(shape))


def test_non_numeric_input_is_rejected():
    with pytest.raises(TypeError):
        poly_features(np.array([["a", "b"]]))


# (dtype, magnitude): large enough that cubing in the input dtype would overflow or wrap.
@pytest.mark.parametrize(
    "dtype, big",
    [
        (np.int64, 10_000_000),  # 1e21 > 2**63
        (np.uint32, 4_000_000),  # 6.4e19 > 2**64
        (np.int32, 2_000),  # 8e9 > 2**31
        (np.float32, 1e15),  # 1e45 > float32 max
        (np.float16, 5_000),  # 1.25e11 > float16 max
    ],
)
def test_non_float64_inputs_are_computed_in_float64(dtype, big):
    x = np.array([[big, 3], [big // 2, 7]]).astype(dtype)
    Phi, deg = poly_features(x)
    expected, expected_deg = poly_features(x.astype(np.float64))
    assert Phi.dtype == np.float64
    assert np.isfinite(Phi).all()
    assert np.array_equal(Phi, expected) and np.array_equal(deg, expected_deg)


def test_integer_cube_does_not_wrap_around():
    Phi, _ = poly_features(np.array([[10_000_000]], dtype=np.int64))
    # n = 1: one monomial per degree, s_k = 1, so the features are 1, x, x^2, x^3.
    assert np.allclose(Phi[0], [1.0, 1e7, 1e14, 1e21])
