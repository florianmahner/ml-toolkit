#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities for distance and similarity metrics"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels

Array = np.ndarray


AVAILABLE_METRICS = (
    "pearson",
    "cosine",
    "euclidean",
    "linear",
    "manhattan",
    "gaussian_kernel",
)


def validate_metric(metric: str):
    if metric not in AVAILABLE_METRICS:
        raise ValueError(f"Metric {metric} not supported.")


def compute_similarity(x: Array, y: Array, metric: str, **kwargs) -> float | Array:
    """Compute similarity between two matrices."""
    validate_metric(metric)
    similarity_functions = {
        "pearson": pearson_similarity,
        "cosine": cosine_similarity,
        "euclidean": euclidean_similarity,
        "linear": dot_similarity,
        "manhattan": manhattan_similarity,
        "gaussian_kernel": gaussian_kernel_similarity,
    }
    return similarity_functions[metric](x, y, **kwargs)


def compute_distance(x: Array, y: Array, metric: str, **kwargs) -> float | Array:
    """Compute distance between two matrices."""
    validate_metric(metric)
    distance_functions = {
        "pearson": pearson_distance,
        "cosine": cosine_distance,
        "euclidean": euclidean_distance,
        "linear": dot_distance,
        "manhattan": manhattan_distance,
        "gaussian_kernel": gaussian_kernel_distance,
    }
    return distance_functions[metric](x, y, **kwargs)


def pearson_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson similarity between two matrices, returning an n x n similarity matrix.
    NOTE this is the same as np.corrcoef(x, y, rowvar=True)"""
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    xy = x_centered @ y_centered.T
    x_norms = np.linalg.norm(x_centered, axis=1, keepdims=True)
    y_norms = np.linalg.norm(y_centered, axis=1, keepdims=True)

    # Safely handle zero norms by setting invalid entries to 0
    norm_product = x_norms @ y_norms.T
    valid = norm_product > 0
    s = np.zeros_like(norm_product)
    s[valid] = xy[valid] / norm_product[valid]
    s = np.nan_to_num(s, nan=0.0)
    s = np.clip(s, a_min=-1, a_max=1)
    np.fill_diagonal(s, 1)
    return s


def pearson_distance(x: Array, y: Array) -> float | Array:
    """Pearson distance between two matrices."""
    return 1 - pearson_similarity(x, y)


def cosine_similarity(x: Array, y: Array) -> float | Array:
    """Cosine similarity between two matrices."""
    return 1 - cosine_distance(x, y)


def cosine_distance(x: Array, y: Array) -> float | Array:
    """Cosine distance between two matrices."""
    distance = pairwise_distances(x, y, metric="cosine")
    return distance


def euclidean_similarity(x: Array, y: Array) -> float | Array:
    """Euclidean similarity between two matrices."""
    distance = euclidean_distance(x, y)
    s = 1 / (1 + distance)
    return s


def euclidean_distance(x: Array, y: Array) -> float | Array:
    """Euclidean distance between two matrices
    see https://scikit-learn.org/dev/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    """
    distance = pairwise_distances(x, y, metric="euclidean")
    return distance


def dot_similarity(x: Array, y: Array) -> float | Array:
    """Dot product similarity between two matrices."""
    return x @ y.T


def dot_distance(x: Array, y: Array) -> float | Array:
    """Dot product distance between two matrices.

    This function is not implemented because the dot product is a measure of similarity,
    not distance. Consider using a different metric for distance calculations.
    """
    raise NotImplementedError(
        "The concept of 'dot product distance' is not meaningful. "
        "The dot product is a similarity measure, not a distance metric. "
        "This function will not be implemented."
    )


def manhattan_similarity(x: Array, y: Array) -> float | Array:
    """Manhattan similarity between two matrices."""
    distance = manhattan_distance(x, y)
    s = 1 / (1 + distance)
    return s


def manhattan_distance(x: Array, y: Array) -> float | Array:
    """Manhattan distance between two matrices."""
    distance = pairwise_distances(x, y, metric="manhattan")
    return distance


def gaussian_kernel_similarity(
    x: Array, y: Array, sigma: float | None = None
) -> float | Array:
    """Gaussian kernel similarity between two matrices aka RBF kernel.

    Params:
        x: Array, First input array.
        y: Array, Second input array.
        sigma: float, Controls the width of the kernel. Higher values lead to more
            smooth similarity functions. If not provided, it is estimated from the data.
    """
    if sigma is None:
        print("Sigma not provided, estimating from data using median distance")
        dist = pairwise_distances(x, metric="euclidean")
        median_dist = np.median(dist)
        sigma = median_dist

    # convert sigma to sklearn kernel parameter
    gamma = 1 / (2 * sigma**2)
    similarity = pairwise_kernels(x, y, metric="rbf", gamma=gamma)
    return similarity


def gaussian_kernel_distance(x: Array, y: Array, sigma: float = 1.0) -> float | Array:
    """Placeholder for Gaussian kernel distance between two matrices.

    Note: The Gaussian kernel is typically used as a similarity measure, not a distance.
    In the Gaussian kernel space, the concept of distance is not directly applicable
    or intuitive, as the kernel maps data into an infinite-dimensional space.
    """
    raise NotImplementedError(
        "Gaussian kernel distance is not implemented. The Gaussian kernel is used for "
        "similarity, not distance. Consider using Gaussian kernel similarity instead."
    )
