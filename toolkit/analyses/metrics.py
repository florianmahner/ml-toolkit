#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities for distance and similarity metrics"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

Array = np.ndarray


AVAILABLE_METRICS = (
    "pearson",
    "cosine",
    "euclidean",
    "dot",
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
        "dot": dot_similarity,
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
        "dot": dot_distance,
        "manhattan": manhattan_distance,
        "gaussian_kernel": gaussian_kernel_distance,
    }
    return distance_functions[metric](x, y, **kwargs)


def pearson_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson similarity between two matrices, returning an n x n similarity matrix."""
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    xy = x_centered @ y_centered.T
    x_norms = np.linalg.norm(x_centered, axis=1, keepdims=True)
    y_norms = np.linalg.norm(y_centered, axis=1, keepdims=True)
    norm_product = x_norms @ y_norms.T
    s = xy / norm_product
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


def gaussian_kernel_similarity(x: Array, y: Array, sigma: float) -> float | Array:
    """Gaussian kernel similarity between two matrices."""
    distance = euclidean_distance(x, y)
    s = np.exp(-(distance**2) / (2 * sigma**2))
    return s


def gaussian_kernel_distance(x: Array, y: Array, sigma: float) -> float | Array:
    """Gaussian kernel distance between two matrices."""
    raise NotImplementedError(
        "Gaussian kernel distance is not implemented."
        "There is no intuitive interpretation of distance in the kernel space."
    )
