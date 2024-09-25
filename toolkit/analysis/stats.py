#!/usr/bin/env python3

""" Statistical utilities"""

import math
import torch
import numpy as np
from joblib import Parallel, delayed


Array = np.ndarray
Tensor = torch.Tensor


# ------- Helper Function for Variance Testing  ------- #


def effective_dimensionality(X: Array) -> float:
    """Calculate the effective dimensionality of an input array. It measure the
    effective number of dimensions that are relevant in the data (ie the number
    of dimensions needed to explain the variance in the data). This is done by
    calculating the ratio of the sum of the eigenvalues to the sum of the squared
    eigenvalues of the covariance matrix (ie PCA explained variance ratio)."""
    if X.ndim > 2:
        raise ValueError("Input array must be two-dimensional; " f"X.shape = {X.shape}")
    if X.ndim == 1:
        X = X[:, None]
    cov = np.cov(X.T, ddof=1)
    eigenvalues = np.linalg.eigvals(cov)
    return np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2)


# ------- Helper Function for Significance and Effect Size Testing  ------- #


def vectorized_pearsonr(x: Array, y: Array) -> float | Array:
    """Alterntive to scipy.stats.pearsonr that is vectorized over the first dimension for
    fast pairwise correlation calculation."""
    if x.shape != y.shape:
        raise ValueError(
            "Input arrays must have the same dimensions; "
            f"x.shape = {x.shape}, y.shape = {y.shape}"
        )
    if x.ndim < 2:
        x = x[:, None]
    if y.ndim < 2:
        y = y[:, None]
    n = x.shape[1]
    covariance = np.cov(x.T, y.T, ddof=1)
    x_std = np.sqrt(covariance[:n, :n].diagonal())
    y_std = np.sqrt(covariance[n:, n:].diagonal())
    pearson_r = covariance[:n, n:] / np.outer(x_std, y_std)
    return pearson_r


def spearman_brown_correction(reliability: float, split_factor: int) -> float:
    """Bute the Spearman-Brown prophecy formula.
    Args:
        reliability (float): The reliability of the original test.
        split_factor (int): The factor of the split (e.g. 2 for split-half reliability).
    """
    return split_factor * reliability / (1 + (split_factor - 1) * reliability)


def split_half_reliability(data: list | Array, num_splits: int = 1000) -> float | Array:
    """Bute the split-half reliability between two matrices.
    Args:
        x (list | Array): The array to apply split half.
        num_splits (int): The number of random splits for calculating reliability."""
    if isinstance(data, list):
        data = np.array(data)
    if data.ndim > 1:
        raise ValueError(
            "Input array must be one-dimensional; " f"data.shape = {data.shape}"
        )
    split_reliablity = np.zeros(num_splits)
    for n in range(num_splits):
        mask = np.random.choice([True, False], size=data.shape)
        split_reliablity[n] = vectorized_pearsonr(data[mask], data[~mask])

    average_reliability = average_pearson_r(split_reliablity)
    corrected_reliability = spearman_brown_correction(average_reliability, 2)
    return corrected_reliability


def reproducibility_across_embeddings(i, embeddings, odd_mask, even_mask, n_embeddings, n_dimensions):
    """Process a single embedding to calculate the split half reproducibility across all other embeddings and dimensions"""
    reproducibility_across_embeddings = np.zeros((n_embeddings, n_dimensions))
    best_matching_dimensions = np.zeros(n_dimensions)
    for j in range(n_embeddings):
        if i == j:
            continue

        emb_i = embeddings[i]
        emb_j = embeddings[j]

        corr_ij = vectorized_pearsonr(emb_i[odd_mask], emb_j[odd_mask])
        highest_corrs = np.argmax(corr_ij, axis=1)

        even_corrs = np.zeros(n_dimensions)
        for k in range(n_dimensions):
            base_even = emb_i[even_mask][:, k]
            dim_match = highest_corrs[k]
            comp_even = emb_j[even_mask][:, dim_match]
            even_corrs[k] = vectorized_pearsonr(base_even, comp_even)[0,0]

        best_matching_dimensions[j] = np.argmax(even_corrs)


        reproducibility_across_embeddings[j] = even_corrs

    z_transformed = np.arctanh(reproducibility_across_embeddings)
    average = np.mean(z_transformed, axis=0)
    back_transformed = np.tanh(average)
    return back_transformed, best_matching_dimensions


def split_half_reliability_across_runs(embeddings: np.ndarray, identifiers: str) -> dict[str, list]:
    """
    Compute the split-half reliability of each dimension for each model.

    The method is as follows:
    1. Split the data objects in half using an odd and even mask.
    2. For a given model run, iterate across all dimensions $i$:
        - Identify the dimension in all other models $k$ that has the highest correlation with dimension $i$,
          calculated using the odd-masked data.
        - For model $k$, correlate this identified dimension with dimension $i$ using the even-masked data.
    This process results in a sampling distribution of Pearson r coefficients across all other model seeds.
    The sampling distribution of Pearson r is then transformed using Fisher-z so that it becomes z-scored (i.e.,
    normally distributed). The mean of this sampling distribution is taken as the average z-transformed
    reliability score. Finally, this score is inverted to get the average Pearson r reliability score.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings in the shape of (n_embeddings, n_objects, n_dimensions).

    identifiers : str
        Identifiers of the embeddings, which could be model names or seeds.

    Returns
    -------
    dict
        A dictionary with identifiers as keys and Pearson r values across all dimensions.


    TODO - Take the pruned dimensions to calc fisher z!
    """
    assert (
        embeddings.ndim == 3
    ), "Embeddings must be 3-dimensional (n_embeddings, n_objects, n_dimensions)"
    n_embeddings, n_objects, n_dimensions = embeddings.shape

    np.random.seed(42)
    odd_mask = np.random.choice([True, False], size=n_objects)
    even_mask = np.invert(odd_mask)

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(reproducibility_across_embeddings)(
            i, embeddings, odd_mask, even_mask, n_embeddings, n_dimensions
        )
        for i in range(n_embeddings)
    )

    reliabs, dims = zip(*results)
    reliabs = [np.array(r) for r in reliabs]
    dims = [np.array(d) for d in dims]

    return reliabs, dims


def fisher_z_transform(pearson_r: list | Array) -> float | Array:
    """Perform Fisher Z-transform on Pearson r values."""
    return np.arctanh(pearson_r)


def average_pearson_r(pearson_r: list | Array, axis: int = 0) -> float | Array:
    """Bute the average of pearson r values, by first fisher z-transforming the values
    and then averaging them + transforming them back."""
    fisher_z = fisher_z_transform(pearson_r)
    mean_z = np.mean(fisher_z, axis=axis)
    return np.tanh(mean_z)


def cosine_similarity(x: Array, y: Array) -> float | Array:
    """Bute the cosine similarity between two matrices."""
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    xy = x @ y.T
    return xy / (x_norm * y_norm.T)


def dot_product_similarity(x: Array, y: Array) -> float | Array:
    """Bute the dot product similarity between two matrices."""
    return x @ y.T


def euclidean_similarity(x: Array, y: Array) -> float | Array:
    """Bute the euclidean similarity between two matrices."""
    distance = np.linalg.norm(x - y, axis=1)
    return 1 / (1 + distance)


# ------- Helper Functions for Probability Densities  ------- #


def normal_pdf(
    x: Array | Tensor, mean: Tensor | Array, std: Tensor | Array
) -> Tensor | Array:
    """Probability density function of a normal distribution."""
    if x.shape != mean.shape or x.shape != std.shape:
        raise ValueError(
            "Input arrays must have the same dimensions; "
            f"x.shape = {x.shape}, mean.shape = {mean.shape}, std.shape = {std.shape}"
        )
    # Check that all arrays are the same type
    if not all(isinstance(x, type(mean)) for x in [x, mean, std]):
        raise TypeError(
            "Input arrays must be of the same type; "
            f"x = {type(x)}, mean = {type(mean)}, std = {type(std)}"
        )
    if not isinstance(x, (Array, Tensor)):
        raise TypeError(
            "Input arrays must be of type Array or Tensor; " f"x = {type(x)}"
        )
    if isinstance(x, Array):
        pdf = np.exp(-((x - mean) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
    else:
        pdf = torch.exp(-((x - mean) ** 2) / (2 * std**2)) / (
            std * torch.sqrt(2 * math.pi)
        )
    return pdf
