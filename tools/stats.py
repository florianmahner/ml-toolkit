#!/usr/bin/env python3

""" Statistical utilities"""

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

Array = np.ndarray
Tensor = torch.Tensor


# ------- Helper Functions for Array Transformations ------- #


def apply_transform(
    x: np.ndarray, transform: str | list[str], axis: int = 0, verbose: bool = False
) -> np.ndarray:
    transform_to_func = {
        "center": center,
        "standardize": standardize,
        "normalize_l2": normalize_l2,
        "normalize_l1": normalize_l1,
        "positive_shift": positive_shift,
        "softmax": softmax,
        "log_softmax": log_softmax,
        "log": log,
    }

    if isinstance(transform, str):
        if transform not in transform_to_func:
            raise ValueError(
                f"Transform {transform} not found. Available transforms: {transform_to_func.keys()}"
            )
        transform = [transform]

    for t in transform:
        if verbose:
            print(f"Applying {t}")
        x = transform_to_func[t](x, axis=axis)
    return x


def center(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Center the input array by subtracting the mean along the specified axis."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    return x - mean


def standardize(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize the input array by centering and scaling to unit variance."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True, ddof=0)
    std[std == 0] = 1  # Prevent division by zero
    return (x - mean) / std


def relu(x: Array) -> Array:
    """Apply the Rectified Linear Unit (ReLU) function element-wise."""
    return np.maximum(0, x)


def normalize_l2(x: Array, axis: int = 0, eps: float = 1e-8) -> Array:
    """Normalize the input array to have unit L2 norm."""
    x = np.asarray(x)
    norms = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    return x / (norms + eps)


def normalize_l1(x: Array, axis: int = 0, eps: float = 1e-8) -> Array:
    """Normalize the input array to have unit L1 norm."""
    x = np.asarray(x)
    norms = np.sum(x, axis=axis, keepdims=True)
    return x / (norms + eps)


def positive_shift(x: Array, **kwargs) -> Array:
    """Shift all values in the array to be non-negative."""
    return x - np.min(x)


def softmax(x: Array, axis: int = 1) -> Array:
    """Apply the softmax function along axis 1."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def log(x: Array) -> Array:
    """Log transform the input array."""
    return np.log(x)


def log_softmax(x: Array, axis: int = 1) -> Array:
    """Apply the log softmax function along axis 1."""
    return np.log(softmax(x, axis=axis))


def one_hot(x: Array, n_classes: int) -> Array:
    """Convert integer labels to one-hot encoded vectors."""
    return np.eye(n_classes)[x]


def log_likelihood(x: Array, mean: Array, std: Array) -> Array:
    """Calculate the log likelihood of the input array under a normal distribution."""
    return -0.5 * np.log(2 * np.pi * std**2) - ((x - mean) ** 2) / (2 * std**2)


def rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the residual sum of squares (RSS)."""
    return np.sum((y_true - y_pred) ** 2)


def aic_linear_regression(
    y_true: np.ndarray, y_pred: np.ndarray, num_predictors: int
) -> float:
    """
    Calculate the Akaike Information Criterion (AIC) for linear regression.
    """
    n = len(y_true)
    k = num_predictors + 1  # +1 for the intercept
    aic = n * np.log(rss(y_true, y_pred) / n) + 2 * k
    return aic


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


def compute_correlation_coeff(x: Array, y: Array, method: str = "pearson"):
    """Compute the correlation coefficient and p-value between two matrices."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    # Dictionary mapping method names to functions
    correlation_methods = {
        "pearson": pearsonr,
        "spearman": spearmanr,
    }

    # Validate the method
    if method not in correlation_methods:
        raise ValueError(f"Method {method} not supported.")

    # Compute the correlation using the selected method
    corr, pval = correlation_methods[method](x.flatten(), y.flatten())
    return corr, pval


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
    """Split-half reliability by randomly splitting the data into two halves and
    calculating the pearson correlation between the two halves.
    Args:
        data (list | Array): The array to calculate split half reliability for.
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


def fisher_z_transform(pearson_r: Array) -> Array:
    """Perform Fisher Z-transform on Pearson r values, ensuring valid input range."""
    pearson_r = np.asarray(pearson_r, dtype=np.float64)  # Ensure floating point type
    eps = 1e-7  # Slightly larger than before for better numerical stability

    # Replace invalid values (e.g., NaNs or infinite)
    pearson_r = np.where(np.isfinite(pearson_r), pearson_r, np.nan)
    pearson_r = np.clip(pearson_r, -1 + eps, 1 - eps)

    return np.arctanh(pearson_r)

def average_pearson_r(pearson_r: Array, axis: int = 0) -> Array:
    """Compute the average Pearson r after Fisher Z-transform and inverse transformation."""
    fisher_z = fisher_z_transform(pearson_r)
    mean_z = np.nanmean(fisher_z, axis=axis)

    return np.tanh(mean_z)


# ------- Helper Functions for Probability Densities  ------- #


def _normal_pdf_numpy(x: Array, mean: Array, std: Array) -> Array:
    """Compute the probability density function of a normal distribution."""
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def _normal_pdf_tensor(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Compute the probability density function of a normal distribution."""
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * torch.sqrt(2 * torch.pi))


def normal_pdf(
    x: Array | Tensor, mean: Array | Tensor, std: Array | Tensor
) -> Array | Tensor:
    """
    Compute the probability density function of a normal distribution.
    Args:
        x (Array | Tensor): The input values.
        mean (Array | Tensor): The mean of the distribution.
        std (Array | Tensor): The standard deviation of the distribution.
    """
    # Check input shapes
    if not (x.shape == mean.shape == std.shape):
        raise ValueError(
            f"Input arrays must have the same shape. "
            f"Shapes: x={x.shape}, mean={mean.shape}, std={std.shape}"
        )

        raise TypeError("All inputs must be of type Array or Tensor.")

    if not all(isinstance(arr, type(x)) for arr in (mean, std)):
        raise TypeError("All inputs must be of the same type (Array or Tensor).")

    if isinstance(x, Array):
        return _normal_pdf_numpy(x, mean, std)
    else:
        return _normal_pdf_tensor(x, mean, std)
