#!/usr/bin/env python3

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from numba import njit, prange
from scipy.stats import pearsonr, spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform


Array = np.ndarray
Tensor = torch.Tensor


@njit(parallel=False, fastmath=False)
def matmul(mat_a: Array, mat_b: Array) -> Array:
    I, K = mat_a.shape
    K, J = mat_b.shape
    mat_c = np.zeros((I, J))
    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                mat_c[i, j] += mat_a[i, k] * mat_b[k, j]
    return mat_c


@njit(parallel=False, fastmath=False)
def reconstruct_rsm(W: Array) -> Array:
    """convert weight matrix corresponding to the mean of each dim distribution for an object into a RSM"""
    N = W.shape[0]
    S = matmul(W, W.T)
    S_e = np.exp(S)  # exponentiate all elements in the inner product matrix S
    rsm = np.zeros((N, N))
    for i in prange(N):
        for j in prange(i + 1, N):
            for k in prange(N):
                if k != i and k != j:
                    rsm[i, j] += S_e[i, j] / (S_e[i, j] + S_e[i, k] + S_e[j, k])

    rsm /= N - 2
    rsm += rsm.T  # make similarity matrix symmetric
    np.fill_diagonal(rsm, 1)
    return rsm


def reconstruct_rsm_torch_batched(
    embedding: np.ndarray | torch.Tensor,
    verbose: bool = False,
    return_type: str = "numpy",
) -> Array | Tensor:
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding, dtype=torch.double)
    elif isinstance(embedding, torch.Tensor):
        embedding = embedding.double()
    else:
        raise ValueError("Embedding must be either a numpy array or a torch tensor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_matrix = torch.matmul(embedding, embedding.T)
    sim_matrix = sim_matrix.to(dtype=torch.double, device=device)
    sim_matrix.exp_()

    n_objects = sim_matrix.shape[0]
    indices = torch.triu_indices(n_objects, n_objects, offset=1)

    n_indices = indices.shape[1]
    batch_size = min(n_indices, 10_000)
    n_batches = (n_indices + batch_size - 1) // batch_size

    rsm = torch.zeros_like(sim_matrix).double()

    if verbose:
        pbar = tqdm(total=n_batches)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_indices)
        batch_indices = indices[:, start_idx:end_idx]
        i, j = batch_indices
        s_ij = sim_matrix[i, j]
        s_ik = sim_matrix[i, :]
        s_jk = sim_matrix[j, :]

        # This is a vectorized definition of 'for k in range(n_objects): if k != i and k != j'
        # By setting this to 0, we can vectorize the for loop by seeting the softmax
        # to 1 if i==k or j==k and then subtract 2 from the sum of the softmax.
        n = end_idx - start_idx
        n_range = np.arange(n)
        s_ik[n_range, i] = 0
        s_ik[n_range, j] = 0
        s_jk[n_range, j] = 0
        s_jk[n_range, i] = 0

        s_ij = s_ij.unsqueeze(1)
        softmax_ij = s_ij / (s_ij + s_jk + s_ik)
        proba_sum = softmax_ij.sum(1) - 2
        mean_proba = proba_sum / (n_objects - 2)
        rsm[i, j] = mean_proba

        if verbose:
            pbar.set_description(f"Batch {batch_idx+1}/{n_batches}")
            pbar.update(1)

    if verbose:
        pbar.close()

    rsm = rsm + rsm.T  # make similarity matrix symmetric
    rsm.fill_diagonal_(1)

    if return_type == "numpy":
        rsm = rsm.cpu().numpy()

    return rsm


def corrcoef_torch(X: Tensor, rowvar: bool = True) -> Tensor:
    """PyTorch implementation of the numpy.corrcoef function."""
    if not rowvar:
        X = X.T  # Transpose to make rows represent variables if rowvar is False
    X = X - torch.mean(X, dim=1, keepdim=True)
    n = X.size(1) - 1
    cov = (X @ X.T) / n
    std = torch.sqrt(torch.diag(cov)).unsqueeze(1)
    corr = cov / (std @ std.T)
    return corr


def pearson_correlation_matrix(F: Array | Tensor, **kwargs) -> Array:
    # """Compute similarity matrix based on correlation distance (on the matrix-level).
    if not isinstance(F, (Array, Tensor)):
        raise ValueError("F must be either a numpy array or a torch tensor.")
    if isinstance(F, Tensor):
        return corrcoef_torch(F, rowvar=True)

    return np.corrcoef(F, rowvar=True, **kwargs)


def compute_correlation_coeff(mat_a: Array, mat_b, method: str = "pearson", **kwargs):
    """Compute the correlation coefficient and p-value between two matrices."""
    if mat_a.shape != mat_b.shape:
        raise ValueError("A and B must have the same shape.")
    if method == "pearson":
        corr, pval = pearsonr(mat_a.flatten(), mat_b.flatten())
    elif method == "spearman":
        corr, pval = spearmanr(mat_a.flatten(), mat_b.flatten())
    else:
        raise ValueError(f"Method {method} not supported.")

    return corr, pval


def fill_diag(rsm: Array) -> Array:
    """Fill main diagonal of the RSM with ones"""
    assert np.allclose(rsm, rsm.T), "\nRSM is required to be a symmetric matrix\n"
    rsm[np.eye(len(rsm)) == 1.0] = 1.0

    return rsm


def compute_rdm(X, method="correlation"):
    assert method in ["correlation", "euclidean"]
    if method == "euclidean":
        rdm = squareform(pdist(X, metric="euclidean"))
    else:
        rsm = correlation_matrix(X)
        rdm = 1 - rsm

    return rdm


def compute_rsm(X, method="correlation"):
    assert method in ["correlation", "euclidean"]
    if method == "euclidean":
        rdm = squareform(pdist(X, metric="euclidean"))
        rsm = 1 - rdm
    else:
        rsm = correlation_matrix(X)

    return rsm


def correlation_matrix(F, a_min=-1.0, a_max=1.0):
    # """Compute dissimilarity matrix based on correlation distance (on the matrix-level).
    # Same as np.corrcoef(rowvar=True)"""
    F_c = F - F.mean(axis=1)[..., None]
    cov = F_c @ F_c.T
    l2_norms = np.linalg.norm(F_c, axis=1)  # compute vector l2-norm across rows
    denom = np.outer(l2_norms, l2_norms) + 1e-12

    corr_mat = cov / denom
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    corr_mat = corr_mat.clip(min=a_min, max=a_max)
    corr_mat = fill_diag(corr_mat)
    return corr_mat


def correlate_rsms(
    rsm_a: Array, rsm_b: Array, corr_type: str = "pearson", return_pval: bool = False
) -> float | tuple[float, float]:
    """Correlate the lower triangular parts of two rsms."""
    if corr_type not in ["pearson", "spearman"]:
        raise ValueError("Correlation must be 'pearson' or 'spearman'")

    rsm_a = fill_diag(rsm_a)
    rsm_b = fill_diag(rsm_b)

    triu_inds = np.triu_indices(len(rsm_a), k=1)
    corr_func = {"pearson": pearsonr, "spearman": spearmanr}[corr_type]
    corr, p = corr_func(rsm_a[triu_inds], rsm_b[triu_inds])

    return (corr, p) if return_pval else corr
