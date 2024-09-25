import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import torch

Array = np.ndarray


class IndexedDict(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            keys = list(self.keys())
            if key < 0:
                key += len(keys)
            if key >= len(keys) or key < 0:
                raise IndexError("Index out of range")
            return self[keys[key]]
        else:
            return super().__getitem__(key)


def load_things_image_data(
    img_root: str,
    filter_behavior: bool = False,
    filter_plus: bool = False,
    return_indices: bool = False,
    filter_from_filenames: list[str] | None = None,
) -> list[int]:
    """Load image data from a folder"""

    def filter_image_names(filter_criterion, img_names):
        return [i for i, img in enumerate(img_names) if filter_criterion in img]

    image_paths = glob.glob(os.path.join(img_root, "**", "*.jpg"))

    # sort by class folder and then by image filenames both as strings
    image_paths = sorted(
        image_paths, key=lambda x: (str(Path(x).parent), str(os.path.basename(x)))
    )

    indices = np.arange(len(image_paths))
    img_names = [os.path.basename(img) for img in image_paths]

    if filter_from_filenames is not None:
        indices = [i for i, img in enumerate(img_names) if img in filter_from_filenames]

    if filter_behavior:
        indices = filter_image_names("01b", img_names)

    if filter_plus:
        indices = filter_image_names("plus", img_names)

    image_paths = np.array(image_paths)[indices]

    if return_indices:
        return indices, image_paths
    else:
        return image_paths


def load_concepts(path: str = "./data/misc/category_mat_manual.tsv") -> pd.DataFrame:
    concepts = pd.read_csv(path, encoding="utf-8", sep="\t")
    return concepts


def preprocess_embedding(embedding: Array) -> Array:
    embedding = np.maximum(0, embedding)
    argsort = np.argsort(-embedding.sum(0))
    embedding = embedding[:, argsort]
    return embedding


def relu(x: Array) -> Array:
    return np.maximum(0, x)


def transform_params(weights, scale):
    """We transform by (i) adding a positivity constraint and the sorting in descending order"""
    weights = relu(weights)
    sorted_dims = np.argsort(-np.linalg.norm(weights, axis=0, ord=1))

    weights = weights[:, sorted_dims]
    scale = scale[:, sorted_dims]
    d1, d2 = weights.shape
    # We transpose so that the matrix is always of shape (n_images, n_dims)
    if d1 < d2:
        weights = weights.T
        scale = scale.T
    return weights, scale, sorted_dims


def load_embedding(
    path: str,
    pruned: bool = True,
) -> Array:

    # Check if path ext is txt
    if path.endswith(".txt"):
        weights = np.loadtxt(path)
        scale = np.zeros_like(weights)
        return transform_params(weights, scale)[0]

    params = np.load(path)
    if params["method"] == "variational":
        key = "pruned_q_mu" if pruned else "q_mu"
        weights = params[key]
        key = "pruned_q_var" if pruned else "q_var"
        vars = params[key]
    else:
        weights = params["pruned_weights"]
        vars = np.zeros_like(weights)

    weights = transform_params(weights, vars)[0]
    return weights


def extract_model_module_from_path(path: str) -> list[str, str]:
    components = Path(path).parts
    model, module = components[-3], components[-2]
    return model, module


def find_files(base_dir: str, file_pattern: str) -> list[str]:
    file_path = os.path.join(base_dir, file_pattern)
    files = glob.glob(file_path, recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No files found for pattern {file_pattern} in {base_dir}"
        )
    return files


def load_data(
    base_dir: str,
    data_type: str = "embeddings",
    pruned: bool = True,
    filter_behavior: bool = False,
    feature_kwargs: dict = {"relu": False, "center": False, "zscore": False},
) -> dict:
    if data_type == "embeddings":
        file_pattern = "param*.npz"
    elif data_type == "features":
        file_pattern = "features*.npy"
    else:
        raise ValueError(f"Unrecognized data type {data_type}")

    files = find_files(base_dir, f"**/{file_pattern}")
    models, modules = zip(*map(extract_model_module_from_path, files))
    identifiers = [f"{model}.{module}" for model, module in zip(models, modules)]

    if data_type == "embeddings":
        data = [load_embedding(f, pruned) for f in files]
    elif data_type == "features":
        data = [load_dnn_activations(f, **feature_kwargs) for f in files]
    if filter_behavior:
        indices, _ = load_things_image_data(
            base_dir, filter_behavior=True, return_indices=True
        )
        if len(indices) == 0:
            raise ValueError("No indices found for filter_behavior")
        data = [d[indices] for d in data]

    return IndexedDict(zip(identifiers, data))


def load_dnn_activations(path, center=False, zscore=False, to_torch=False):
    if center and zscore:
        raise ValueError("Cannot center and zscore activations at the same time")

    if not path.endswith(".npy") and not path.endswith(".npz"):
        raise ValueError("Path must end with .npy or .npz")
    act = np.load(path)
    act = transform_activations(act, zscore=zscore, center=center)
    return torch.from_numpy(act) if to_torch else act


def transform_activations(act, relu=False, center=False, zscore=False):
    def center_act(x):
        return x - x.mean(0)

    def zscore_act(x):
        return (x - x.mean(0)) / x.std(0)

    """Transform activations"""
    if center and zscore:
        raise ValueError("Cannot center and zscore activations at the same time")
    if relu:
        act = relu(act)
    # We standardize or center AFTER the relu. neg vals. are then meaningful
    if center:
        act = center_act(act)
    if zscore:
        act = zscore_act(act)

    return act
