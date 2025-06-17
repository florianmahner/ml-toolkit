#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import itertools
import joblib
import h5py

import numpy as np

from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Union, Callable, Optional, Iterable

from ..analyses.metrics import (
    dot_similarity,
    cosine_similarity,
    euclidean_similarity,
)

from ..analyses.stats import (
    relu,
    positive_shift,
    standardize,
    center,
)


Array = np.ndarray


def get_similarity_matrix(X: Array, similarity: Union[str, Callable] = "dot") -> Array:
    similarity_functions = {
        "cosine": cosine_similarity,
        "dot": dot_similarity,
        "euclidean": euclidean_similarity,
    }
    print(f"Using similarity function {similarity}")
    if callable(similarity):
        S = similarity(X, X)
    elif isinstance(similarity, str):
        try:
            S = similarity_functions[similarity](X, X)
        except KeyError:
            raise ValueError(f"Similarity metric {similarity} not supported")
    return S


def get_nested_data(X: dict, nested_key: str) -> Array:
    keys = nested_key.split("-")
    for key in keys:
        X = X[key]
    return X


def transforms(X: Array, transform: str) -> Array:
    """Apply a transformation to the data."""
    transform_functions = {
        "relu": relu,
        "center": center,
        "zscore": standardize,
        "positive_shift": positive_shift,
        "none": lambda x: x,  # Identity function for 'none'
    }

    if transform not in transform_functions:
        raise ValueError(f"Transform {transform} not supported")

    print(f"Applying transform {transform}")
    return transform_functions[transform](X)


def load_domain(
    path: str, key: Optional[str] = None, use_memmap: bool = False
) -> Array:
    """Load features from a file can either be a .npy or .txt file or a dict"""
    endings = ["npy", "txt", "pkl", "h5", "p"]
    search = any(path.endswith(end) for end in endings)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")
    if not search:
        raise ValueError("Input file must be one of", endings)
    if re.search(r"(npy)$", path):
        return np.load(path, mmap_mode="r" if use_memmap else None)
    elif re.search(r"(txt)$", path):
        return np.loadtxt(path)
    # check if .pkl or .p file
    elif re.search(r"(pkl)$", path) or re.search(r"(p)$", path):
        data = joblib.load(path)
        if key:
            return get_nested_data(data, key)
        return data

    elif re.search(r"(h5)$", path):
        with h5py.File(path, "r") as handler:
            if key:
                data = get_nested_data(handler, key)
            else:
                data = handler
            return data[:]


@dataclass(init=True, repr=True)
class Sampler(object):
    file_path: str
    out_path: str
    n_samples: int
    k: int = 3
    train_fraction: float = 0.9
    seed: int = 42
    sample_type: str = "random"
    similarity: Union[str, Callable] = "dot"
    transforms: Union[str, Optional[Callable]] = None
    triplet_path: Optional[str] = None
    key: Optional[str] = None
    use_memmap: bool = False

    def __post_init__(self):
        if self.k not in [2, 3]:
            raise ValueError(
                "Only triplets (k=3) and pairwise (k=2) are supported at the moment"
            )
        if self.train_fraction > 1 or self.train_fraction < 0:
            raise ValueError("Train fraction must be between 0 and 1")
        if self.sample_type not in ["random", "adaptive", "on_the_fly"]:
            raise ValueError(
                "Sample type must be either 'random', 'adaptive', or 'on_the_fly'"
            )
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.file_path = str(self.file_path)
        self.X = load_domain(self.file_path, self.key, self.use_memmap)

        if self.use_memmap:
            print(
                "Using memmap and therefore not loading into memory and preprocessing data"
            )
        else:
            self.X = self.remove_nan(self.X)
            if callable(self.transforms):
                self.X = self.transforms(self.X)
            elif isinstance(self.transforms, str):
                self.X = transforms(self.X, self.transforms)

            if self.sample_type != "on_the_fly":
                self.S = get_similarity_matrix(self.X, similarity=self.similarity)

        self.n_objects, self.n_features = self.X.shape

    def remove_nan(self, X: Array) -> Array:
        nan_indices = np.isnan(X[:, :]).any(axis=1)
        X = X[~nan_indices]
        return X

    def softmax(self, z: Array) -> Array:
        proba = np.exp(z) / np.sum(np.exp(z))
        return proba

    def get_choice(self, S: Array, triplet: Array) -> Array:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))

        return choice

    def get_choice_on_the_fly(self, triplet: Array) -> Array:
        combs = list(itertools.combinations(triplet, 2))
        sims = [self.compute_similarity_on_the_fly(comb[0], comb[1]) for comb in combs]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))

        return choice

    def compute_similarity_on_the_fly(self, idx1: int, idx2: int) -> float:
        """Compute the similarity between two indices without constructing the similarity matrix"""
        if self.similarity == "cosine":
            return cosine_similarity(self.X[[idx1, idx2]])[0, 1]
        elif self.similarity == "dot":
            return dot_similarity(self.X[[idx1, idx2]])[0, 1]
        elif self.similarity == "euclidean":
            return euclidean_similarity(self.X[[idx1, idx2]])[0, 1]
        else:
            raise ValueError(f"Similarity metric {self.similarity} not supported")

    def log_softmax_scaled(self, X: Array, const: float = 0.0) -> Array:
        """see https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/"""
        X = X - const
        scaled_proba = np.exp(X) / np.sum(np.exp(X))
        scaled_log_proba = const + np.log(scaled_proba)
        return scaled_log_proba

    def find_triplet_argmax(self, S: Array, triplet: Array) -> Array:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))
        return choice

    def select_odd_one_outs(self, triplets: Iterable) -> Array:
        ooo = np.zeros((self.n_samples, self.k), dtype=int)
        for i, triplet in enumerate(triplets):
            ooo[i] = self.find_triplet_argmax(self.S, triplet)
        return ooo

    def random_combination(self, iterable: Iterable, r: int):
        """Random selection from itertools.combinations(iterable, r)"""
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(
            random.sample(range(n), r)
        )  # sorting prevents adding duplicates!
        return tuple(pool[i] for i in indices)

    def sample_adaptive(self):
        """Create similarity judgements."""
        unique_triplets = set()
        count = Counter()
        count.update({x: 0 for x in range(self.n_objects)})

        # At the start all classes have zero counts and we sample uniformly
        p_per_item = [1 / self.n_objects for _ in range(self.n_objects)]
        sample_idx, n_iter = 1, 1
        while sample_idx < self.n_samples + 1:
            n_iter += 1
            print(
                f"{n_iter} samples drawn, {sample_idx}/{self.n_samples} added", end="\r"
            )
            triplet = np.random.choice(
                range(self.n_objects), 3, replace=False, p=p_per_item
            )

            # Using this we can avoid duplicate triplets when adding to the set
            triplet.sort()
            triplet = tuple(triplet)

            # Add to set and increase count if triplet is still unique
            if triplet not in unique_triplets:
                count.update(triplet)
                unique_triplets.add(triplet)
                sample_idx += 1

            # Update histogram of each class and sample random choices with the inverse of the actual distribution
            if sample_idx % 100_000 == 0:
                sum_count = sum(count.values())
                sorted_count = sorted(count.items())

                # Make smallest proba the largest
                inverse_probas_per_item = [1 - s[1] / sum_count for s in sorted_count]

                # Correct uniform distribution
                norm_probas = [
                    float(i) / sum(inverse_probas_per_item)
                    for i in inverse_probas_per_item
                ]
                p_per_item = norm_probas

        ooo_choices = self.select_odd_one_outs(unique_triplets)
        return ooo_choices

    def sample_random(self) -> Array:
        """Sample triplets based on the similarity matrix."""
        unique_triplets = set()
        items = list(range(self.n_objects))
        n_triplets = 0
        while n_triplets < self.n_samples:
            print(f"{n_triplets}/{self.n_samples} added", end="\r")
            sample = self.random_combination(items, 3)
            unique_triplets.add(sample)
            n_triplets = len(unique_triplets)

        ooo_choices = self.select_odd_one_outs(unique_triplets)
        return ooo_choices

    def sample_pairs(self) -> Array:
        combs = np.array(list(itertools.combinations(range(self.n_objects), self.k)))
        random_sample = combs[
            np.random.choice(
                np.arange(combs.shape[0]), size=self.n_samples, replace=False
            )
        ]
        return random_sample

    def sample_on_the_fly(self) -> Array:
        """Sample triplets and find the odd one out on the fly without constructing the similarity matrix"""
        unique_triplets = set()
        items = list(range(self.n_objects))
        n_triplets = 0
        ooo_choices = np.zeros((self.n_samples, self.k), dtype=int)
        while n_triplets < self.n_samples:
            print(f"{n_triplets}/{self.n_samples} added", end="\r")
            sample = self.random_combination(items, 3)
            if sample not in unique_triplets:
                ooo_choices[n_triplets] = self.get_choice_on_the_fly(sample)
                unique_triplets.add(sample)
                n_triplets += 1

        return ooo_choices

    def train_test_split(self, ooo_choices: Union[list, Array]):
        """Split triplet data into train and test splits."""
        random.seed(0)
        np.random.shuffle(ooo_choices)
        N = ooo_choices.shape[0]
        frac = int(N * self.train_fraction)
        train_split = ooo_choices[:frac]
        test_split = ooo_choices[frac:]
        return train_split, test_split

    def run(self) -> None:
        self()

    def __call__(self) -> None:
        """Sample triplets and save them to disk."""
        if self.k == 2:
            choices = self.sample_pairs()

        # If the triplets are already provided, just load them and select the odd one out
        if self.triplet_path:
            unique_triplets = load_domain(self.triplet_path)
            unique_triplets = unique_triplets.astype(int)
            self.n_samples = unique_triplets.shape[0]
            choices = self.select_odd_one_outs(unique_triplets)
            fname = Path(self.triplet_path).stem + ".npy"
            with open(os.path.join(self.out_path, fname), "wb") as f:
                np.save(f, choices)
            return

        if self.sample_type == "adaptive":
            choices = self.sample_adaptive()
        elif self.sample_type == "on_the_fly":
            choices = self.sample_on_the_fly()
        else:
            choices = self.sample_random()

        train_split, test_split = self.train_test_split(choices)
        percentage = int(self.train_fraction * 100)
        with open(
            os.path.join(
                self.out_path,
                f"train_{percentage}.npy",
            ),
            "wb",
        ) as f:
            np.save(f, train_split)
        with open(
            os.path.join(self.out_path, f"test_{100 - percentage}.npy"), "wb"
        ) as f:
            np.save(f, test_split)
