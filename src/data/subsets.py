#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

Array = np.ndarray


def create_finetuning_splits(
    dataset: torch.utils.data.Dataset, embedding: Array, train_frac: float = 0.8
) -> tuple[list, list]:
    """Creates finetuning splits based on embedding of objects and their class
    labels"""
    train_set, val_set = [], []

    targets = torch.tensor(list(map(torch.tensor, dataset.targets)))
    rnd_perm = np.random.permutation(np.arange(len(dataset)))
    targets = targets[rnd_perm]
    embedding = embedding[rnd_perm]
    for i, y in enumerate(targets):
        if i < int(len(dataset) * train_frac):
            train_set.append((embedding[i], y))
        else:
            val_set.append((embedding[i], y))
    return train_set, val_set


def create_few_shot_subset(
    dataset: torch.utils.data.Dataset, samples_per_class: int = 5
) -> torch.utils.data.Subset:
    torch.manual_seed(0)

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "_labels"):
        targets = dataset._labels
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    classes = torch.unique(torch.tensor(targets))
    few_shot_indices = []

    for c in classes:
        class_indices = torch.where(torch.tensor(targets) == c)[0]
        selected_indices = class_indices[
            torch.randperm(len(class_indices))[:samples_per_class]
        ]
        few_shot_indices.append(selected_indices)
    few_shot_indices = torch.cat(few_shot_indices)
    few_shot_subset = torch.utils.data.Subset(dataset, few_shot_indices)

    return few_shot_subset
