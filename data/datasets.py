#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import glob
import json
import torch.utils
import torchvision
from scipy.io import loadmat
from dataclasses import dataclass

from torchvision.datasets import MNIST
from sklearn.datasets import load_iris as sklearn_load_iris
from sklearn.datasets import load_digits as sklearn_load_digits
from typing import Callable


import numpy as np

from PIL import Image
from pathlib import Path

Array = np.ndarray
Compose = torchvision.transforms.Compose


def get_computer_vision_dataset(
    dataset_name: str = "cifar100",
    dataset_locator: str = "/SSD/datasets",
    transform: Compose | None = None,
    **kwargs,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    factory = DatasetFactory(dataset_locator)
    train_dataset = factory.get_dataset(
        dataset_name, train=True, transform=transform, **kwargs
    )
    test_dataset = factory.get_dataset(
        dataset_name, train=False, transform=transform, **kwargs
    )

    return train_dataset, test_dataset


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list[str], transform: Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class CIFAR100Coarse(torchvision.datasets.CIFAR100):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super(CIFAR100Coarse, self).__init__(
            root, train, transform, target_transform, download
        )

        # update labels
        coarse_labels = np.array(
            [
                4,
                1,
                14,
                8,
                0,
                6,
                7,
                7,
                18,
                3,
                3,
                14,
                9,
                18,
                7,
                11,
                3,
                9,
                7,
                11,
                6,
                11,
                5,
                10,
                7,
                6,
                13,
                15,
                3,
                15,
                0,
                11,
                1,
                10,
                12,
                14,
                16,
                9,
                11,
                5,
                5,
                19,
                8,
                8,
                15,
                13,
                14,
                17,
                18,
                10,
                16,
                4,
                17,
                4,
                2,
                0,
                17,
                4,
                18,
                17,
                10,
                3,
                2,
                12,
                12,
                16,
                12,
                1,
                9,
                19,
                2,
                10,
                0,
                1,
                16,
                12,
                9,
                13,
                15,
                13,
                16,
                19,
                2,
                4,
                6,
                19,
                5,
                5,
                8,
                19,
                18,
                1,
                2,
                15,
                6,
                0,
                17,
                8,
                14,
                13,
            ]
        )
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [
            ["beaver", "dolphin", "otter", "seal", "whale"],
            ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            ["orchid", "poppy", "rose", "sunflower", "tulip"],
            ["bottle", "bowl", "can", "cup", "plate"],
            ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            ["clock", "keyboard", "lamp", "telephone", "television"],
            ["bed", "chair", "couch", "table", "wardrobe"],
            ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            ["bear", "leopard", "lion", "tiger", "wolf"],
            ["bridge", "castle", "house", "road", "skyscraper"],
            ["cloud", "forest", "mountain", "plain", "sea"],
            ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            ["fox", "porcupine", "possum", "raccoon", "skunk"],
            ["crab", "lobster", "snail", "spider", "worm"],
            ["baby", "boy", "girl", "man", "woman"],
            ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
        ]


class DatasetFactory:
    def __init__(self, dataset_locator: str):
        self.dataset_locator = dataset_locator

    def get_dataset(
        self, dataset_name: str, train: bool, transform: Compose | None = None, **kwargs
    ):
        dataset_map = {
            "cifar100coarse": CIFAR100Coarse,
            "cifar10": torchvision.datasets.CIFAR10,
            "cifar100": torchvision.datasets.CIFAR100,
            "dtd": torchvision.datasets.DTD,
            "flickr30k": torchvision.datasets.Flickr30k,
            "mnist": torchvision.datasets.MNIST,
            "fashion": torchvision.datasets.FashionMNIST,
            "sun397": torchvision.datasets.SUN397,
            "ILSVRC": ImageNetKaggle,
            "things": ThingsDataset,
            "orl": ORL,
            "stl10": torchvision.datasets.STL10,
        }

        dataset_args = {
            "cifar100": {"train": train},
            "cifar10": {"train": train},
            "cifar100coarse": {"train": train},
            "sun397": {},
            "dtd": {"split": "train" if train else "test"},
            "flickr30k": {"split": "train" if train else "val"},
            "mnist": {"train": train},
            "fashion": {"train": train},
            "ILSVRC": {"train": train},
            "things": {"tripletize": False},
            "orl": {},
            "stl10": {"split": "train" if train else "test"},
        }

        if dataset_name not in dataset_map:
            raise ValueError(
                f"Unknown dataset {dataset_name}. Available datasets are: {list(dataset_map.keys())}"
            )

        dataset_class = dataset_map[dataset_name]
        args = {
            **dataset_args[dataset_name],
            **kwargs,
            "download": True,
            "transform": transform,
        }

        dataset = dataset_class(
            root=os.path.join(self.dataset_locator, dataset_name), **args
        )

        if dataset_name == "sun397":
            split_type = "Training_01.txt" if train else "Testing_01.txt"
            split_file = os.path.join(self.dataset_locator, "sun397", split_type)
            dataset = filter_sun397(dataset, split_file)

        return dataset


class ThingsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        triplets: Array | None = None,
        tripletize: bool = False,
        transform: torchvision.transforms.Compose | None = None,
        **kwargs,
    ):
        self.root = root
        self.transform = transform
        self.triplets = triplets
        self.tripletize = tripletize
        self.image_paths = glob.glob(os.path.join(self.root, "*", "*01b.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.root}")

        self.find_category_identifiers()

    def load_images_into_memory(self, image_paths):
        # TODO Pre allocate the things image on memory for faster loading on the fly (should be much faster!!)
        images = []
        for path in image_paths:
            image = Image.open(path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return images

    def find_category_identifiers(self):
        classes = []
        for path in self.image_paths:
            cls = Path(path).parent.name
            classes.append((cls, path))
        classes = sorted(classes, key=lambda x: x[0])
        self.path_to_idx = {path: i for i, (cls, path) in enumerate(classes)}
        self.idx_to_path = {i: path for i, (cls, path) in enumerate(classes)}
        self.classes = classes

    def __len__(self):
        return len(self.triplets) if self.tripletize else len(self.image_paths)

    def get_image_from_path(self, idx: int):
        path = self.idx_to_path[idx]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, idx

    def get_image_triplet(self, idx):
        images = map(self.get_image_from_path, self.triplets[idx])
        return images, self.triplets[idx]

    def __getitem__(self, idx: int):
        if self.tripletize:
            return self.get_image_triplet(idx)
        return self.get_image_from_path(idx)


class ImageNetKaggle(torch.utils.data.Dataset):
    """ImageNet training dataset based on the downloaded kaggle dataset. The synset
    mapping is used to map the class labels to the actual class names and is the
    way the dataset is organized. The dataset is organized in the following way:
    ```
    root
    ├── Data
    │   └── CLS-LOC
    │       └── train
    │           ├── n01440764 (synset)
    │           │   ├── n01440764_10026.JPEG
    │           │   ├── n01440764_10027.JPEG
    │           │   ├── ...
    │           ├── n01443537
    ```
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        tripletize: bool = False,
        num_triplets: int = 500_000,
        transform: torchvision.transforms.Compose | None = None,
        **kwargs,
    ):
        self.transform = transform
        self.tripletize = tripletize
        self.syn_to_class, self.val_to_syn = self._load_class_mappings(root)
        split = "train" if train else "val"
        self.samples, self.targets = self._find_samples(root, split)

        self.rng = seed_generator(int(1e12))
        np.random.seed(0)

        self.triplets = np.random.randint(0, len(self.samples), (num_triplets, 3))

        if self.tripletize:
            self.num_triplets = num_triplets

    def _load_class_mappings(self, root: str) -> tuple[dict, dict]:
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            syn_to_class = {v[0]: int(class_id) for class_id, v in json.load(f).items()}

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            val_to_syn = json.load(f)

        return syn_to_class, val_to_syn

    def _find_samples(self, root: str, split: str) -> tuple[list, list]:
        samples_dir = os.path.join(root, "Data/CLS-LOC", split, "**", "*.JPEG")
        files = sorted(glob.glob(samples_dir, recursive=True))

        samples = []
        targets = []

        for entry in files:
            syn_id = Path(entry).parent.name
            target = (
                self.syn_to_class.get(syn_id, None)
                if split == "train"
                else self.syn_to_class.get(
                    self.val_to_syn.get(Path(entry).name, None), None
                )
            )
            if target is not None:
                samples.append(entry)
                targets.append(target)

        return samples, targets

    def _load_transform_image(self, path: str) -> torch.Tensor | Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return self.num_triplets if self.tripletize else len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor | Image.Image, int]
        | tuple[tuple[torch.Tensor | Image.Image], tuple[int]]
    ):
        if self.tripletize:

            anchor, positive, negative = self.triplets[idx]

            img_anchor = self._load_transform_image(self.samples[anchor])
            img_positive = self._load_transform_image(self.samples[positive])
            img_negative = self._load_transform_image(self.samples[negative])

            targets = [
                self.targets[anchor],
                self.targets[positive],
                self.targets[negative],
            ]
            images = [img_anchor, img_positive, img_negative]
            return images, targets

        else:
            return self._load_transform_image(self.samples[idx]), self.targets[idx]


def filter_sun397(dataset: torch.utils.data.Dataset, split_file: str):
    """Filter given relative paths from text file
    TODO Check if we can not make this even more generic using a subset?"""
    split_files = np.loadtxt(split_file, dtype=str)
    split_files = set([Path(i).name for i in split_files])
    filenames = set([str(Path(f).name) for f in dataset._image_files])
    indices = [i for i, f in enumerate(filenames) if f in split_files]
    dataset._image_files = [dataset._image_files[i] for i in indices]
    dataset._labels = [dataset._labels[i] for i in indices]
    return dataset


def seed_generator(seed_range):
    for seed in range(seed_range):
        yield seed
