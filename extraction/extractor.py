import torch
import torchvision

import numpy as np

from tqdm import tqdm
from ._hooks import HookModel
from ._ssl import SSLModelLoader
from ._torchvision import TorchvisionModelLoader
from ._open_clip import OpenCLIPLoader
from ._timm import TimmModelLoader
from ..data.datasets import ImageDataset

from torch.utils.data import DataLoader
from typing import Callable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Module = torch.nn.Module
Tensor = torch.Tensor


SOURCES = ("open_clip", "torchvision", "timm", "ssl")


def load_model(
    model_name: str,
    weights: str | None = "DEFAULT",
    source: str = "torchvision",
):
    if weights is None:
        print(
            "No weights specified, attempting to load model without pretrained weights."
        )
    loaders = {
        "open_clip": OpenCLIPLoader.load,
        "torchvision": TorchvisionModelLoader.load,
        "timm": TimmModelLoader.load,
        "ssl": SSLModelLoader.load,
    }

    # Validate the source and load the model
    if source in loaders:
        return loaders[source](model_name, weights)
    else:
        raise ValueError(f"Source '{source}' is not recognized.")


def build_feature_extractor(
    model_name: str,
    module_name: list[str] | str,
    weights: str | None = "DEFAULT",
    source: str = "torchvision",
    feature_transform: Callable[[Tensor], Tensor] | None = None,
) -> tuple[HookModel, torchvision.transforms.Compose]:

    if source not in SOURCES:
        raise ValueError(
            f"Source '{source}' is not recognized. Available sources: {SOURCES}."
        )

    model, image_transform = load_model(model_name, weights, source)
    feature_transform = feature_transform or (lambda x: x)

    hook_model = HookModel(model, feature_transform=feature_transform)
    hook_model.register_hook(module_name)

    return hook_model, image_transform


@torch.no_grad()
def _extract_from_images_using_hook(model: HookModel, images: torch.Tensor):
    logits, features = model(images)
    return logits, features


def process_batch_features(features, token, average_spatial, flatten_features):
    if token == "cls_token":
        features = features[:, 0, :]
    if average_spatial and features.ndim == 4:
        features = features.mean(dim=(-2, -1))
    if flatten_features:
        features = features.flatten(start_dim=1)
    return features


@torch.no_grad()
def extract_features_from_model(
    image_paths: list[str],
    model_name: str,
    module_names: list[str] | str | None = None,
    weights: str | None = "DEFAULT",
    source: str = "torchvision",
    flatten_features: bool = True,
    token: str | None = None,
    average_spatial: bool = False,
) -> np.ndarray:
    extractor, tsfm = build_feature_extractor(model_name, module_names, weights, source)
    extractor.eval()
    extractor.to(device)
    dataset = ImageDataset(image_paths, tsfm)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Initialize variables for feature storage
    num_samples = len(dataset)
    feature_shape = None
    features = None

    start_idx = 0
    for images in tqdm(dataloader):
        images = images.to(device)
        _, batch_features = _extract_from_images_using_hook(extractor, images)

        processed_features = process_batch_features(
            batch_features, token, average_spatial, flatten_features
        )

        # Determine feature shape and initialize the features array on the first batch
        if features is None:
            feature_shape = processed_features.shape[1:]
            if token == "cls_token":
                feature_shape = feature_shape[1:]
            features = np.empty((num_samples, np.prod(feature_shape)), dtype=np.float32)

        end_idx = start_idx + len(processed_features)
        features[start_idx:end_idx] = processed_features.cpu().numpy()
        start_idx = end_idx

        del images, batch_features

    return features
