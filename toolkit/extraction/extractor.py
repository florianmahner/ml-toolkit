import torch
import torchvision

import numpy as np

from tqdm import tqdm
from ._hooks import HookModel
from ._ssl import load_ssl_model
from ._torchvision import load_torchvision_model
from ._open_clip import load_open_clip_model
from ._timm import load_timm_model
from ..data.datasets import ImageDataset

from torch.utils.data import DataLoader


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
    if source == "open_clip":
        return load_open_clip_model(model_name, weights)
    if source == "torchvision":
        return load_torchvision_model(model_name, weights)
    elif source == "timm":
        return load_timm_model(model_name, weights)
    elif source == "ssl":
        return load_ssl_model(model_name)
    else:
        raise ValueError(f"Source '{source}' is not recognized.")


def build_feature_extractor(
    model_name: str,
    module_name: list[str] | str,
    weights: str | None = "DEFAULT",
    source: str = "torchvision",
) -> tuple[HookModel, torchvision.transforms.Compose]:

    if source not in SOURCES:
        raise ValueError(
            f"Source '{source}' is not recognized. Available sources: {SOURCES}."
        )

    def transform_clip(x):
        return x / x.norm(dim=-1, keepdim=True)

    model, image_transform = load_model(model_name, weights, source)

    if source == "open_clip":
        feature_transform = transform_clip
    else:
        feature_transform = lambda x: x

    hook_model = HookModel(model, feature_transform=feature_transform)
    hook_model.register_hook(module_name)

    return hook_model, image_transform


@torch.no_grad()
def _extract_from_images_using_hook(model: HookModel, images: torch.Tensor):
    logits, features = model(images)
    return logits, features


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
    # NOTE CLS_TOKEN and AVG_POOL of feature model is probably only applicable to DINO and MAE models.
    extractor, tsfm = build_feature_extractor(model_name, module_names, weights, source)
    extractor.eval()
    extractor.to(device)
    dataset = ImageDataset(image_paths, tsfm)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Determine the total number of samples
    num_samples = len(dataset)
    batch_size = dataloader.batch_size

    sample_images = next(iter(dataloader)).to(device)
    sample_features = _extract_from_images_using_hook(extractor, sample_images)[1]
    feature_shape = sample_features.shape[1:]

    features = np.empty((num_samples, np.prod(feature_shape)), dtype=np.float32)

    start_idx = 0
    for images in tqdm(dataloader):
        images = images.to(device)
        batch_logits, batch_features = _extract_from_images_using_hook(
            extractor, images
        )

        batch_features = batch_features.squeeze().cpu().to(torch.float32).numpy()
        batch_logits = batch_logits.squeeze().cpu().to(torch.float32).numpy()

        if batch_features.ndim == 3 and token == "cls_token":
            batch_features = batch_features[:, 0, :]

        if average_spatial and batch_features.ndim == 4:
            batch_features = batch_features.mean(axis=(2, 3))

        if flatten_features:
            batch_features = batch_features.reshape(batch_features.shape[0], -1)

        end_idx = min(start_idx + batch_size, num_samples)
        features[start_idx:end_idx] = batch_features
        start_idx = end_idx

        del images, batch_logits, batch_features

    return features
