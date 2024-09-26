import open_clip
import torchvision
from torch.nn import Module


def load_open_clip_model(
    variant: str | None = "RN50",
    weights: str | None = None,
) -> tuple[Module, torchvision.transforms.Compose]:

    available_models = open_clip.list_pretrained()
    available_models, _ = zip(*available_models)
    if variant not in available_models:
        raise ValueError(
            f"Model '{variant}' is not recognized in OpenCLIP. Available models: {set(available_models)}."
        )
    model, _, tsfms = open_clip.create_model_and_transforms(variant, pretrained=weights)
    return model, tsfms
