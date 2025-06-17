import open_clip
import torchvision
from torch.nn import Module
from ._base import BaseModelLoader


class OpenCLIPLoader(BaseModelLoader):
    @staticmethod
    def load(
        model_name: str, weights: str | None = "DEFAULT"
    ) -> tuple[Module, torchvision.transforms.Compose]:
        if weights == "DEFAULT":
            weights = None  # OpenCLIP uses None for default weights

        available_models = open_clip.list_pretrained()
        available_models, _ = zip(*available_models)
        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' is not recognized in OpenCLIP. Available models: {set(available_models)}."
            )
        model, _, tsfms = open_clip.create_model_and_transforms(
            model_name, pretrained=weights
        )

        return model, tsfms
