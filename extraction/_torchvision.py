from torchvision.models import get_model, get_model_weights
from ._base import BaseModelLoader


class TorchvisionModelLoader(BaseModelLoader):
    @staticmethod
    def load(model_name: str, weights: str | None = "DEFAULT"):
        model = get_model(model_name, weights=weights)
        weight_enum = get_model_weights(model_name)
        tsfms = getattr(weight_enum, weights, None).transforms()

        return model, tsfms
