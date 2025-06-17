import timm
from ._base import BaseModelLoader


class TimmModelLoader(BaseModelLoader):
    @staticmethod
    def load(model_name: str, weights: str | None = "DEFAULT"):
        pretrained = weights != None
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        tsfms = timm.data.create_transform(**data_config, is_training=False)

        return model, tsfms
