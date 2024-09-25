import timm


def load_timm_model(model_name: str, weights: str):
    pretrained = True if weights else False
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)
    tsfms = timm.data.create_transform(**data_config, is_training=False)

    return model, tsfms
