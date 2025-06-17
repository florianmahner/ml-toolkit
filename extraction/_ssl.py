import torch
import os
import torch.nn as nn
import torchvision.models as models
from ._transforms import imagenet_transforms
from ._base import BaseModelLoader

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


MODELS = {
    "simclr-rn50": {
        "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch",
        "arch": "resnet50",
        "type": "vissl",
    },
    "mocov2-rn50": {
        "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch",
        "arch": "resnet50",
        "type": "vissl",
    },
    "jigsaw-rn50": {
        "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.db174a43/model_final_checkpoint_phase104.torch",
        "arch": "resnet50",
        "type": "vissl",
    },
    "rotnet-rn50": {
        "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch",
        "arch": "resnet50",
        "type": "vissl",
    },
    "swav-rn50": {
        "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/model_final_checkpoint_phase799.torch",
        "arch": "resnet50",
        "type": "vissl",
    },
    "pirl-rn50": {
        "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch",
        "arch": "resnet50",
        "type": "vissl",
    },
    "barlowtwins-rn50": {
        "arch": "resnet50",
        "type": "checkpoint_url",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/barlowtwins/ljng/resnet50.pth",
    },
}


def download_model(model_name, save_dir: str):
    model_info = MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found in MODELS dictionary.")

    os.makedirs(save_dir, exist_ok=True)

    if model_info["type"] in ["vissl", "checkpoint_url"]:
        url = model_info.get("url") or model_info.get("checkpoint_url")
        state_dict = load_state_dict_from_url(url, model_dir=save_dir)

        if model_info["type"] == "vissl" and "classy_state_dict" in state_dict:
            state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        torch.save(state_dict, os.path.join(save_dir, f"{model_name}.pth"))


def clean_state_dict(state_dict):
    if "classy_state_dict" in state_dict:
        state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
    return {k.replace("_feature_blocks.", ""): v for k, v in state_dict.items()}


class SSLModelLoader(BaseModelLoader):
    @staticmethod
    def load(
        model_name, weights: str | None = "DEFAULT", save_dir="./cache", fresh=True
    ):
        model_info = MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found in MODELS dictionary.")

        cache_dir = os.path.join(torch.hub.get_dir(), "vissl")
        model_path = os.path.join(cache_dir, f"{model_name}.pth")
        if not os.path.exists(model_path) or fresh:
            download_model(model_name, save_dir)

        if model_info["type"] in ["vissl", "checkpoint_url"]:
            model_filepath = os.path.join(cache_dir, model_name + ".pth")
            if not os.path.exists(model_filepath):
                os.makedirs(cache_dir, exist_ok=True)
                url = model_info.get("url") or model_info.get("checkpoint_url")
                state_dict = load_state_dict_from_url(url, model_dir=cache_dir)
                state_dict = clean_state_dict(state_dict)
                torch.save(state_dict, model_filepath)
            else:
                state_dict = torch.load(
                    model_filepath, map_location=torch.device("cpu")
                )
                state_dict = clean_state_dict(state_dict)

            model = getattr(models, model_info["arch"])(weights=weights)

            if model_info["arch"] == "resnet50":
                model.fc = nn.Identity()
            model.load_state_dict(state_dict, strict=True)
        else:
            raise ValueError(
                f"Unknown model type {model_info['type']} for model {model_name}."
            )

        return model, imagenet_transforms()
