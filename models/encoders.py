import os

import torch
import torch.nn as nn
from torchvision import models


def _repo_root():
    return os.path.dirname(os.path.dirname(__file__))


def _first_existing_file(paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return ""


def _first_hf_model_dir(paths):
    for path in paths:
        if path and os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
            return path
    return ""


def _hf_snapshot_dir(model_id):
    repo_name = f"models--{model_id.replace('/', '--')}"
    cache_roots = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        env_path = os.environ.get(env_name, "").strip()
        if not env_path:
            continue
        if env_name == "HF_HOME":
            cache_roots.append(os.path.join(env_path, "hub"))
        else:
            cache_roots.append(env_path)

    seen = set()
    for cache_root in cache_roots:
        cache_root = os.path.abspath(cache_root)
        if cache_root in seen or not os.path.isdir(cache_root):
            continue
        seen.add(cache_root)

        snapshot_root = os.path.join(cache_root, repo_name, "snapshots")
        if not os.path.isdir(snapshot_root):
            continue

        for snapshot_name in sorted(os.listdir(snapshot_root), reverse=True):
            snapshot_dir = os.path.join(snapshot_root, snapshot_name)
            if os.path.isfile(os.path.join(snapshot_dir, "config.json")):
                return snapshot_dir
    return ""


def _resolve_medclip_vision_model_dir():
    candidates = [
        os.environ.get("MEDCLIP_VISION_MODEL_DIR", "").strip(),
        os.environ.get("MEDCLIP_VIT_DIR", "").strip(),
        os.path.join(_repo_root(), "pretrained", "medclip-vit"),
        _hf_snapshot_dir("microsoft/swin-tiny-patch4-window7-224"),
    ]
    return _first_hf_model_dir(candidates)


def _resolve_medclip_vision_config_dir():
    candidates = [
        os.environ.get("MEDCLIP_VISION_MODEL_DIR", "").strip(),
        os.environ.get("MEDCLIP_VIT_DIR", "").strip(),
        os.path.join(_repo_root(), "pretrained", "medclip-vit"),
        _hf_snapshot_dir("microsoft/swin-tiny-patch4-window7-224"),
    ]
    return _first_hf_model_dir(candidates)


def _resolve_medclip_weights_path():
    candidates = [
        os.environ.get("MEDCLIP_VIT_WEIGHTS_PATH", "").strip(),
        os.environ.get("MEDCLIP_WEIGHTS_PATH", "").strip(),
        os.path.join(_repo_root(), "pretrained", "medclip-vit", "pytorch_model.bin"),
    ]
    return _first_existing_file(candidates)


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict):
                return nested
    if not isinstance(state, dict):
        raise TypeError("Unsupported MedCLIP checkpoint format.")
    return state


def _load_medclip_vision_weights(vision_model, ckpt_path):
    state = _extract_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    model_state = vision_model.state_dict()

    vision_state = {}
    for key, value in state.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key.startswith("vision_model."):
            vision_state[normalized_key[len("vision_model."):]] = value

    if not vision_state:
        raise KeyError(f"No vision_model.* weights found in MedCLIP checkpoint: {ckpt_path}")

    matched_keys = [
        key for key, value in vision_state.items() if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    ]
    if len(matched_keys) < 10:
        raise ValueError(
            f"MedCLIP checkpoint does not match the ViT backbone: {ckpt_path}. "
            "Check that MEDCLIP_WEIGHTS_PATH points to the medclip-vit weights."
        )

    vision_model.load_state_dict(vision_state, strict=False)


class LocalMedCLIPVisionModelViT(nn.Module):
    def __init__(self, config_dir):
        super().__init__()
        try:
            from transformers import SwinConfig, SwinModel
        except Exception as e:
            raise ImportError(
                "transformers is required to build the local MedCLIP ViT fallback."
            ) from e

        if not config_dir or not os.path.isfile(os.path.join(config_dir, "config.json")):
            raise FileNotFoundError(
                "Missing local Swin config.json for MedCLIP ViT fallback. "
                "Populate pretrained/medclip-vit/config.json or set MEDCLIP_VISION_MODEL_DIR."
            )

        config = SwinConfig.from_pretrained(config_dir, local_files_only=True)
        self.model = SwinModel(config)
        self.projection_head = nn.Linear(int(config.hidden_size), 512, bias=False)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state.mean(dim=1)
        return self.projection_head(pooled)


class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=False):
        super().__init__()
        self.name = name
        self.use_medclip = False

        if name == "resnet18":
            if pretrained:
                try:
                    weights = models.ResNet18_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"
            else:
                weights = None
            net = models.resnet18(weights=weights)
            feat_dim = 512

            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.pool = net.avgpool

        elif name == "resnet50":
            if pretrained:
                try:
                    weights = models.ResNet50_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"
            else:
                weights = None
            net = models.resnet50(weights=weights)
            feat_dim = 2048

            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.pool = net.avgpool

        elif name == "resnet34":
            if pretrained:
                try:
                    weights = models.ResNet34_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"
            else:
                weights = None
            net = models.resnet34(weights=weights)
            feat_dim = 512

            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.pool = net.avgpool

        elif name == "medclip_vit":
            vision_checkpoint = _resolve_medclip_vision_model_dir()
            ckpt_path = _resolve_medclip_weights_path()
            vision_model = None
            medclip_error = None

            if vision_checkpoint:
                try:
                    from medclip import MedCLIPVisionModelViT

                    vision_model = MedCLIPVisionModelViT(checkpoint=vision_checkpoint)
                except Exception as e:
                    medclip_error = e

            if vision_model is None:
                config_dir = _resolve_medclip_vision_config_dir()
                try:
                    vision_model = LocalMedCLIPVisionModelViT(config_dir=config_dir)
                except Exception as e:
                    raise OSError(
                        "Failed to initialize MedCLIP ViT offline. Provide a local Hugging Face Swin directory via "
                        "MEDCLIP_VISION_MODEL_DIR (or MEDCLIP_VIT_DIR), or ensure pretrained/medclip-vit contains "
                        "config.json plus the MedCLIP vision checkpoint."
                    ) from (medclip_error or e)

            self.vision_model = vision_model

            if pretrained:
                if not ckpt_path:
                    raise FileNotFoundError(
                        "Could not find MedCLIP weights. Set MEDCLIP_WEIGHTS_PATH or place the checkpoint at "
                        "./pretrained/medclip-vit/pytorch_model.bin"
                    )
                _load_medclip_vision_weights(self.vision_model, ckpt_path)

            self.use_medclip = True
            feat_dim = 512

        else:
            raise ValueError("backbone must be resnet18, resnet34, resnet50, or medclip_vit")

        self.feat_dim = feat_dim

    def forward(self, x):
        if self.use_medclip:
            out = self.vision_model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x
