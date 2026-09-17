import os

import torch


def _maybe_unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _strip_module_prefix(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _add_module_prefix(state_dict):
    return {
        (key if key.startswith("module.") else f"module.{key}"): value
        for key, value in state_dict.items()
    }


def load_state_dict_flexible(model, checkpoint_path, map_location="cpu"):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _maybe_unwrap_state_dict(state)
    if not isinstance(state_dict, dict):
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")

    candidates = [
        state_dict,
        _strip_module_prefix(state_dict),
        _add_module_prefix(state_dict),
    ]

    errors = []
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError(
        "failed to load checkpoint with flexible matching:\n"
        + "\n---\n".join(errors)
    )
