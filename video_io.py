"""Small IO helpers shared by causal VSR training and testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Only batch size 1 can be converted to a PIL image.")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
        raise ValueError(f"Expected [1/3, H, W], got {tuple(tensor.shape)}.")

    array = torch.round(tensor * 255.0).byte().permute(1, 2, 0).numpy()
    if tensor.shape[0] == 1:
        array = array[..., 0]
        return Image.fromarray(array, mode="L")
    return Image.fromarray(array, mode="RGB")


def pil_to_tensor(path: str | Path, channels: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode), dtype=np.float32).copy()
    if channels == 1:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(255.0)


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Accept checkpoints saved with or without DataParallel."""

    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(checkpoint)}.")
    return checkpoint
