"""PSNR/SSIM metrics based on pyiqa.

Inputs are RGB tensors in [0, 1] with shape [B, C, H, W].
Both RGB and Y-channel (YCbCr) PSNR/SSIM are calculated.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pyiqa
import torch


_METRIC_CACHE = {}


def _get_metric_models(device: torch.device, border: int):
    """Create and cache pyiqa metric models for the current device and border."""
    if border < 0:
        raise ValueError(f"border must be non-negative, got {border}.")

    key: Tuple[str, int] = (str(device), int(border))

    if key not in _METRIC_CACHE:
        _METRIC_CACHE[key] = {
            "psnr_rgb": pyiqa.create_metric(
                "psnr",
                test_y_channel=False,
                color_space="rgb",
                crop_border=border,
                device=device,
            ),
            "ssim_rgb": pyiqa.create_metric(
                "ssim",
                test_y_channel=False,
                color_space="rgb",
                crop_border=border,
                device=device,
            ),
            "psnr_y": pyiqa.create_metric(
                "psnr",
                test_y_channel=True,
                color_space="ycbcr",
                crop_border=border,
                device=device,
            ),
            "ssim_y": pyiqa.create_metric(
                "ssim",
                test_y_channel=True,
                color_space="ycbcr",
                crop_border=border,
                device=device,
            ),
        }

        for metric in _METRIC_CACHE[key].values():
            metric.eval()

    return _METRIC_CACHE[key]


@torch.no_grad()
def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    border: int,
) -> Dict[str, float]:
    """Return RGB and Y-channel PSNR/SSIM using pyiqa."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Metric input shapes differ: {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )

    if prediction.ndim != 4:
        raise ValueError(
            f"Expected [B, C, H, W], got {tuple(prediction.shape)}."
        )

    if prediction.shape[1] not in (1, 3):
        raise ValueError(
            f"Expected 1 or 3 channels, got {prediction.shape[1]}."
        )

    height, width = prediction.shape[-2:]
    if border > 0 and (height <= 2 * border or width <= 2 * border):
        raise ValueError(
            f"Cannot crop a {border}-pixel border from image size "
            f"{height}x{width}."
        )

    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)

    metric_models = _get_metric_models(prediction.device, border)

    psnr_rgb = metric_models["psnr_rgb"](prediction, target)
    ssim_rgb = metric_models["ssim_rgb"](prediction, target)

    if prediction.shape[1] == 3:
        psnr_y = metric_models["psnr_y"](prediction, target)
        ssim_y = metric_models["ssim_y"](prediction, target)
    else:
        psnr_y = psnr_rgb
        ssim_y = ssim_rgb

    return {
        "psnr_rgb": float(psnr_rgb.mean().item()),
        "ssim_rgb": float(ssim_rgb.mean().item()),
        "psnr_y": float(psnr_y.mean().item()),
        "ssim_y": float(ssim_y.mean().item()),
    }