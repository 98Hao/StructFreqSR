#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def charbonnier_loss(img1, img2, epsilon=1e-3):
    return torch.sqrt((img1 - img2).square() + epsilon ** 2).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def temporal_consistency_loss(pred, target):
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    return F.mse_loss(pred_diff, target_diff)

def temporal_consistency_loss2(pred, target, scales=(1, 3, 5, 7)):
    B, T, C, H, W = pred.shape

    if T < 2:
        return pred.sum() * 0.0

    pred_pre = pred[:, :-1].reshape(-1, C, H, W)
    pred_cur = pred[:, 1:].reshape(-1, C, H, W)
    gt_pre = target[:, :-1].reshape(-1, C, H, W)
    gt_cur = target[:, 1:].reshape(-1, C, H, W)

    temporal_mags, temporal_losses = [], []

    for k in scales:
        if k == 1:
            pred_diff = pred_cur - pred_pre
            gt_diff = gt_cur - gt_pre
        else:
            pred_diff = F.avg_pool2d(pred_cur, k, 1, k // 2, count_include_pad=False) - F.avg_pool2d(pred_pre, k, 1, k // 2, count_include_pad=False)
            gt_diff = F.avg_pool2d(gt_cur, k, 1, k // 2, count_include_pad=False) - F.avg_pool2d(gt_pre, k, 1, k // 2, count_include_pad=False)

        temporal_mags.append(pred_diff.abs())
        temporal_losses.append((pred_diff - gt_diff).abs())

    temporal_mags = torch.stack(temporal_mags, dim=0)
    temporal_losses = torch.stack(temporal_losses, dim=0)

    best_idx = temporal_mags.argmin(dim=0, keepdim=True)
    loss = temporal_losses.gather(0, best_idx).mean()

    return loss


import math
def dct_1d(x, dim=-1):
    N = x.shape[dim]
    even_idx = torch.arange(0, N, 2, device=x.device)
    odd_idx = torch.arange(1, N, 2, device=x.device).flip(0)
    x = torch.cat([x.index_select(dim, even_idx), x.index_select(dim, odd_idx)], dim=dim)
    X = torch.fft.fft(x, dim=dim)

    k = torch.arange(N, device=x.device, dtype=x.dtype)
    shape = [1] * x.ndim
    shape[dim] = N
    phase = torch.exp((-1j * math.pi / (2 * N)) * k).reshape(shape)

    y = 2.0 * (X * phase).real

    scale = torch.full((N,), math.sqrt(2.0 / N) / 2.0, device=x.device, dtype=x.dtype)
    scale[0] = math.sqrt(1.0 / N) / 2.0

    return y * scale.reshape(shape)
def dct_2d(x):
    return dct_1d(dct_1d(x, dim=-1), dim=-2)
def dct_frequency_loss(pred, target):
    B, T, C, H, W = pred.shape

    dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype

    pred = pred.to(dtype).reshape(-1, H, W)
    target = target.to(dtype).reshape(-1, H, W)

    pred_dct = dct_2d(pred)
    target_dct = dct_2d(target)

    return charbonnier_loss(pred_dct, target_dct)

def dct_frequency_loss_weight(pred, target, band_weights=(1.0, 1.0, 2.0), low_ratio=0.25, mid_ratio=0.60):
    B, T, C, H, W = pred.shape

    dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
    pred = pred.to(dtype).reshape(-1, H, W)
    target = target.to(dtype).reshape(-1, H, W)

    pred_dct = dct_2d(pred)
    target_dct = dct_2d(target)
    diff = (pred_dct - target_dct).abs()

    fy = torch.linspace(0, 1, H, device=pred.device).view(H, 1)
    fx = torch.linspace(0, 1, W, device=pred.device).view(1, W)
    rho = torch.sqrt((fy.square() + fx.square()) / 2.0)

    masks = [
        rho < low_ratio,
        (rho >= low_ratio) & (rho < mid_ratio),
        rho >= mid_ratio
    ]

    losses, weights = [], []

    for mask, weight in zip(masks, band_weights):
        if mask.any():
            losses.append(diff[:, mask].mean() * weight)
            weights.append(weight)

    return torch.stack(losses).sum() / sum(weights)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
def ssim_loss(img1, img2, window_size=11, size_average=True):
    B, T, C, H, W = img1.shape

    img1 = img1.reshape(B * T, C, H, W)
    img2 = img2.reshape(B * T, C, H, W)

    return 1 - ssim(img1, img2, window_size, size_average)





