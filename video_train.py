"""Causal video-SR training with ordinary variables configured in __main__."""

from __future__ import annotations

import csv
import gc
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    MultiStepLR,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from openpyxl import Workbook
from openpyxl.styles import Font

from video_datasets import (
    CausalVideoClipDataset,
    VideoScene,
    discover_video_scenes,
    load_paired_frame,
    seed_worker,
)
from video_io import clean_state_dict, load_checkpoint, resolve_device
from video_metrics import calculate_metrics
from metrics.loss_utils import (
    charbonnier_loss, temporal_consistency_loss, dct_frequency_loss, dct_frequency_loss_weight,
    temporal_consistency_loss2
)

METRIC_NAMES = ("psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y")
BEST_METRIC_NAMES = ("psnr_rgb", "ssim_rgb")


def seed_everything(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def clear_training_memory() -> None:

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def normalized_scheduler_name(name: str) -> str:

    normalized = name.strip().lower()
    if normalized == "warmuo_cosine":
        normalized = "warmup_cosine"
    return normalized


def create_scheduler(optimizer: torch.optim.Optimizer):

    scheduler_name = normalized_scheduler_name(scheduler_type)

    if scheduler_name == "multistep":
        if not multistep_milestones:
            raise ValueError("multistep_milestones must not be empty.")
        scheduler = MultiStepLR(
            optimizer,
            milestones=multistep_milestones,
            gamma=multistep_gamma,
        )
        print(
            "LR Scheduler: MultiStepLR | "
            f"milestones={multistep_milestones} | gamma={multistep_gamma}"
        )

    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epoch_max,
            eta_min=min_learning_rate,
        )
        print(
            "LR Scheduler: CosineAnnealingLR | "
            f"T_max={epoch_max} | eta_min={min_learning_rate:.2e}"
        )

    elif scheduler_name == "warmup_cosine":
        if not 0 < warmup_epochs < epoch_max:
            raise ValueError(
                "warmup_epochs must satisfy 0 < warmup_epochs < epoch_max, "
                f"got {warmup_epochs} and {epoch_max}."
            )
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError(
                "warmup_start_factor must be in (0, 1], got "
                f"{warmup_start_factor}."
            )
        cosine_epochs = epoch_max - warmup_epochs

        def make_lr_lambda(base_lr: float):
            minimum_ratio = min_learning_rate / base_lr

            def lr_multiplier(scheduler_epoch: int) -> float:
                if scheduler_epoch < warmup_epochs:
                    warmup_progress = scheduler_epoch / warmup_epochs
                    return warmup_start_factor + (
                        1.0 - warmup_start_factor
                    ) * warmup_progress
                cosine_progress = min(
                    (scheduler_epoch - warmup_epochs) / cosine_epochs, 1.0
                )
                return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * cosine_progress)
                )

            return lr_multiplier

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                make_lr_lambda(group["lr"])
                for group in optimizer.param_groups
            ],
        )
        print(
            "LR Scheduler: Warmup + CosineAnnealingLR | "
            f"warmup_epochs={warmup_epochs} | "
            f"start_factor={warmup_start_factor} | "
            f"cosine_epochs={cosine_epochs} | "
            f"eta_min={min_learning_rate:.2e}"
        )

    else:
        raise ValueError(
            f"Unknown scheduler_type={scheduler_type!r}. Choose "
            "'multistep', 'cosine', or 'warmup_cosine'."
        )
    return scheduler


def current_model_config() -> Dict[str, object]:
    return {
        "scale": scale,
        "channels": channels,
        "num_features": num_features,
        "semantic_channels": semantic_channels,
        "dct_block": dct_block,
        "unet_levels": unet_levels,
        "unet_base_channels": unet_base_channels,
        "unet_max_channels": unet_max_channels,
        "conv_type": conv_type,
        "memory_groups": memory_groups,
        "memory_topk": memory_topk,
        "max_offset": max_offset,
        "num_recon_blocks": num_recon_blocks,
        "use_aligned_prev": use_aligned_prev,
    }


def current_training_settings() -> Dict[str, object]:

    return {
        "flag": flag,
        "data_backend": data_backend.strip().lower(),
        "data_root": str(data_root),
        "h5_data_root": str(h5_data_root),
        "train_split": train_split,
        "test_split": test_split,
        "hr_folder_name": hr_folder_name,
        "lr_folder_name": lr_folder_name,
        "sequence_length": sequence_length,
        "frame_interval": frame_interval,
        "clip_stride": clip_stride,
        "patch_size": patch_size,
        "batch_size": batch_size,
        "workers": workers,
        "epoch_max": epoch_max,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "loss_skip_frames": loss_skip_frames,
        "scheduler_type": normalized_scheduler_name(scheduler_type),
        "multistep_milestones": list(multistep_milestones),
        "multistep_gamma": multistep_gamma,
        "min_learning_rate": min_learning_rate,
        "warmup_epochs": warmup_epochs,
        "warmup_start_factor": warmup_start_factor,
        "device_name": device_name,
        "use_amp": use_amp,
        "seed": seed,
        "save_every": save_every,
        "test_crop_border": test_crop_border,
        "skip_test_start_frames": skip_test_start_frames,
        "test_scene_names": (
            None if test_scene_names is None else list(test_scene_names)
        ),
        "best_metric_names": list(best_metric_names),
    }


def validate_settings() -> None:
    if data_backend.strip().lower() not in ("images", "h5"):
        raise ValueError("data_backend must be 'images' or 'h5'.")
    if save_every < 0:
        raise ValueError("save_every must be non-negative; use 0 to disable it.")
    if epoch_start < 1 or epoch_max < epoch_start:
        raise ValueError("Require 1 <= epoch_start <= epoch_max.")
    if min(batch_size, sequence_length, patch_size, workers + 1) < 1:
        raise ValueError(
            "batch_size, sequence_length, and patch_size must be positive; "
            "workers must be non-negative."
        )
    if learning_rate <= 0 or not 0 <= min_learning_rate <= learning_rate:
        raise ValueError(
            "Require learning_rate > 0 and 0 <= min_learning_rate <= "
            f"learning_rate, got {learning_rate} and {min_learning_rate}."
        )
    if not 0 <= loss_skip_frames < sequence_length:
        raise ValueError(
            "loss_skip_frames must be in [0, sequence_length), got "
            f"{loss_skip_frames}."
        )
    if skip_test_start_frames < 0:
        raise ValueError("skip_test_start_frames must be non-negative.")
    if test_scene_names is not None:
        if isinstance(test_scene_names, str) or not isinstance(
            test_scene_names, (list, tuple)
        ):
            raise ValueError(
                "test_scene_names must be None, a list, or a tuple. For "
                "example: ['Bar', 'Bistro']."
            )
        if not test_scene_names:
            raise ValueError(
                "test_scene_names cannot be empty. Use None to test all scenes."
            )
    if not isinstance(best_metric_names, (list, tuple)):
        raise ValueError(
            "best_metric_names must be a list or tuple."
        )
    if tuple(best_metric_names) != BEST_METRIC_NAMES:
        raise ValueError(
            "Only PSNR-RGB and SSIM-RGB best checkpoints are saved. Set "
            "best_metric_names exactly to ['psnr_rgb', 'ssim_rgb']."
        )


def select_test_scenes(
    scenes: List[VideoScene], requested_names
) -> List[VideoScene]:

    if requested_names is None:
        return scenes

    lookup = {scene.name.casefold(): scene for scene in scenes}
    normalized_names = [str(name).casefold() for name in requested_names]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(
            f"test_scene_names contains duplicates: {list(requested_names)}."
        )
    missing_names = [
        str(name)
        for name, normalized in zip(requested_names, normalized_names)
        if normalized not in lookup
    ]
    if missing_names:
        available_names = [scene.name for scene in scenes]
        raise ValueError(
            f"Unknown test scenes: {missing_names}. Available scenes are: "
            f"{available_names}."
        )
    return [lookup[name] for name in normalized_names]


def atomic_torch_save(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary_path)
    temporary_path.replace(path)


def migrate_legacy_best_checkpoints(experiment_dir: Path) -> None:

    legacy_alias = experiment_dir / "best.pth"
    if legacy_alias.is_file():
        legacy_alias.unlink()

    for metric_name in BEST_METRIC_NAMES:
        legacy_path = experiment_dir / f"best_{metric_name}.pth"
        if legacy_path.is_file():
            checkpoint = torch.load(legacy_path, map_location="cpu")
            checkpoint_epoch = int(checkpoint.get("epoch", 0))
            if checkpoint_epoch < 1:
                legacy_path.unlink()
            else:
                epoch_path = experiment_dir / (
                    f"best_{metric_name}_epoch_{checkpoint_epoch:04d}.pth"
                )
                if epoch_path.is_file():
                    legacy_path.unlink()
                else:
                    legacy_path.replace(epoch_path)
        saved_epoch = find_saved_best_epoch(experiment_dir, metric_name)
        if saved_epoch > 0:
            keep_path = experiment_dir / (
                f"best_{metric_name}_epoch_{saved_epoch:04d}.pth"
            )
            remove_older_best_checkpoints(
                experiment_dir,
                metric_name,
                keep_path,
            )


def remove_older_best_checkpoints(
    experiment_dir: Path,
    metric_name: str,
    keep_path: Path,
) -> None:

    candidates = list(
        experiment_dir.glob(f"best_{metric_name}_epoch_*.pth")
    )
    candidates.append(experiment_dir / f"best_{metric_name}.pth")
    for path in candidates:
        if path != keep_path and path.is_file():
            path.unlink()


def find_saved_best_epoch(experiment_dir: Path, metric_name: str) -> int:
    epochs = []
    prefix = f"best_{metric_name}_epoch_"
    for path in experiment_dir.glob(f"{prefix}*.pth"):
        epoch_text = path.stem[len(prefix) :]
        if epoch_text.isdigit():
            epochs.append(int(epoch_text))
    return max(epochs, default=0)


def save_history(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _safe_sheet_name(name: str) -> str:
    invalid = '[]:*?/\\'
    sheet_name = str(name)
    for char in invalid:
        sheet_name = sheet_name.replace(char, '_')
    sheet_name = sheet_name.strip() or 'Scene'
    return sheet_name[:31]


def save_scene_history_excel(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)

    scene_names = []
    for row in rows:
        scene_name = str(row["scene"])
        if scene_name not in scene_names:
            scene_names.append(scene_name)

    used_sheet_names = set()
    headers = [
        "epoch", "frames", "test_loss", "test_psnr_rgb", "test_ssim_rgb",
        "test_psnr_y", "test_ssim_y", "time_ms",
    ]

    for scene_name in scene_names:
        base_name = _safe_sheet_name(scene_name)
        sheet_name = base_name
        suffix = 1
        while sheet_name in used_sheet_names:
            suffix_text = f"_{suffix}"
            sheet_name = f"{base_name[:31-len(suffix_text)]}{suffix_text}"
            suffix += 1
        used_sheet_names.add(sheet_name)

        sheet = workbook.create_sheet(title=sheet_name)
        sheet.append(headers)
        for cell in sheet[1]:
            cell.font = Font(bold=True)

        for row in rows:
            if str(row["scene"]) == scene_name:
                sheet.append([row.get(header, None) for header in headers])

        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        widths = {"A": 10, "B": 10, "C": 14, "D": 16, "E": 16, "F": 16, "G": 16, "H": 14}
        for column, width in widths.items():
            sheet.column_dimensions[column].width = width

        for row_index in range(2, sheet.max_row + 1):
            sheet[f"C{row_index}"].number_format = "0.000000"
            sheet[f"D{row_index}"].number_format = "0.0000"
            sheet[f"E{row_index}"].number_format = "0.000000"
            sheet[f"F{row_index}"].number_format = "0.0000"
            sheet[f"G{row_index}"].number_format = "0.000000"
            sheet[f"H{row_index}"].number_format = "0.000"

    temporary_path = path.with_name(path.stem + ".tmp.xlsx")
    workbook.save(temporary_path)
    temporary_path.replace(path)


def build_checkpoint(
    model: RenderVSR,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch_index: int,
    global_step: int,
    best_metric_values: Dict[str, float],
    best_metric_epochs: Dict[str, int],
    history: List[Dict[str, object]],
    scene_history: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "epoch": epoch_index,
        "global_step": global_step,
        "model": model.state_dict(),
        "model_config": current_model_config(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler_type": normalized_scheduler_name(scheduler_type),
        "best_metric_names": list(best_metric_names),
        "best_metric_values": dict(best_metric_values),
        "best_metric_epochs": dict(best_metric_epochs),
        "training_settings": current_training_settings(),
        "history": history,
        "scene_history": scene_history,
    }


def restore_training_state(
    checkpoint_path: Path,
    model: RenderVSR,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[
    int,
    int,
    Dict[str, float],
    Dict[str, int],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    saved_scheduler = normalized_scheduler_name(
        str(checkpoint.get("scheduler_type", scheduler_type))
    )
    current_scheduler = normalized_scheduler_name(scheduler_type)
    if saved_scheduler != current_scheduler:
        raise ValueError(
            f"Checkpoint scheduler is {saved_scheduler!r}, but the current "
            f"scheduler_type is {current_scheduler!r}."
        )
    saved_model_config = checkpoint.get("model_config")
    if saved_model_config is not None and dict(saved_model_config) != current_model_config():
        raise ValueError(
            "Checkpoint model_config differs from current variables. "
            f"Saved={saved_model_config}, current={current_model_config()}."
        )

    saved_training_settings = checkpoint.get("training_settings") or {}
    saved_test_scene_names = saved_training_settings.get("test_scene_names")
    current_test_scene_names = (
        None if test_scene_names is None else list(test_scene_names)
    )
    if saved_test_scene_names != current_test_scene_names:
        raise ValueError(
            "Checkpoint test_scene_names differs from the current setting. "
            f"Saved={saved_test_scene_names}, current={current_test_scene_names}. "
            "These settings produce incomparable best metrics. Restore the old "
            "scene list, or start a new experiment flag."
        )

    model.load_state_dict(
        clean_state_dict(checkpoint.get("model", checkpoint)), strict=True
    )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    restored_global_step = int(checkpoint.get("global_step", 0))
    restored_history = list(checkpoint.get("history", []))
    restored_scene_history = list(checkpoint.get("scene_history", []))

    saved_metric_names = checkpoint.get("best_metric_names")
    if saved_metric_names is None:
        legacy_metric_name = str(
            checkpoint.get("best_metric_name", best_metric_names[0])
        )
        saved_metric_names = [legacy_metric_name]
    saved_metric_names = list(saved_metric_names)
    if saved_metric_names != list(best_metric_names):
        raise ValueError(
            "Checkpoint best-metric list differs from the current setting. "
            f"Saved={saved_metric_names}, current={list(best_metric_names)}. "
            "Keep the same list when resuming, or start a new experiment flag."
        )

    saved_best_values = checkpoint.get("best_metric_values")
    if saved_best_values is None:
        saved_best_values = {
            saved_metric_names[0]: float(
                checkpoint.get("best_metric_value", float("-inf"))
            )
        }
    restored_best_values = {
        metric_name: float(
            saved_best_values.get(metric_name, float("-inf"))
        )
        for metric_name in best_metric_names
    }
    saved_best_epochs = checkpoint.get("best_metric_epochs") or {}
    restored_best_epochs = {
        metric_name: int(
            saved_best_epochs.get(
                metric_name,
                find_saved_best_epoch(checkpoint_path.parent, metric_name),
            )
        )
        for metric_name in best_metric_names
    }
    print(
        f"Resumed: {checkpoint_path.resolve()} | next epoch={start_epoch} | "
        f"global step={restored_global_step} | "
        f"LR={optimizer.param_groups[0]['lr']:.6e} | "
        f"best={restored_best_values} | best epochs={restored_best_epochs}"
    )
    return (
        start_epoch,
        restored_global_step,
        restored_best_values,
        restored_best_epochs,
        restored_history,
        restored_scene_history,
    )


def train_one_epoch(
    model: RenderVSR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    epoch_index: int,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    sample_count = 0

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Train {epoch_index:04d}/{epoch_max:04d}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for batch in progress:
        lr_sequence = batch["lr"].to(device, non_blocking=True)
        hr_sequence = batch["hr"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            prediction = model(lr_sequence)

            loss = 0.0
            if weichar!= 0:
                loss = weichar * charbonnier_loss(prediction[:, loss_skip_frames:], hr_sequence[:, loss_skip_frames:])
            if weitem!= 0:
                loss = weitem * temporal_consistency_loss(prediction[:, loss_skip_frames:], hr_sequence[:, loss_skip_frames:])
            if weidct!= 0:
                loss = weidct * dct_frequency_loss(prediction[:, loss_skip_frames:], hr_sequence[:, loss_skip_frames:])
            if weidct2!= 0:
                loss = weidct2 * dct_frequency_loss_weight(prediction[:, loss_skip_frames:], hr_sequence[:, loss_skip_frames:])
            if weitem2!= 0:
                loss = weitem2 * temporal_consistency_loss2(prediction[:, loss_skip_frames:], hr_sequence[:, loss_skip_frames:])

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        current_batch_size = lr_sequence.shape[0]
        loss_sum += loss_value * current_batch_size
        sample_count += current_batch_size
        global_step += 1
        progress.set_postfix(
            loss=f"{loss_value:.5f}",
            avg=f"{loss_sum / sample_count:.5f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            refresh=False,
        )
    return loss_sum / sample_count, global_step


@torch.no_grad()
def test_complete_scenes(
    model: RenderVSR,
    scenes: List[VideoScene],
    device: torch.device,
    amp_enabled: bool,
    epoch_index: int,
) -> Dict[str, object]:

    model.eval()
    border = scale if test_crop_border is None else test_crop_border
    average_names = ("loss",) + METRIC_NAMES
    scene_metrics: List[Dict[str, object]] = []
    total_frames = sum(scene.num_frames for scene in scenes)
    progress = tqdm(
        total=total_frames,
        desc=f"Test  {epoch_index:04d}/{epoch_max:04d}",
        unit="frame",
        dynamic_ncols=True,
        leave=False,
    )

    try:
        for scene in scenes:
            scene_totals = {name: 0.0 for name in average_names}
            scene_measured_frames = 0
            scene_time_ms = 0.0
            scene_timed_frames = 0
            state = None
            progress.set_postfix(scene=scene.name, refresh=False)

            for frame_index in range(scene.num_frames):
                lr_image, hr_image = load_paired_frame(scene, frame_index, channels, scale)
                lr_image = lr_image.unsqueeze(0).to(device)
                hr_image = hr_image.unsqueeze(0).to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_start = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction, state = model.forward_step(lr_image, state)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - inference_start) * 1000.0

                if frame_index >= skip_test_start_frames:
                    scene_time_ms += elapsed_ms
                    scene_timed_frames += 1
                    frame_loss = charbonnier_loss(prediction, hr_image).item()
                    scene_totals["loss"] += frame_loss
                    metric_prediction = prediction.float().clamp(0.0, 1.0)
                    metrics = calculate_metrics(metric_prediction, hr_image.float(), border)
                    for metric_name in METRIC_NAMES:
                        scene_totals[metric_name] += metrics[metric_name]
                    scene_measured_frames += 1
                    progress.set_postfix(
                        scene=scene.name,
                        loss=f"{scene_totals['loss'] / scene_measured_frames:.5f}",
                        psnr=f"{scene_totals['psnr_rgb'] / scene_measured_frames:.2f}",
                        ssim=f"{scene_totals['ssim_rgb'] / scene_measured_frames:.4f}",
                        time=f"{scene_time_ms / scene_timed_frames:.2f}ms",
                        refresh=False,
                    )
                progress.update(1)

            if scene_measured_frames == 0:
                raise ValueError(
                    f"Scene {scene.name!r} has no test frames after "
                    f"skip_test_start_frames={skip_test_start_frames}."
                )

            scene_result: Dict[str, object] = {
                "scene": scene.name,
                "frames": scene_measured_frames,
                "time_ms": scene_time_ms / scene_timed_frames,
            }
            for name in average_names:
                scene_result[name] = scene_totals[name] / scene_measured_frames
            scene_metrics.append(scene_result)
    finally:
        progress.close()

    if not scene_metrics:
        raise ValueError("No test scenes were evaluated.")

    result: Dict[str, object] = {}
    for name in average_names:
        result[name] = sum(float(scene_result[name]) for scene_result in scene_metrics) / len(scene_metrics)
    result["time_ms"] = sum(float(scene_result["time_ms"]) for scene_result in scene_metrics) / len(scene_metrics)
    result["frames"] = sum(int(scene_result["frames"]) for scene_result in scene_metrics)
    result["scenes"] = len(scene_metrics)
    result["aggregation"] = "scene_macro"
    result["scene_metrics"] = scene_metrics
    return result


def print_scene_metrics(scene_metrics: List[Dict[str, object]]) -> None:
    print("Per-scene test metrics:")
    print(
        f"{'Scene':<24} {'Frames':>7} {'Loss':>11} "
        f"{'PSNR-RGB':>10} {'SSIM-RGB':>10} "
        f"{'PSNR-Y':>10} {'SSIM-Y':>10} {'Time(ms)':>10}"
    )
    for row in scene_metrics:
        print(
            f"{str(row['scene']):<24} {int(row['frames']):>7d} "
            f"{float(row['loss']):>11.6f} "
            f"{float(row['psnr_rgb']):>10.4f} "
            f"{float(row['ssim_rgb']):>10.6f} "
            f"{float(row['psnr_y']):>10.4f} "
            f"{float(row['ssim_y']):>10.6f} "
            f"{float(row['time_ms']):>10.3f}"
        )


def main() -> None:
    validate_settings()
    seed_everything(seed)
    device = resolve_device(device_name)
    amp_enabled = use_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ============================================================
    # 1. Save directories
    # ============================================================
    experiment_dir = output_root / flag
    last_checkpoint_path = experiment_dir / "last.pth"
    if (
        last_checkpoint_path.exists()
        and not auto_resume
        and not overwrite_existing
    ):
        raise FileExistsError(
            f"{last_checkpoint_path} already exists. Enable auto_resume, "
            "change flag, or explicitly set overwrite_existing=True."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    migrate_legacy_best_checkpoints(experiment_dir)

    # ============================================================
    # 2. Dataset
    # ============================================================
    backend = data_backend.strip().lower()
    selected_data_root = h5_data_root if backend == "h5" else data_root
    train_dataset = CausalVideoClipDataset(
        root=selected_data_root,
        split=train_split,
        scale=scale,
        channels=channels,
        sequence_length=sequence_length,
        patch_size=patch_size,
        frame_interval=frame_interval,
        clip_stride=clip_stride,
        augment=True,
        hr_folder_name=hr_folder_name,
        lr_folder_name=lr_folder_name,
        data_backend=backend,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    all_test_scenes = discover_video_scenes(
        selected_data_root,
        test_split,
        data_backend=backend,
        hr_folder_name=hr_folder_name,
        lr_folder_name=lr_folder_name,
    )
    test_scenes = select_test_scenes(all_test_scenes, test_scene_names)

    # ============================================================
    # 3. Model, optimizer, scheduler
    # ============================================================
    model = RenderVSR(**current_model_config()).to(device)
    optimizer = Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = create_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # ============================================================
    # 4. Resume
    # ============================================================
    current_epoch = epoch_start
    global_step = 0
    best_metric_values = {
        metric_name: float("-inf") for metric_name in best_metric_names
    }
    best_metric_epochs = {metric_name: 0 for metric_name in best_metric_names}
    history: List[Dict[str, object]] = []
    scene_history: List[Dict[str, object]] = []
    if auto_resume and last_checkpoint_path.is_file():
        (
            current_epoch,
            global_step,
            best_metric_values,
            best_metric_epochs,
            history,
            scene_history,
        ) = restore_training_state(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
    elif auto_resume:
        print("No last.pth found; training starts from scratch.")
    else:
        print("Auto resume is disabled; training starts from scratch.")

    # ============================================================
    # 5. Information
    # ============================================================
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    total_test_frames = sum(scene.num_frames for scene in test_scenes)
    print("\n" + "=" * 78)
    print(f"Experiment: {flag}")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Data backend: {backend}; root: {selected_data_root.resolve()}")
    print(train_dataset.summary())
    print(
        f"DataLoader: [B,T,C,H,W] = [{batch_size},{sequence_length},"
        f"{channels},{patch_size},{patch_size}] | shuffle=True"
    )
    print(
        f"Test every epoch: split={test_split!r}, "
        f"scenes={len(test_scenes)}, frames={total_test_frames}, "
        f"names={[scene.name for scene in test_scenes]}"
    )
    print(f"Model: {current_model_config()}; parameters: {parameter_count:,}")
    print(f"Best checkpoint metrics: test {list(best_metric_names)}")
    print(f"Output: {experiment_dir.resolve()}")
    print("=" * 78)

    if current_epoch > epoch_max:
        print(
            f"Checkpoint has completed epoch {current_epoch - 1}, which is "
            f">= epoch_max={epoch_max}. Nothing to train."
        )
        return

    # ============================================================
    # 6. Training: test performance is measured after every epoch
    # ============================================================
    while current_epoch <= epoch_max:
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        epoch_start_time = time.perf_counter()
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            current_epoch,
            global_step,
        )
        train_seconds = time.perf_counter() - epoch_start_time

        test_start_time = time.perf_counter()
        test_metrics = test_complete_scenes(
            model, test_scenes, device, amp_enabled, current_epoch
        )
        test_seconds = time.perf_counter() - test_start_time
        elapsed_seconds = time.perf_counter() - epoch_start_time
        current_scene_metrics = list(test_metrics["scene_metrics"])
        print_scene_metrics(current_scene_metrics)
        print(
            f"EPOCH {current_epoch:04d} | "
            f"train {train_seconds:.1f}s | test {test_seconds:.1f}s | "
            f"total {elapsed_seconds:.1f}s | "
            f"train loss {train_loss:.6f} | "
            f"scene-macro test loss {float(test_metrics['loss']):.6f} | "
            f"PSNR-RGB {float(test_metrics['psnr_rgb']):.4f} | "
            f"SSIM-RGB {float(test_metrics['ssim_rgb']):.6f} | "
            f"PSNR-Y {float(test_metrics['psnr_y']):.4f} | "
            f"SSIM-Y {float(test_metrics['ssim_y']):.6f} | "
            f"Time {float(test_metrics['time_ms']):.3f} ms/frame | "
            f"LR {epoch_learning_rate:.6e}"
        )

        improved_metrics = []
        for metric_name in best_metric_names:
            current_value = float(test_metrics[metric_name])
            if current_value > best_metric_values[metric_name]:
                best_metric_values[metric_name] = current_value
                best_metric_epochs[metric_name] = current_epoch
                improved_metrics.append(metric_name)

        row = {
            "epoch": current_epoch,
            "learning_rate": epoch_learning_rate,
            "train_loss": train_loss,
            "test_loss": float(test_metrics["loss"]),
            "test_psnr_rgb": float(test_metrics["psnr_rgb"]),
            "test_ssim_rgb": float(test_metrics["ssim_rgb"]),
            "test_psnr_y": float(test_metrics["psnr_y"]),
            "test_ssim_y": float(test_metrics["ssim_y"]),
            "test_time_ms": float(test_metrics["time_ms"]),
            "test_scenes": int(test_metrics["scenes"]),
            "test_frames": int(test_metrics["frames"]),
            "test_aggregation": str(test_metrics["aggregation"]),
        }
        history.append(row)
        save_history(history, experiment_dir / "training_history.csv")

        for scene_result in current_scene_metrics:
            scene_history.append(
                {
                    "epoch": current_epoch,
                    "scene": str(scene_result["scene"]),
                    "frames": int(scene_result["frames"]),
                    "test_loss": float(scene_result["loss"]),
                    "test_psnr_rgb": float(scene_result["psnr_rgb"]),
                    "test_ssim_rgb": float(scene_result["ssim_rgb"]),
                    "test_psnr_y": float(scene_result["psnr_y"]),
                    "test_ssim_y": float(scene_result["ssim_y"]),
                    "time_ms": float(scene_result["time_ms"]),
                }
            )
        save_scene_history_excel(
            scene_history,
            experiment_dir / "scene_metrics_history.xlsx",
        )

        scheduler.step()
        checkpoint = build_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            current_epoch,
            global_step,
            best_metric_values,
            best_metric_epochs,
            history,
            scene_history,
        )
        checkpoint.update(row)
        atomic_torch_save(checkpoint, last_checkpoint_path)

        for metric_name in improved_metrics:
            metric_checkpoint = dict(checkpoint)
            metric_checkpoint["selected_best_metric"] = metric_name
            metric_checkpoint["selected_best_value"] = best_metric_values[
                metric_name
            ]
            metric_checkpoint["selected_best_epoch"] = current_epoch
            best_checkpoint_path = experiment_dir / (
                f"best_{metric_name}_epoch_{current_epoch:04d}.pth"
            )
            atomic_torch_save(metric_checkpoint, best_checkpoint_path)
            remove_older_best_checkpoints(
                experiment_dir,
                metric_name,
                best_checkpoint_path,
            )
            print(
                f"New best test {metric_name}: "
                f"{best_metric_values[metric_name]:.6f} "
                f"(epoch {current_epoch}) -> {best_checkpoint_path.name}"
            )
        current_epoch += 1

    print(f"Finished: {flag}")


if __name__ == "__main__":
    # ============================================================
    # 1. Dataset
    # ============================================================
    data_root = Path(r"/media/ubuntu/b668ccd2-a59e-469f-bad1-8b09aecd5923/RDG_Dataset")
    # "h5" reads files generated by convert_images_to_h5.py. Change to
    # "images" to retain the original image-folder reader for comparison.
    data_backend = "h5"     # images   h5
    h5_data_root = data_root / "h5_data"
    train_split = "test_data"
    test_split = "test_data"  # Evaluated after every training epoch.
    hr_folder_name = "HR"
    lr_folder_name = "X4LR"

    # ============================================================
    # 3. Video clip and DataLoader
    # ============================================================
    sequence_length = 16
    frame_interval = 1
    clip_stride = 1
    patch_size = 128
    batch_size = 4
    workers = 4

    # ============================================================
    # 4. Training
    # ============================================================
    epoch_start = 1
    epoch_max = 500
    learning_rate = 7e-4
    min_learning_rate = 1e-7
    weight_decay = 0.0
    grad_clip = 1.0
    device_name = "auto"
    use_amp = True
    seed = 42
    save_every = 1000
    loss_skip_frames = 0

    # ============================================================
    # 5. Learning-rate scheduler
    # ============================================================
    scheduler_type = "multistep"

    multistep_milestones = [int(0.5 * epoch_max), int(0.7 * epoch_max), int(0.8 * epoch_max), int(0.9 * epoch_max)]
    multistep_gamma = 0.5

    warmup_epochs = 7
    warmup_start_factor = 0.1

    # ============================================================
    # 6. Test after every epoch
    # ============================================================
    test_crop_border = None
    skip_test_start_frames = 0
    test_scene_names = ["Bistro"]
    best_metric_names = ["psnr_rgb", "ssim_rgb"]

    # ============================================================
    # 7. Checkpoint and output
    # ============================================================
    output_root = Path("experiments")
    auto_resume = True
    overwrite_existing = False

    # ============================================================
    # 8. Model
    # ============================================================
    scale = 4
    channels = 3
    num_features = 64
    semantic_channels = 64
    dct_block = 8
    unet_levels = 2
    unet_base_channels = 64
    unet_max_channels = 1024
    conv_type = "depthwise"
    memory_groups = 8
    memory_topk = 4
    max_offset = 8.0
    num_recon_blocks = 2
    use_aligned_prev = True
    weichar, weitem, weidct, weidct2, weitem2 = 1.0, 0.0, 0.0, 0.0, 0.0
    from model.StructFreqSR import RenderVSR as RenderVSR
    time.sleep(600)
    flag = "StructFreqSR"
    main()
