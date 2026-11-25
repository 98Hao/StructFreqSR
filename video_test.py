"""Test causal VSR checkpoints with configurable IQA metrics and VMAF."""

from __future__ import annotations

import gc
import re
import shutil
import subprocess
import tempfile
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json
import pyiqa
import torch
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from tqdm import tqdm

from video_datasets import discover_video_scenes, load_paired_frame, scene_frame_name
from video_io import clean_state_dict, load_checkpoint, resolve_device, tensor_to_pil


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


FULL_REFERENCE_METRICS = {"psnr", "ssim", "lpips", "dists"}
NO_REFERENCE_METRICS = {"niqe", "clipiqa", "musiq"}
DISTRIBUTION_METRICS = {"fid"}
VIDEO_METRICS = {"vmaf"}
SUPPORTED_METRICS = FULL_REFERENCE_METRICS | NO_REFERENCE_METRICS | DISTRIBUTION_METRICS | VIDEO_METRICS


@dataclass
class TestConfig:
    """One test experiment; actual values are written in __main__."""

    name: str
    enabled: bool
    weights: Path
    data_root: Path
    h5_data_root: Path
    data_backend: str
    split: str
    hr_folder_name: str
    lr_folder_name: str
    output_root: Path
    device: str
    use_amp: bool
    crop_border: int | None
    skip_start_frames: int
    save_images: bool
    metrics: tuple[str, ...]
    scene_names: tuple[str, ...] | None = None
    vmaf_fps: int = 24
    vmaf_model: str = "version=vmaf_v0.6.1"
    vmaf_threads: int = 4
    warmup_steps: int = 10
    profile_complexity: bool = True


PROFILE_COLUMNS = ("params_k", "gflops_frame")
PROFILE_LABELS = {"params_k": "Params (K)", "gflops_frame": "GFLOPs/frame"}


@torch.inference_mode()
def measure_gflops_per_frame(model, lr: torch.Tensor, state, use_amp: bool) -> float | None:

    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=True) as prof:
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction, next_state = model.forward_step(lr, state)
            if lr.is_cuda:
                torch.cuda.synchronize(lr.device)
            del prediction, next_state

        flops = sum(int(getattr(evt, "flops", 0) or 0) for evt in prof.key_averages())
        if flops <= 0:
            print("[模型统计] Profiler 没有记录到 FLOPs；GFLOPs/frame 留空。")
            return None
        return flops / 1e9
    except Exception as exc:
        print(f"[模型统计] GFLOPs 分析失败，结果留空：{type(exc).__name__}: {exc}")
        return None


@torch.inference_mode()
def warmup_and_profile(model, scenes, model_config, device, use_amp,
                       warmup_steps: int, profile_complexity: bool) -> dict:

    stats = {
        "params_k": sum(parameter.numel() for parameter in model.parameters()) / 1e3,
        "gflops_frame": None,
    }
    if not scenes or (warmup_steps == 0 and not profile_complexity):
        return stats

    scene = scenes[0]
    state = None
    lr = None
    try:
        if warmup_steps:
            print(f"[预热] {scene.name}，真实 LR 分辨率，递归预热 {warmup_steps} 帧（不计入测试）")
        for i in range(warmup_steps):
            lr_cpu, _ = load_paired_frame(
                scene, i % scene.num_frames,
                int(model_config["channels"]), int(model_config["scale"]),
            )
            lr = lr_cpu.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction, state = model.forward_step(lr, state)
            del prediction, lr_cpu
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if profile_complexity:
            lr_cpu, _ = load_paired_frame(
                scene, warmup_steps % scene.num_frames,
                int(model_config["channels"]), int(model_config["scale"]),
            )
            lr = lr_cpu.unsqueeze(0).to(device)
            del lr_cpu
            stats["gflops_frame"] = measure_gflops_per_frame(model, lr, state, use_amp)
        return stats
    finally:
        del state, lr
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def print_model_profile(stats: dict) -> None:
    print(f"[模型统计] 参数量（K）: {stats['params_k']:.3f}")
    value = stats.get("gflops_frame")
    print(f"[模型统计] GFLOPs/frame: {'N/A' if value is None else f'{value:.4f}'}")
    print("[说明] GFLOPs 基于 PyTorch Profiler 可覆盖算子；不保证包含全部自定义运算。")


def normalize_metrics(metric_names: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(str(name).strip().lower() for name in metric_names)
    if not names:
        raise ValueError("metrics must contain at least one metric.")
    if len(names) != len(set(names)):
        raise ValueError(f"metrics contains duplicates: {names}.")
    unsupported = [name for name in names if name not in SUPPORTED_METRICS]
    if unsupported:
        raise ValueError(
            f"Unsupported metrics: {unsupported}. Supported metrics are "
            f"{sorted(SUPPORTED_METRICS)}."
        )
    return names


def build_metric_models(metric_names: tuple[str, ...], device: torch.device) -> Dict[str, object]:
    metric_models: Dict[str, object] = {}

    for name in metric_names:
        if name == "vmaf":
            continue

        if name in ("psnr", "ssim"):
            metric = pyiqa.create_metric(
                name,
                test_y_channel=False,
                color_space="rgb",
                device=device,
            )
        else:
            metric = pyiqa.create_metric(name, device=device)

        if hasattr(metric, "eval"):
            metric.eval()
        metric_models[name] = metric

    return metric_models


def crop_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    border: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if border < 0:
        raise ValueError(f"crop_border must be non-negative, got {border}.")
    if border == 0:
        return prediction, target

    height, width = prediction.shape[-2:]
    if height <= 2 * border or width <= 2 * border:
        raise ValueError(
            f"Cannot crop a {border}-pixel border from image size {height}x{width}."
        )

    return (
        prediction[..., border:-border, border:-border],
        target[..., border:-border, border:-border],
    )


@torch.no_grad()
def calculate_frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metric_models: Dict[str, object],
    metric_names: tuple[str, ...],
    border: int,
) -> Dict[str, float]:
    """Calculate frame-level pyiqa metrics. FID/VMAF are calculated per scene."""
    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    prediction_fr, target_fr = crop_pair(prediction, target, border)

    values: Dict[str, float] = {}

    for name in metric_names:
        if name in DISTRIBUTION_METRICS or name in VIDEO_METRICS:
            continue

        metric = metric_models[name]

        if name in FULL_REFERENCE_METRICS:
            score = metric(prediction_fr, target_fr)
        elif name in NO_REFERENCE_METRICS:
            score = metric(prediction)
        else:
            continue

        values[name] = float(score.detach().mean().item())

    return values


def check_libvmaf() -> str:

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError("没有找到 FFmpeg")

    vmaf_bin = (
        Path.home()
        / "tools/vmaf/libvmaf/build/tools/vmaf"
    )

    if not vmaf_bin.is_file():
        raise RuntimeError(
            f"没有找到独立 VMAF 程序: {vmaf_bin}"
        )

    result = subprocess.run(
        [str(vmaf_bin), "--version"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"独立 VMAF 无法运行:\n{result.stderr}"
        )

    print(f"FFmpeg 路径: {ffmpeg}")
    print(f"独立 VMAF 路径: {vmaf_bin}")

    return ffmpeg

def calculate_vmaf(
    ffmpeg: str,
    distorted_folder: Path,
    reference_folder: Path,
    fps: int,
    model: str,
    threads: int,
) -> float:

    vmaf_bin = (
        Path.home()
        / "tools/vmaf/libvmaf/build/tools/vmaf"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        sr_y4m = tmp / "sr.y4m"
        hr_y4m = tmp / "hr.y4m"
        vmaf_json = tmp / "vmaf.json"

        for folder, output in (
            (distorted_folder, sr_y4m),
            (reference_folder, hr_y4m),
        ):
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-framerate", str(fps),
                "-start_number", "0",
                "-i", str(folder / "%06d.png"),
                "-pix_fmt", "yuv420p",
                "-f", "yuv4mpegpipe",
                str(output),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"PNG 转 Y4M 失败:\n{result.stderr}"
                )

        cmd = [
            str(vmaf_bin),
            "--reference", str(hr_y4m),
            "--distorted", str(sr_y4m),
            "--model", model,
            "--threads", str(threads),
            "--output", str(vmaf_json),
            "--json",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"VMAF 计算失败:\n{result.stderr}"
            )

        with open(vmaf_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        return float(
            data["pooled_metrics"]["vmaf"]["mean"]
        )



def save_summary_excel(
    excel_path: Path,
    scene_summaries: list[Dict[str, object]],
    overall_summary: Dict[str, object],
    metric_names: tuple[str, ...],
    model_stats: dict | None = None,
) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    ws = workbook.active
    ws.title = "Summary"

    model_stats = model_stats or {}
    headers = ["scene", "frames", "time_ms"] + list(metric_names) + [PROFILE_LABELS[k] for k in PROFILE_COLUMNS]
    ws.append(headers)

    for cell in ws[1]:
        cell.font = Font(bold=True)

    for summary in scene_summaries:
        ws.append(
            [
                summary["scene"],
                summary["frames"],
                summary["time_ms"],
                *[summary.get(metric_name) for metric_name in metric_names],
                *[model_stats.get(key) for key in PROFILE_COLUMNS],
            ]
        )

    ws.append(
        [
            "Average",
            overall_summary["frames"],
            overall_summary["time_ms"],
            *[overall_summary.get(metric_name) for metric_name in metric_names],
            *[model_stats.get(key) for key in PROFILE_COLUMNS],
        ]
    )
    for cell in ws[ws.max_row]:
        cell.font = Font(bold=True)

    ws.freeze_panes = "A2"

    for column_index, column_cells in enumerate(ws.columns, start=1):
        max_length = max(
            len("" if cell.value is None else str(cell.value))
            for cell in column_cells
        )
        ws.column_dimensions[get_column_letter(column_index)].width = min(
            max_length + 2, 24
        )

    detail = workbook.create_sheet("ModelStats")
    detail.append(["Statistic", "Value", "Meaning"])
    notes = {
        "params_k": "模型参数量（千个，1 K = 1000 个参数，不含 buffers）",
        "gflops_frame": "预热后单帧 forward_step 的 PyTorch Profiler 可统计 FLOPs（1 G = 10^9）；不保证覆盖所有自定义算子",
    }
    for k in PROFILE_COLUMNS:
        detail.append([PROFILE_LABELS[k], model_stats.get(k), notes.get(k, "")])
    for cell in detail[1]:
        cell.font = Font(bold=True)
    detail.freeze_panes = "A2"
    detail.column_dimensions["A"].width = 22
    detail.column_dimensions["B"].width = 22
    detail.column_dimensions["C"].width = 90
    workbook.save(excel_path)


def format_metric_summary(
    summary: Dict[str, object],
    metric_names: tuple[str, ...],
) -> str:
    parts = []
    for name in metric_names:
        value = summary.get(name)
        if value is not None:
            parts.append(f"{name.upper()} {float(value):.6f}")
    return " | ".join(parts)


def run_test(config: TestConfig) -> None:
    if config.skip_start_frames < 0:
        raise ValueError("skip_start_frames must be non-negative.")

    metric_names = normalize_metrics(config.metrics)

    weights_path = resolve_project_path(config.weights)
    output_root = resolve_project_path(config.output_root)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    device = resolve_device(config.device)
    use_amp = config.use_amp and device.type == "cuda"
    checkpoint = load_checkpoint(weights_path, device)

    if "model_config" not in checkpoint:
        raise KeyError(
            f"{weights_path} has no model_config. Use a checkpoint saved by "
            "the current video_train.py."
        )

    model_config = dict(checkpoint["model_config"])
    model = RenderVSR(**model_config).to(device)
    model.load_state_dict(
        clean_state_dict(checkpoint.get("model", checkpoint)),
        strict=True,
    )
    model.eval()
    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    del checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if int(model_config["channels"]) != 3:
        raise ValueError(
            "This evaluation script is configured for RGB metrics, but "
            f"model channels={model_config['channels']}."
        )

    backend = config.data_backend.strip().lower()
    if backend not in ("images", "h5"):
        raise ValueError("data_backend must be 'images' or 'h5'.")

    selected_data_root = resolve_project_path(
        config.h5_data_root if backend == "h5" else config.data_root
    )
    scenes = discover_video_scenes(
        selected_data_root,
        config.split,
        data_backend=backend,
        hr_folder_name=config.hr_folder_name,
        lr_folder_name=config.lr_folder_name,
    )

    if config.scene_names is not None:
        if not config.scene_names:
            raise ValueError("scene_names must be None or a nonempty tuple of scene names.")
        names = [name.casefold() for name in config.scene_names]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate scene names: {config.scene_names}")
        scene_lookup = {scene.name.casefold(): scene for scene in scenes}
        missing = [name for name in config.scene_names if name.casefold() not in scene_lookup]
        if missing:
            raise ValueError(f"Unknown scenes: {missing}. Available: {[scene.name for scene in scenes]}")
        scenes = [scene_lookup[name] for name in names]

    border = (
        int(model_config["scale"])
        if config.crop_border is None
        else int(config.crop_border)
    )

    total_frames = sum(scene.num_frames for scene in scenes)
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    model_stats = warmup_and_profile(
        model, scenes, model_config, device, use_amp,
        config.warmup_steps, config.profile_complexity,
    )
    print_model_profile(model_stats)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    checkpoint_bytes = weights_path.stat().st_size
    print(f"总参数量: {total_params:,} | {total_params / 1e3:.2f} K | {total_params / 1e6:.4f} M")
    print(f"可训练参数: {trainable_params:,}")
    print(f"参数张量内存: {parameter_bytes / (1024 ** 2):.2f} MiB")
    print(f"权重文件大小: {checkpoint_bytes / (1024 ** 2):.2f} MiB")
    metric_models = build_metric_models(metric_names, device)
    ffmpeg = check_libvmaf() if "vmaf" in metric_names else None
    excel_path = output_root / f"{config.name}.xlsx"

    print("\n" + "=" * 100)
    print(f"Test name: {config.name}")
    print(f"Device: {device}; AMP: {use_amp}")
    print(f"Data backend: {backend}; root: {selected_data_root.resolve()}")
    print(f"Scenes: {len(scenes)}; total frames: {total_frames}")
    print(f"Metrics: {list(metric_names)}")
    print(f"Model: {model_config}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Weights: {weights_path.resolve()}")
    print(f"Image output: {output_root}/<scene>/{config.name}/")
    print(f"Excel output: {excel_path.resolve()}")
    if "vmaf" in metric_names:
        print(
            f"VMAF: FFmpeg/libvmaf | model={config.vmaf_model} | "
            f"fps={config.vmaf_fps} | pool=mean"
        )
    print("=" * 100, flush=True)

    scene_summaries: list[Dict[str, object]] = []
    need_temp_sequences = bool(
        set(metric_names) & (DISTRIBUTION_METRICS | VIDEO_METRICS)
    )

    with tempfile.TemporaryDirectory(prefix=f"{config.name}_video_metrics_") as temp_dir:
        temp_root = Path(temp_dir)

        with torch.inference_mode():
            with tqdm(
                total=total_frames, desc="All scenes", unit="frame", position=0,
                leave=True, mininterval=0.5, dynamic_ncols=False, ncols=110,
                file=sys.stdout, disable=False,
            ) as total_progress:
                for scene_index, scene in enumerate(scenes, start=1):
                    total_progress.set_postfix_str(
                        f"scene {scene_index}/{len(scenes)}: {scene.name}", refresh=True
                    )
                    state = None
                    scene_metric_sums = {
                        name: 0.0 for name in metric_names
                        if name not in DISTRIBUTION_METRICS and name not in VIDEO_METRICS
                    }
                    scene_metric_counts = {name: 0 for name in scene_metric_sums}
                    scene_time_ms = 0.0
                    scene_measured_frames = 0

                    output_folder = output_root / scene.name / config.name
                    if config.save_images:
                        output_folder.mkdir(parents=True, exist_ok=True)

                    if need_temp_sequences:
                        temp_sr_folder = temp_root / scene.name / "SR"
                        temp_gt_folder = temp_root / scene.name / "GT"
                        temp_sr_folder.mkdir(parents=True, exist_ok=True)
                        temp_gt_folder.mkdir(parents=True, exist_ok=True)

                    sequence_index = 0

                    with tqdm(
                        total=scene.num_frames, desc=f"Scene {scene.name}", unit="frame",
                        position=1, leave=False, mininterval=0.5,
                        dynamic_ncols=False, ncols=110, file=sys.stdout, disable=False,
                    ) as frame_progress:
                        for frame_index in range(scene.num_frames):
                            lr, hr = load_paired_frame(
                                scene,
                                frame_index,
                                int(model_config["channels"]),
                                int(model_config["scale"]),
                            )
                            lr = lr.unsqueeze(0).to(device)
                            hr = hr.unsqueeze(0).to(device)

                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            start_time = time.perf_counter()
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                prediction, state = model.forward_step(lr, state)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                            prediction = prediction.float().clamp(0.0, 1.0)
                            prediction = torch.round(prediction * 255.0) / 255.0
                            hr = hr.float().clamp(0.0, 1.0)

                            frame_name = scene_frame_name(scene, frame_index)
                            output_name = f"{Path(frame_name).stem}.png"
                            if config.save_images:
                                tensor_to_pil(prediction).save(output_folder / output_name)

                            if frame_index >= config.skip_start_frames:
                                frame_metrics = calculate_frame_metrics(
                                    prediction, hr, metric_models, metric_names, border
                                )
                                for metric_name, value in frame_metrics.items():
                                    scene_metric_sums[metric_name] += value
                                    scene_metric_counts[metric_name] += 1

                                scene_time_ms += elapsed_ms
                                scene_measured_frames += 1

                                if need_temp_sequences:
                                    sequence_name = f"{sequence_index:06d}.png"
                                    tensor_to_pil(prediction).save(temp_sr_folder / sequence_name)
                                    tensor_to_pil(hr).save(temp_gt_folder / sequence_name)
                                    sequence_index += 1

                                info = {"infer": f"{scene_time_ms / scene_measured_frames:.1f}ms"}
                                for name in ("psnr", "ssim"):
                                    if scene_metric_counts.get(name, 0):
                                        info[name] = f"{scene_metric_sums[name] / scene_metric_counts[name]:.4f}"
                                frame_progress.set_postfix(info, refresh=False)

                            frame_progress.update(1)
                            total_progress.update(1)

                        if scene_measured_frames == 0:
                            raise ValueError(
                                f"Scene {scene.name!r} has no measured frames after "
                                f"skip_start_frames={config.skip_start_frames}."
                            )

                        scene_summary: Dict[str, object] = {
                            "scene": scene.name,
                            "frames": scene_measured_frames,
                            "time_ms": scene_time_ms / scene_measured_frames,
                        }
                        for metric_name in scene_metric_sums:
                            count = scene_metric_counts[metric_name]
                            if count > 0:
                                scene_summary[metric_name] = scene_metric_sums[metric_name] / count

                        if "fid" in metric_names:
                            frame_progress.set_postfix_str("Calculating FID...", refresh=True)
                            fid_score = metric_models["fid"](str(temp_sr_folder), str(temp_gt_folder))
                            scene_summary["fid"] = float(fid_score.detach().mean().item())

                        if "vmaf" in metric_names:
                            frame_progress.set_postfix_str("Calculating VMAF...", refresh=True)
                            scene_summary["vmaf"] = calculate_vmaf(
                                ffmpeg=ffmpeg,
                                distorted_folder=temp_sr_folder,
                                reference_folder=temp_gt_folder,
                                fps=config.vmaf_fps,
                                model=config.vmaf_model,
                                threads=config.vmaf_threads,
                            )

                    scene_summaries.append(scene_summary)
                    tqdm.write(
                        f"Scene [{scene_index}/{len(scenes)}]: {scene.name} | "
                        f"Frames: {scene_measured_frames} | "
                        f"Average time: {float(scene_summary['time_ms']):.3f} ms/frame",
                        file=sys.stdout,
                    )
                    metric_text = format_metric_summary(scene_summary, metric_names)
                    if metric_text:
                        tqdm.write(metric_text, file=sys.stdout)

    if not scene_summaries:
        raise ValueError(f"No scenes were evaluated: {config.name}")

    overall_summary: Dict[str, object] = {
        "frames": sum(int(row["frames"]) for row in scene_summaries),
        "time_ms": sum(float(row["time_ms"]) for row in scene_summaries) / len(scene_summaries),
    }
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in scene_summaries if row.get(metric_name) is not None]
        if values:
            overall_summary[metric_name] = sum(values) / len(values)

    save_summary_excel(
        excel_path=excel_path,
        scene_summaries=scene_summaries,
        overall_summary=overall_summary,
        metric_names=metric_names,
        model_stats=model_stats,
    )

    print("\n" + "=" * 100)
    print(
        f"All scenes average | Scenes: {len(scene_summaries)} | "
        f"Frames: {overall_summary['frames']} | "
        f"Average time: {float(overall_summary['time_ms']):.3f} ms/frame"
    )
    metric_text = format_metric_summary(overall_summary, metric_names)
    if metric_text:
        print(metric_text)
    print(f"Excel saved to: {excel_path.resolve()}")
    print("=" * 100)


if __name__ == "__main__":

    from model.StructFreqSR import RenderVSR
    # ======================================================================
    # 1. Dataset
    # ======================================================================
    data_root = Path(
        r"/media/ubuntu/b668ccd2-a59e-469f-bad1-8b09aecd5923/RDG_Dataset"
    )
    h5_data_root = data_root / "h5_data"
    data_backend = "h5"

    # ======================================================================
    # 2. Metrics
    # ======================================================================
    metrics = (
        "psnr",
        "ssim",
        "lpips",
        "dists",
        "fid",
        "niqe",
        "clipiqa",
        "musiq",
        "vmaf",
    )

    # ======================================================================
    # 3. Test experiments
    # ======================================================================
    test_experiments = [
        TestConfig(
            name="ours",
            enabled=True,
            weights=Path(
                "best.pth"
            ),
            data_root=data_root,
            h5_data_root=h5_data_root,
            data_backend=data_backend,
            split="test_data",
            hr_folder_name="HR",
            lr_folder_name="X4LR",
            output_root=Path("OutputImage"),
            device="auto",
            use_amp=True,
            crop_border=None,
            skip_start_frames=0,
            save_images=True,
            metrics=metrics,
            scene_names=(None),

            vmaf_fps=24,
            vmaf_model="version=vmaf_v0.6.1",
            vmaf_threads=4,
            warmup_steps=10,
            profile_complexity=True,
        ),
    ]

    enabled_experiments = [
        config for config in test_experiments if config.enabled
    ]

    if not enabled_experiments:
        raise ValueError("No enabled test experiments in video_test.py.")

    experiment_names = [config.name for config in enabled_experiments]
    if len(experiment_names) != len(set(experiment_names)):
        raise ValueError(
            f"Enabled test experiment names must be unique: {experiment_names}"
        )

    print(f"Enabled test experiments: {experiment_names}")

    for index, experiment_config in enumerate(enabled_experiments, start=1):
        print(f"\nExperiment {index}/{len(enabled_experiments)}: {experiment_config.name}", flush=True)
        run_test(experiment_config)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
