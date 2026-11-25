
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import h5py
except ImportError as error:
    raise ImportError(
        "convert_images_to_h5.py requires h5py. Install it in the same Python "
        "environment used to run this file, for example: pip install h5py"
    ) from error

from video_datasets import load_image_uint8


FORMAT_VERSION = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SceneSequence:
    name: str
    lr_frames: Tuple[Path, ...]
    hr_frames: Tuple[Path, ...]

    @property
    def num_frames(self) -> int:
        return len(self.hr_frames)


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).casefold())


def _natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def _list_image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=_natural_key,
    )


def _resolve_named_directory(parent: Path, requested_name: str) -> Path | None:
    exact = parent / requested_name
    if exact.is_dir():
        return exact

    if not parent.is_dir():
        return None

    target = _normalized_name(requested_name)
    matches = [
        child
        for child in parent.iterdir()
        if child.is_dir() and _normalized_name(child.name) == target
    ]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous folder name {requested_name!r} under {parent}: "
            f"{[path.name for path in matches]}"
        )
    return None


def _resolve_split_directory(image_data_root: Path, split: str) -> Path:
    image_data_root = image_data_root.expanduser()

    if not image_data_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {image_data_root}"
        )

    if _normalized_name(image_data_root.name) == _normalized_name(split):
        return image_data_root

    split_dir = _resolve_named_directory(image_data_root, split)
    if split_dir is not None:
        return split_dir

    children = sorted(
        [path.name for path in image_data_root.iterdir() if path.is_dir()]
    )
    raise FileNotFoundError(
        f"Could not find split {split!r} under:\n"
        f"  {image_data_root}\n"
        f"Available folders are:\n"
        f"  {children}\n\n"
        "Folder matching already ignores spaces, underscores, hyphens and case. "
        "If the split is elsewhere, change image_data_root or splits in __main__."
    )


def _resolve_frame_directory(base: Path) -> Path | None:
    if not base.is_dir():
        return None

    image_dir = _resolve_named_directory(base, "Image")
    if image_dir is not None and _list_image_files(image_dir):
        return image_dir

    if _list_image_files(base):
        return base

    return None


def _pair_scene(
    scene_name: str,
    lr_folder: Path,
    hr_folder: Path,
) -> SceneSequence:
    lr_frames = _list_image_files(lr_folder)
    hr_frames = _list_image_files(hr_folder)

    if not lr_frames:
        raise FileNotFoundError(
            f"No LR image frames found for scene {scene_name!r}: {lr_folder}"
        )
    if not hr_frames:
        raise FileNotFoundError(
            f"No HR image frames found for scene {scene_name!r}: {hr_folder}"
        )

    lr_by_name = {path.name: path for path in lr_frames}
    hr_by_name = {path.name: path for path in hr_frames}

    lr_names = set(lr_by_name)
    hr_names = set(hr_by_name)

    if lr_names != hr_names:
        missing_lr = sorted(hr_names - lr_names)[:10]
        missing_hr = sorted(lr_names - hr_names)[:10]
        raise ValueError(
            f"LR/HR frame names differ in scene {scene_name!r}.\n"
            f"Missing in LR (first 10): {missing_lr}\n"
            f"Missing in HR (first 10): {missing_hr}\n"
            f"LR folder: {lr_folder}\n"
            f"HR folder: {hr_folder}"
        )

    ordered_names = [path.name for path in hr_frames]
    return SceneSequence(
        name=scene_name,
        lr_frames=tuple(lr_by_name[name] for name in ordered_names),
        hr_frames=tuple(hr_by_name[name] for name in ordered_names),
    )


def _discover_modality_first(
    split_dir: Path,
    hr_folder_name: str,
    lr_folder_name: str,
) -> List[SceneSequence]:
    hr_root = _resolve_named_directory(split_dir, hr_folder_name)
    lr_root = _resolve_named_directory(split_dir, lr_folder_name)

    if hr_root is None or lr_root is None:
        return []

    hr_scene_dirs = {
        path.name: path for path in hr_root.iterdir() if path.is_dir()
    }
    lr_scene_dirs = {
        path.name: path for path in lr_root.iterdir() if path.is_dir()
    }

    common_names = sorted(
        set(hr_scene_dirs).intersection(lr_scene_dirs),
        key=lambda name: _natural_key(Path(name)),
    )

    scenes: List[SceneSequence] = []
    for scene_name in common_names:
        hr_frames_dir = _resolve_frame_directory(hr_scene_dirs[scene_name])
        lr_frames_dir = _resolve_frame_directory(lr_scene_dirs[scene_name])
        if hr_frames_dir is None or lr_frames_dir is None:
            continue
        scenes.append(_pair_scene(scene_name, lr_frames_dir, hr_frames_dir))

    return scenes


def _discover_scene_first(
    split_dir: Path,
    hr_folder_name: str,
    lr_folder_name: str,
) -> List[SceneSequence]:
    scenes: List[SceneSequence] = []

    candidate_scene_dirs = sorted(
        [path for path in split_dir.iterdir() if path.is_dir()],
        key=_natural_key,
    )

    for scene_dir in candidate_scene_dirs:
        if _normalized_name(scene_dir.name) in {
            _normalized_name(hr_folder_name),
            _normalized_name(lr_folder_name),
            _normalized_name("h5_data"),
        }:
            continue

        hr_base = _resolve_named_directory(scene_dir, hr_folder_name)
        lr_base = _resolve_named_directory(scene_dir, lr_folder_name)

        if hr_base is None or lr_base is None:
            continue

        hr_frames_dir = _resolve_frame_directory(hr_base)
        lr_frames_dir = _resolve_frame_directory(lr_base)

        if hr_frames_dir is None or lr_frames_dir is None:
            continue

        scenes.append(
            _pair_scene(scene_dir.name, lr_frames_dir, hr_frames_dir)
        )

    return scenes


def _print_directory_preview(folder: Path, max_depth: int = 2, max_entries: int = 40) -> str:
    lines: List[str] = []
    count = 0

    def walk(current: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_entries or not current.is_dir():
            return

        try:
            children = sorted(current.iterdir(), key=lambda p: p.name.casefold())
        except OSError:
            return

        for child in children:
            if count >= max_entries:
                break
            if child.name == "h5_data":
                continue
            indent = "  " * depth
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{indent}{child.name}{suffix}")
            count += 1
            if child.is_dir():
                walk(child, depth + 1)

    walk(folder, 0)
    if count >= max_entries:
        lines.append("  ...")
    return "\n".join(lines)


def discover_paired_scenes_flexible(
    image_data_root: Path,
    split: str,
    hr_folder_name: str,
    lr_folder_name: str,
) -> Tuple[Path, List[SceneSequence]]:
    split_dir = _resolve_split_directory(image_data_root, split)

    scenes = _discover_modality_first(
        split_dir,
        hr_folder_name,
        lr_folder_name,
    )
    if scenes:
        layout = (
            f"{split_dir}/{hr_folder_name}/<scene>/Image and "
            f"{split_dir}/{lr_folder_name}/<scene>/Image"
        )
        print(f"[layout] {split!r} -> modality-first")
        print(f"         resolved split: {split_dir}")
        print(f"         example layout: {layout}")
        return split_dir, scenes

    scenes = _discover_scene_first(
        split_dir,
        hr_folder_name,
        lr_folder_name,
    )
    if scenes:
        layout = (
            f"{split_dir}/<scene>/{hr_folder_name}/Image and "
            f"{split_dir}/<scene>/{lr_folder_name}/Image"
        )
        print(f"[layout] {split!r} -> scene-first")
        print(f"         resolved split: {split_dir}")
        print(f"         example layout: {layout}")
        return split_dir, scenes

    preview = _print_directory_preview(split_dir)
    raise FileNotFoundError(
        f"Could not discover paired HR/LR scenes in:\n"
        f"  {split_dir}\n\n"
        "Supported layouts are:\n"
        f"  1. <split>/{hr_folder_name}/<scene>/Image/*.png\n"
        f"     <split>/{lr_folder_name}/<scene>/Image/*.png\n"
        f"  2. <split>/<scene>/{hr_folder_name}/Image/*.png\n"
        f"     <split>/<scene>/{lr_folder_name}/Image/*.png\n"
        "The intermediate Image folder may also be omitted.\n\n"
        f"Directory preview:\n{preview}"
    )


def dataset_options(
    compression: str,
    gzip_level: int,
) -> Dict[str, object]:
    compression_name = compression.strip().lower()
    if compression_name in ("", "none"):
        return {}
    if compression_name == "lzf":
        return {"compression": "lzf", "shuffle": True}
    if compression_name == "gzip":
        if not 0 <= gzip_level <= 9:
            raise ValueError("gzip_level must be in [0, 9].")
        return {
            "compression": "gzip",
            "compression_opts": gzip_level,
            "shuffle": True,
        }
    raise ValueError("compression must be 'none', 'lzf', or 'gzip'.")


def validate_existing_file(
    h5_path: Path,
    scene: SceneSequence,
    scale: int,
    channels: int,
) -> None:
    with h5py.File(str(h5_path), "r") as handle:
        if "lr" not in handle or "hr" not in handle or "frame_names" not in handle:
            raise ValueError(f"Existing file is incomplete: {h5_path}")

        lr_dataset = handle["lr"]
        hr_dataset = handle["hr"]

        if lr_dataset.dtype != np.uint8 or hr_dataset.dtype != np.uint8:
            raise ValueError(f"Existing file is not uint8: {h5_path}")

        if lr_dataset.shape[0] != scene.num_frames:
            raise ValueError(
                f"Existing {h5_path} has {lr_dataset.shape[0]} frames, but "
                f"the source scene now has {scene.num_frames}."
            )

        if int(handle.attrs.get("scale", -1)) != scale:
            raise ValueError(f"Existing file uses another scale: {h5_path}")

        if int(handle.attrs.get("channels", -1)) != channels:
            raise ValueError(
                f"Existing file uses another channel count: {h5_path}"
            )

        stored_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["frame_names"][:]
        ]
        source_names = [path.name for path in scene.hr_frames]

        if stored_names != source_names:
            raise ValueError(
                f"Existing frame-name list differs from the source: {h5_path}"
            )


def validate_written_file(
    h5_path: Path,
    scene: SceneSequence,
    scale: int,
    channels: int,
) -> None:
    validate_existing_file(h5_path, scene, scale, channels)

    indices = sorted(set((0, scene.num_frames - 1)))
    with h5py.File(str(h5_path), "r") as handle:
        for frame_index in indices:
            expected_lr = load_image_uint8(
                scene.lr_frames[frame_index],
                channels,
            )
            expected_hr = load_image_uint8(
                scene.hr_frames[frame_index],
                channels,
            )

            if not np.array_equal(handle["lr"][frame_index], expected_lr):
                raise ValueError(
                    f"LR read-back verification failed: {h5_path}, "
                    f"frame {frame_index}."
                )

            if not np.array_equal(handle["hr"][frame_index], expected_hr):
                raise ValueError(
                    f"HR read-back verification failed: {h5_path}, "
                    f"frame {frame_index}."
                )


def convert_scene(
    scene: SceneSequence,
    output_path: Path,
    split: str,
    scale: int,
    channels: int,
    lr_chunk_size: int,
    compression: str,
    gzip_level: int,
    overwrite_existing: bool,
    verify_after_write: bool,
    progress_every_frames: int,
) -> str:
    if output_path.exists() and not overwrite_existing:
        validate_existing_file(output_path, scene, scale, channels)
        print(f"[skip] {scene.name}: valid file already exists -> {output_path}")
        return "skipped"

    first_lr = load_image_uint8(scene.lr_frames[0], channels)
    first_hr = load_image_uint8(scene.hr_frames[0], channels)

    expected_hr_shape = (
        first_lr.shape[0] * scale,
        first_lr.shape[1] * scale,
        channels,
    )
    if first_hr.shape != expected_hr_shape:
        raise ValueError(
            f"{scene.name}/{scene.hr_frames[0].name}: HR shape "
            f"{first_hr.shape} != LR shape {first_lr.shape} x{scale}; "
            f"expected {expected_hr_shape}."
        )

    frame_count = scene.num_frames
    lr_height, lr_width, _ = first_lr.shape
    hr_height, hr_width, _ = first_hr.shape

    lr_chunk = (
        1,
        min(lr_chunk_size, lr_height),
        min(lr_chunk_size, lr_width),
        channels,
    )
    hr_chunk = (
        1,
        min(lr_chunk_size * scale, hr_height),
        min(lr_chunk_size * scale, hr_width),
        channels,
    )

    options = dataset_options(compression, gzip_level)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    if temporary_path.exists():
        temporary_path.unlink()

    start_time = time.perf_counter()

    try:
        with h5py.File(str(temporary_path), "w") as handle:
            handle.attrs["format"] = "causal_video_sr_scene"
            handle.attrs["format_version"] = FORMAT_VERSION
            handle.attrs["scene_name"] = scene.name
            handle.attrs["split"] = split
            handle.attrs["scale"] = scale
            handle.attrs["channels"] = channels
            handle.attrs["layout"] = "THWC"
            handle.attrs["dtype"] = "uint8"

            lr_dataset = handle.create_dataset(
                "lr",
                shape=(frame_count, lr_height, lr_width, channels),
                dtype=np.uint8,
                chunks=lr_chunk,
                **options,
            )

            hr_dataset = handle.create_dataset(
                "hr",
                shape=(frame_count, hr_height, hr_width, channels),
                dtype=np.uint8,
                chunks=hr_chunk,
                **options,
            )

            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                "frame_names",
                data=np.asarray(
                    [path.name for path in scene.hr_frames],
                    dtype=object,
                ),
                dtype=string_dtype,
            )

            for frame_index, (lr_path, hr_path) in enumerate(
                zip(scene.lr_frames, scene.hr_frames)
            ):
                if frame_index == 0:
                    lr_array, hr_array = first_lr, first_hr
                else:
                    lr_array = load_image_uint8(lr_path, channels)
                    hr_array = load_image_uint8(hr_path, channels)

                if lr_array.shape != first_lr.shape:
                    raise ValueError(
                        f"{scene.name}/{lr_path.name}: LR shape changed from "
                        f"{first_lr.shape} to {lr_array.shape}. One H5 array "
                        "requires a fixed resolution within each scene."
                    )

                if hr_array.shape != first_hr.shape:
                    raise ValueError(
                        f"{scene.name}/{hr_path.name}: HR shape changed from "
                        f"{first_hr.shape} to {hr_array.shape}."
                    )

                lr_dataset[frame_index] = lr_array
                hr_dataset[frame_index] = hr_array

                completed = frame_index + 1
                if (
                    progress_every_frames > 0
                    and (
                        completed % progress_every_frames == 0
                        or completed == frame_count
                    )
                ):
                    print(
                        f"  {scene.name}: {completed}/{frame_count} frames"
                    )

            handle.flush()

        temporary_path.replace(output_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    if verify_after_write:
        validate_written_file(
            output_path,
            scene,
            scale,
            channels,
        )

    elapsed = time.perf_counter() - start_time
    size_gib = output_path.stat().st_size / (1024**3)

    print(
        f"[done] {scene.name}: {frame_count} frames, {elapsed:.1f}s, "
        f"{size_gib:.3f} GiB -> {output_path}"
    )

    return "converted"


def convert_split(
    image_data_root: Path,
    h5_data_root: Path,
    split: str,
    hr_folder_name: str,
    lr_folder_name: str,
    scale: int,
    channels: int,
    lr_chunk_size: int,
    compression: str,
    gzip_level: int,
    overwrite_existing: bool,
    verify_after_write: bool,
    progress_every_frames: int,
) -> Tuple[int, int]:
    split_dir, scenes = discover_paired_scenes_flexible(
        image_data_root=image_data_root,
        split=split,
        hr_folder_name=hr_folder_name,
        lr_folder_name=lr_folder_name,
    )

    print(
        f"\nSplit requested={split!r}, resolved={split_dir.name!r}: "
        f"{len(scenes)} scenes, "
        f"{sum(scene.num_frames for scene in scenes)} frames"
    )

    converted = 0
    skipped = 0

    output_split_dir = h5_data_root / split

    for scene in scenes:
        output_path = output_split_dir / f"{scene.name}.h5"

        status = convert_scene(
            scene=scene,
            output_path=output_path,
            split=split,
            scale=scale,
            channels=channels,
            lr_chunk_size=lr_chunk_size,
            compression=compression,
            gzip_level=gzip_level,
            overwrite_existing=overwrite_existing,
            verify_after_write=verify_after_write,
            progress_every_frames=progress_every_frames,
        )

        converted += int(status == "converted")
        skipped += int(status == "skipped")

    return converted, skipped


if __name__ == "__main__":
    image_data_root = Path(
        r"/media/ubuntu/b668ccd2-a59e-469f-bad1-8b09aecd5923/RDG_Dataset"
    )

    h5_data_root = image_data_root / "h5_data"

    splits = ["training_data"]

    hr_folder_name = "HR"
    lr_folder_name = "X4LR"

    scale = 4
    channels = 3

    lr_chunk_size = 64

    compression = "lzf"
    gzip_level = 1

    overwrite_existing = False
    verify_after_write = True
    progress_every_frames = 20

    if scale < 2:
        raise ValueError("scale must be >= 2.")

    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3.")

    if lr_chunk_size < 1:
        raise ValueError("lr_chunk_size must be positive.")

    if not splits:
        raise ValueError("splits must not be empty.")

    total_converted = 0
    total_skipped = 0
    overall_start = time.perf_counter()

    for split_name in splits:
        converted_count, skipped_count = convert_split(
            image_data_root=image_data_root,
            h5_data_root=h5_data_root,
            split=split_name,
            hr_folder_name=hr_folder_name,
            lr_folder_name=lr_folder_name,
            scale=scale,
            channels=channels,
            lr_chunk_size=lr_chunk_size,
            compression=compression,
            gzip_level=gzip_level,
            overwrite_existing=overwrite_existing,
            verify_after_write=verify_after_write,
            progress_every_frames=progress_every_frames,
        )

        total_converted += converted_count
        total_skipped += skipped_count

    print("\n" + "=" * 78)
    print(f"H5 root: {h5_data_root.resolve()}")
    print(
        f"Converted scenes: {total_converted}; "
        f"skipped scenes: {total_skipped}"
    )
    print(
        f"Total time: "
        f"{time.perf_counter() - overall_start:.1f}s"
    )
    print("Conversion completed successfully.")
