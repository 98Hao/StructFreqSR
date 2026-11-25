"""Multi-scene sequence loading for causal recurrent video SR."""

from __future__ import annotations

import random
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None

Image.init()
_COMMON_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".webp",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    ".jp2",
    ".j2k",
    ".jpf",
    ".jpx",
    ".j2c",
    ".gif",
    ".tga",
    ".targa",
    ".pcx",
    ".ico",
    ".dds",
}
IMAGE_EXTENSIONS = frozenset(
    _COMMON_IMAGE_EXTENSIONS
    | {
        extension.lower()
        for extension, image_format in Image.registered_extensions().items()
        if image_format in Image.OPEN
    }
)


def natural_key(value: str) -> Tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def image_files(folder: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: natural_key(path.name),
    )


def _index_by_stem(paths: Iterable[Path], folder: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in paths:
        key = path.stem.casefold()
        if key in result:
            raise ValueError(
                f"Frame stem is not unique in {folder}: "
                f"{result[key].name!r} and {path.name!r}."
            )
        result[key] = path
    return result


@dataclass(frozen=True)
class SceneSequence:
    name: str
    lr_frames: Tuple[Path, ...]
    hr_frames: Tuple[Path, ...]

    @property
    def num_frames(self) -> int:
        return len(self.lr_frames)


@dataclass
class H5SceneSequence:
    name: str
    h5_path: Path
    frame_names: Tuple[str, ...]
    lr_shape: Tuple[int, int, int]
    hr_shape: Tuple[int, int, int]
    scale: int
    channels: int
    _handle: Any = field(default=None, init=False, repr=False, compare=False)
    _handle_pid: Optional[int] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def _get_handle(self):
        _require_h5py()
        current_pid = os.getpid()
        if self._handle is None or self._handle_pid != current_pid:
            self.close()

            self._handle = h5py.File(str(self.h5_path), "r")
            self._handle_pid = current_pid
        return self._handle

    def read_frame(self, frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
        handle = self._get_handle()
        return (
            np.asarray(handle["lr"][frame_index], dtype=np.uint8),
            np.asarray(handle["hr"][frame_index], dtype=np.uint8),
        )

    def read_hr_frame(self, frame_index: int) -> np.ndarray:
        handle = self._get_handle()
        return np.asarray(handle["hr"][frame_index], dtype=np.uint8)

    def read_patch(
        self,
        frame_indices: Sequence[int],
        top: int,
        left: int,
        patch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        handle = self._get_handle()
        indices = list(frame_indices)
        bottom = top + patch_size
        right = left + patch_size
        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_bottom = bottom * self.scale
        hr_right = right * self.scale
        return (
            np.asarray(
                handle["lr"][indices, top:bottom, left:right, :],
                dtype=np.uint8,
            ),
            np.asarray(
                handle["hr"][
                    indices, hr_top:hr_bottom, hr_left:hr_right, :
                ],
                dtype=np.uint8,
            ),
        )

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None
                self._handle_pid = None

    def __getstate__(self) -> Dict[str, object]:
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()


VideoScene = Union[SceneSequence, H5SceneSequence]


@dataclass(frozen=True)
class ClipSample:
    scene_index: int
    start_index: int


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "HDF5 loading requires h5py. Install the dependencies in "
            "requirements_vsr.txt or run: pip install h5py"
        )


def discover_paired_scenes(
    root: str | Path,
    split: str,
    hr_folder_name: str = "HR",
    lr_folder_name: str = "x4LR",
) -> List[SceneSequence]:
    split_root = Path(root).expanduser() / split
    hr_root = split_root / hr_folder_name
    lr_root = split_root / lr_folder_name

    if not hr_root.is_dir():
        raise FileNotFoundError(
            f"High-resolution directory does not exist: {hr_root}. Expected "
            f"<data_root>/{split}/{hr_folder_name}/<scene>/Image."
        )
    if not lr_root.is_dir():
        raise FileNotFoundError(
            f"Low-resolution directory does not exist: {lr_root}. Expected "
            f"<data_root>/{split}/{lr_folder_name}/<scene>/Image."
        )

    hr_image_dirs = {
        path.parent.relative_to(hr_root): path
        for path in hr_root.rglob("Image")
        if path.is_dir()
    }
    lr_image_dirs = {
        path.parent.relative_to(lr_root): path
        for path in lr_root.rglob("Image")
        if path.is_dir()
    }
    if not hr_image_dirs:
        raise FileNotFoundError(
            f"No scene Image directories found under {hr_root}. Expected "
            f"{hr_folder_name}/<scene>/Image/<frame>."
        )
    if not lr_image_dirs:
        raise FileNotFoundError(
            f"No scene Image directories found under {lr_root}. Expected "
            f"{lr_folder_name}/<scene>/Image/<frame>."
        )

    scenes: List[SceneSequence] = []
    errors: List[str] = []
    missing_lr_scenes = sorted(
        set(hr_image_dirs) - set(lr_image_dirs),
        key=lambda path: natural_key(path.as_posix()),
    )
    extra_lr_scenes = sorted(
        set(lr_image_dirs) - set(hr_image_dirs),
        key=lambda path: natural_key(path.as_posix()),
    )
    if missing_lr_scenes:
        errors.append(
            f"Scenes present in {hr_folder_name} but missing from "
            f"{lr_folder_name}: "
            f"{[path.as_posix() for path in missing_lr_scenes[:20]]}."
        )
    if extra_lr_scenes:
        errors.append(
            f"Scenes present in {lr_folder_name} but missing from "
            f"{hr_folder_name}: "
            f"{[path.as_posix() for path in extra_lr_scenes[:20]]}."
        )

    common_scene_paths = sorted(
        set(hr_image_dirs) & set(lr_image_dirs),
        key=lambda path: natural_key(path.as_posix()),
    )
    for scene_rel in common_scene_paths:
        hr_image_dir = hr_image_dirs[scene_rel]
        scene_name = scene_rel.as_posix()
        lr_image_dir = lr_image_dirs[scene_rel]

        hr_by_stem = _index_by_stem(image_files(hr_image_dir), hr_image_dir)
        lr_by_stem = _index_by_stem(image_files(lr_image_dir), lr_image_dir)
        missing_lr = sorted(set(hr_by_stem) - set(lr_by_stem), key=natural_key)
        missing_hr = sorted(set(lr_by_stem) - set(hr_by_stem), key=natural_key)
        if missing_lr or missing_hr:
            errors.append(
                f"Scene {scene_name!r}: LR/HR names differ. "
                f"Missing LR={missing_lr[:10]}, missing HR={missing_hr[:10]}."
            )
            continue

        stems = sorted(hr_by_stem, key=natural_key)
        if not stems:
            errors.append(f"Scene {scene_name!r}: no supported images.")
            continue
        scenes.append(
            SceneSequence(
                name=scene_name,
                lr_frames=tuple(lr_by_stem[stem] for stem in stems),
                hr_frames=tuple(hr_by_stem[stem] for stem in stems),
            )
        )

    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))
    if not scenes:
        raise ValueError(f"No usable scenes found in {split_root}.")
    return scenes


def _decode_h5_strings(values: np.ndarray) -> Tuple[str, ...]:
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return tuple(names)


def discover_h5_scenes(root: str | Path, split: str) -> List[H5SceneSequence]:
    _require_h5py()
    split_root = Path(root).expanduser() / split
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"HDF5 split directory does not exist: {split_root}. Expected "
            f"<h5_root>/{split}/<scene>.h5."
        )
    h5_paths = sorted(
        (
            path
            for path in split_root.rglob("*.h5")
            if path.is_file() and not path.name.endswith(".tmp.h5")
        ),
        key=lambda path: natural_key(path.relative_to(split_root).as_posix()),
    )
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 scene files found under {split_root}.")

    scenes: List[H5SceneSequence] = []
    errors: List[str] = []
    scene_names = set()
    for h5_path in h5_paths:
        try:
            with h5py.File(str(h5_path), "r") as handle:
                if "lr" not in handle or "hr" not in handle:
                    raise KeyError("datasets 'lr' and 'hr' are required")
                lr_dataset = handle["lr"]
                hr_dataset = handle["hr"]
                if lr_dataset.ndim != 4 or hr_dataset.ndim != 4:
                    raise ValueError(
                        f"lr/hr must be THWC arrays, got "
                        f"{lr_dataset.shape} and {hr_dataset.shape}"
                    )
                if lr_dataset.dtype != np.uint8 or hr_dataset.dtype != np.uint8:
                    raise TypeError(
                        f"lr/hr dtype must be uint8, got "
                        f"{lr_dataset.dtype} and {hr_dataset.dtype}"
                    )
                if lr_dataset.shape[0] != hr_dataset.shape[0]:
                    raise ValueError("lr/hr frame counts differ")
                if lr_dataset.shape[-1] not in (1, 3):
                    raise ValueError(
                        f"channel count must be 1 or 3, got {lr_dataset.shape[-1]}"
                    )
                if hr_dataset.shape[-1] != lr_dataset.shape[-1]:
                    raise ValueError("lr/hr channel counts differ")

                stored_scale = int(handle.attrs.get("scale", 0))
                if stored_scale < 2:
                    raise ValueError(f"invalid scale attribute: {stored_scale}")
                expected_hr_shape = (
                    lr_dataset.shape[0],
                    lr_dataset.shape[1] * stored_scale,
                    lr_dataset.shape[2] * stored_scale,
                    lr_dataset.shape[3],
                )
                if tuple(hr_dataset.shape) != expected_hr_shape:
                    raise ValueError(
                        f"hr shape {hr_dataset.shape} is not lr shape "
                        f"{lr_dataset.shape} x{stored_scale}"
                    )

                if "frame_names" in handle:
                    frame_names = _decode_h5_strings(handle["frame_names"][:])
                else:
                    frame_names = tuple(
                        f"{index:08d}.png"
                        for index in range(lr_dataset.shape[0])
                    )
                if len(frame_names) != lr_dataset.shape[0]:
                    raise ValueError(
                        "frame_names length differs from the frame count"
                    )
                default_name = h5_path.relative_to(split_root).with_suffix("").as_posix()
                stored_name = handle.attrs.get("scene_name", default_name)
                if isinstance(stored_name, bytes):
                    stored_name = stored_name.decode("utf-8")
                scene_name = str(stored_name)
                if scene_name in scene_names:
                    raise ValueError(f"duplicate scene name {scene_name!r}")

                scene_names.add(scene_name)
                scenes.append(
                    H5SceneSequence(
                        name=scene_name,
                        h5_path=h5_path,
                        frame_names=frame_names,
                        lr_shape=tuple(int(value) for value in lr_dataset.shape[1:]),
                        hr_shape=tuple(int(value) for value in hr_dataset.shape[1:]),
                        scale=stored_scale,
                        channels=int(lr_dataset.shape[-1]),
                    )
                )
        except (OSError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{h5_path}: {error}")

    if errors:
        raise ValueError("HDF5 dataset validation failed:\n" + "\n".join(errors))
    if not scenes:
        raise ValueError(f"No usable HDF5 scenes found in {split_root}.")
    return scenes


def discover_video_scenes(
    root: str | Path,
    split: str,
    data_backend: str = "images",
    hr_folder_name: str = "HR",
    lr_folder_name: str = "X4LR",
) -> List[VideoScene]:
    backend = data_backend.strip().lower()
    if backend == "images":
        return discover_paired_scenes(
            root,
            split,
            hr_folder_name=hr_folder_name,
            lr_folder_name=lr_folder_name,
        )
    if backend == "h5":
        return discover_h5_scenes(root, split)
    raise ValueError(
        f"Unknown data_backend={data_backend!r}; choose 'images' or 'h5'."
    )


def load_image_uint8(path: Path, channels: int) -> np.ndarray:
    mode = "L" if channels == 1 else "RGB"
    try:
        with Image.open(path) as image:

            image.seek(0)
            image = ImageOps.exif_transpose(image)
            array = np.asarray(image.convert(mode), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise OSError(
            f"Failed to decode image {path}. The extension may be recognized "
            "but its optional Pillow codec is unavailable or the file is "
            "damaged."
        ) from error
    if channels == 1:
        array = array[..., None]
    return array


def _uint8_sequence_to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim != 4:
        raise ValueError(f"Expected THWC uint8 array, got shape {array.shape}.")
    return (
        torch.from_numpy(np.ascontiguousarray(array))
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .div_(255.0)
    )


def load_image(path: Path, channels: int) -> torch.Tensor:
    array = load_image_uint8(path, channels)
    return _uint8_sequence_to_tensor(array[None])[0]


def load_paired_frame(
    scene: VideoScene,
    frame_index: int,
    channels: int,
    scale: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(scene, H5SceneSequence):
        if scene.channels != channels:
            raise ValueError(
                f"Scene {scene.name!r} stores {scene.channels} channels, but "
                f"the model requests channels={channels}. Recreate the H5 data "
                "with the matching channels setting."
            )
        if scene.scale != scale:
            raise ValueError(
                f"Scene {scene.name!r} stores scale=x{scene.scale}, but the "
                f"model requests scale=x{scale}."
            )
        lr_array, hr_array = scene.read_frame(frame_index)
        lr = _uint8_sequence_to_tensor(lr_array[None])[0]
        hr = _uint8_sequence_to_tensor(hr_array[None])[0]
        return lr, hr

    lr = load_image(scene.lr_frames[frame_index], channels)
    hr = load_image(scene.hr_frames[frame_index], channels)
    expected_hr = (channels, lr.shape[-2] * scale, lr.shape[-1] * scale)
    if tuple(hr.shape) != expected_hr:
        raise ValueError(
            f"{scene.name}/{scene.hr_frames[frame_index].name}: HR shape "
            f"{tuple(hr.shape)} != LR shape {tuple(lr.shape)} x{scale}; "
            f"expected {expected_hr}."
        )
    return lr, hr


def scene_frame_name(scene: VideoScene, frame_index: int) -> str:
    if isinstance(scene, H5SceneSequence):
        return scene.frame_names[frame_index]
    return scene.hr_frames[frame_index].name


def load_hr_frame(
    scene: VideoScene,
    frame_index: int,
    channels: int,
    scale: int,
) -> torch.Tensor:
    if isinstance(scene, H5SceneSequence):
        if scene.channels != channels or scene.scale != scale:
            raise ValueError(
                f"Scene {scene.name!r} stores channels={scene.channels}, "
                f"scale=x{scene.scale}; requested channels={channels}, "
                f"scale=x{scale}."
            )
        return _uint8_sequence_to_tensor(
            scene.read_hr_frame(frame_index)[None]
        )[0]
    return load_image(scene.hr_frames[frame_index], channels)


def _paired_random_crop(
    lr_sequence: torch.Tensor,
    hr_sequence: torch.Tensor,
    patch_size: int,
    scale: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = lr_sequence.shape
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"LR size {(height, width)} is smaller than patch "
            f"{patch_size}x{patch_size}."
        )
    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)
    bottom = top + patch_size
    right = left + patch_size
    return (
        lr_sequence[..., top:bottom, left:right],
        hr_sequence[
            ...,
            top * scale : bottom * scale,
            left * scale : right * scale,
        ],
    )


def _paired_augment(
    tensors: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    outputs = list(tensors)
    if random.random() < 0.5:
        outputs = [torch.flip(tensor, dims=(-1,)) for tensor in outputs]
    if random.random() < 0.5:
        outputs = [torch.flip(tensor, dims=(-2,)) for tensor in outputs]
    rotations = random.randint(0, 3)
    if rotations:
        outputs = [torch.rot90(tensor, rotations, dims=(-2, -1)) for tensor in outputs]
    return tuple(tensor.contiguous() for tensor in outputs)


class CausalVideoClipDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        scale: int,
        channels: int,
        sequence_length: int,
        patch_size: int,
        frame_interval: int = 1,
        clip_stride: int = 1,
        augment: bool = True,
        hr_folder_name: str = "HR",
        lr_folder_name: str = "X4LR",
        data_backend: str = "images",
    ) -> None:
        if scale < 2:
            raise ValueError("scale must be >= 2.")
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3.")
        if min(sequence_length, patch_size, frame_interval, clip_stride) < 1:
            raise ValueError(
                "sequence_length, patch_size, frame_interval, and clip_stride "
                "must all be positive."
            )

        self.root = Path(root).expanduser()
        self.split = split
        self.scale = int(scale)
        self.channels = int(channels)
        self.sequence_length = int(sequence_length)
        self.patch_size = int(patch_size)
        self.frame_interval = int(frame_interval)
        self.clip_stride = int(clip_stride)
        self.augment = bool(augment)
        self.hr_folder_name = hr_folder_name
        self.lr_folder_name = lr_folder_name
        self.data_backend = data_backend.strip().lower()
        self.scenes = discover_video_scenes(
            self.root,
            self.split,
            data_backend=self.data_backend,
            hr_folder_name=self.hr_folder_name,
            lr_folder_name=self.lr_folder_name,
        )

        for scene in self.scenes:
            if isinstance(scene, H5SceneSequence):
                if scene.scale != self.scale:
                    raise ValueError(
                        f"Scene {scene.name!r} stores scale=x{scene.scale}, but "
                        f"the dataset requests scale=x{self.scale}."
                    )
                if scene.channels != self.channels:
                    raise ValueError(
                        f"Scene {scene.name!r} stores {scene.channels} channels, "
                        f"but the dataset requests channels={self.channels}."
                    )

        required_span = (self.sequence_length - 1) * self.frame_interval + 1
        self.samples: List[ClipSample] = []
        short_scenes: List[str] = []
        for scene_index, scene in enumerate(self.scenes):
            max_start = scene.num_frames - required_span
            if max_start < 0:
                short_scenes.append(f"{scene.name}({scene.num_frames})")
                continue
            starts = list(range(0, max_start + 1, self.clip_stride))
            if starts[-1] != max_start:
                starts.append(max_start)
            self.samples.extend(ClipSample(scene_index, start) for start in starts)

        if short_scenes:
            raise ValueError(
                f"sequence_length={self.sequence_length}, frame_interval="
                f"{self.frame_interval} requires {required_span} frames, but "
                f"these scenes are too short: {', '.join(short_scenes)}"
            )
        if not self.samples:
            raise ValueError("No training clips could be constructed.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        scene = self.scenes[sample.scene_index]
        indices = [
            sample.start_index + offset * self.frame_interval
            for offset in range(self.sequence_length)
        ]
        if isinstance(scene, H5SceneSequence):
            lr_height, lr_width, _ = scene.lr_shape
            if lr_height < self.patch_size or lr_width < self.patch_size:
                raise ValueError(
                    f"Scene {scene.name!r}: LR size {(lr_height, lr_width)} "
                    f"is smaller than patch {self.patch_size}x{self.patch_size}."
                )
            top = random.randint(0, lr_height - self.patch_size)
            left = random.randint(0, lr_width - self.patch_size)
            lr_array, hr_array = scene.read_patch(
                indices, top, left, self.patch_size
            )
            lr_sequence = _uint8_sequence_to_tensor(lr_array)
            hr_sequence = _uint8_sequence_to_tensor(hr_array)
        else:
            lr_frames = []
            hr_frames = []
            lr_shape = None
            for frame_index in indices:
                lr, hr = load_paired_frame(
                    scene, frame_index, self.channels, self.scale
                )
                if lr_shape is None:
                    lr_shape = tuple(lr.shape)
                elif tuple(lr.shape) != lr_shape:
                    raise ValueError(
                        f"LR size changes within scene {scene.name!r}: "
                        f"{lr_shape} vs {tuple(lr.shape)}."
                    )
                lr_frames.append(lr)
                hr_frames.append(hr)

            lr_sequence = torch.stack(lr_frames, dim=0)
            hr_sequence = torch.stack(hr_frames, dim=0)
            lr_sequence, hr_sequence = _paired_random_crop(
                lr_sequence, hr_sequence, self.patch_size, self.scale
            )
        if self.augment:
            lr_sequence, hr_sequence = _paired_augment(
                (lr_sequence, hr_sequence)
            )
        return {
            "lr": lr_sequence,
            "hr": hr_sequence,
            "scene": scene.name,
            "start_index": sample.start_index,
        }

    def summary(self) -> str:
        total_frames = sum(scene.num_frames for scene in self.scenes)
        return (
            f"backend={self.data_backend!r}, split={self.split!r}, "
            f"scenes={len(self.scenes)}, "
            f"frames={total_frames}, clips={len(self)}, "
            f"T={self.sequence_length}, interval={self.frame_interval}, "
            f"scale=x{self.scale}, channels={self.channels}, "
            f"folders={self.hr_folder_name}/{self.lr_folder_name}"
        )

    def close(self) -> None:
        for scene in self.scenes:
            if isinstance(scene, H5SceneSequence):
                scene.close()

    def __del__(self) -> None:
        if hasattr(self, "scenes"):
            self.close()


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
