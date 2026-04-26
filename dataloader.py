from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from datasets import load_dataset


KEY_MAPPING: Dict[str, int] = {
    "q": 0,
    "w": 1,
    "e": 2,
    "r": 3,
    "t": 4,
    "y": 5,
    "u": 6,
    "i": 7,
    "o": 8,
    "p": 9,
    "a": 10,
    "s": 11,
    "d": 12,
    "f": 13,
    "g": 14,
    "h": 15,
    "j": 16,
    "k": 17,
    "l": 18,
    "z": 19,
    "x": 20,
    "c": 21,
    "v": 22,
    "b": 23,
    "n": 24,
    "m": 25,
    "1": 26,
    "2": 27,
    "3": 28,
    "4": 29,
    "5": 30,
    "6": 31,
    "7": 32,
    "8": 33,
    "9": 34,
    "0": 35,
    "space": 36,
    "nothing": 37,
}


@dataclass(slots=True)
class DatasetBuildConfig:
    dataset_name: str = "andrewt28/keystroke-typing-videos"
    crop_top: int = 0
    crop_left: int = 40
    crop_height: int = 300
    crop_width: int = 560
    resize_scale: float = 0.5
    grayscale: bool = True
    contrast_factor: float = 2.0
    negative_keep_prob: float = 1.0 / 30.0

    # Experiment 2: presence of data transformation and selection.
    apply_crop: bool = True
    apply_resize: bool = True
    apply_contrast: bool = True
    enable_negative_sampling: bool = True

    # Experiment 3: addition of time context.
    include_time_context: bool = False

    # Experiment 4: presence of generous labelling.
    label_strategy: str = "original"
    tolerance_before_ms: int = 30
    tolerance_after_ms: int = 30
    stale_keystroke_ms: int = 100

    def output_channels(self) -> int:
        return 3 if self.include_time_context else 1


DATASET_PRESETS: Dict[str, DatasetBuildConfig] = {
    "DATA B&W": DatasetBuildConfig(
        label_strategy="original",
        tolerance_before_ms=30,
        tolerance_after_ms=30,
        stale_keystroke_ms=100,
    ),
    "DATA B&W GENEROUS": DatasetBuildConfig(
        label_strategy="generous",
        tolerance_before_ms=30,
        tolerance_after_ms=30,
        stale_keystroke_ms=50,
    ),
    "DATA B&W w MEMORY": DatasetBuildConfig(
        include_time_context=True,
        label_strategy="original",
        tolerance_before_ms=60,
        tolerance_after_ms=60,
        stale_keystroke_ms=100,
    ),
}


def _prepare_frame_tensor(frame_data: torch.Tensor) -> torch.Tensor:
    image = frame_data
    if image.ndim == 3 and image.shape[-1] in (1, 3):
        image = image.permute(2, 0, 1)
    elif image.ndim == 2:
        image = image.unsqueeze(0)

    if image.dtype != torch.float32:
        image = image.to(torch.float32) / 255.0

    return image


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] == 1:
        return image
    r, g, b = image[0], image[1], image[2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.unsqueeze(0)


def _adjust_contrast(gray: torch.Tensor, factor: float) -> torch.Tensor:
    return torch.clamp((gray - 0.5) * factor + 0.5, 0.0, 1.0)


def preprocess_frame(frame_data: torch.Tensor, config: DatasetBuildConfig) -> torch.Tensor:
    image = _prepare_frame_tensor(frame_data)

    if config.apply_crop:
        image = TF.crop(
            image,
            top=config.crop_top,
            left=config.crop_left,
            height=config.crop_height,
            width=config.crop_width,
        )

    if config.apply_resize and config.resize_scale != 1.0:
        resize = nn.Upsample(scale_factor=config.resize_scale, mode="bilinear", align_corners=False)
        image = resize(image.unsqueeze(0)).squeeze(0)

    if config.grayscale:
        image = _to_grayscale(image)

    if config.apply_contrast and config.grayscale:
        image = _adjust_contrast(image, config.contrast_factor)

    return image


def _clean_keystrokes(entry_keystrokes: Iterable[dict]) -> List[dict]:
    cleaned: List[dict] = []
    for keystroke in entry_keystrokes:
        key = keystroke["key"].lower()
        if key in KEY_MAPPING:
            cleaned.append({"key": key, "timestamp_ms": keystroke["timestamp_ms"]})
    cleaned.sort(key=lambda item: item["timestamp_ms"])
    return cleaned


def _match_label_original(
    current_ms: float,
    keystrokes: List[dict],
    start_index: int,
    config: DatasetBuildConfig,
) -> Tuple[str, int]:
    label = "nothing"
    index = start_index

    while index < len(keystrokes):
        keystroke = keystrokes[index]
        timestamp = keystroke["timestamp_ms"]
        if current_ms < timestamp - config.tolerance_before_ms:
            break
        if current_ms <= timestamp + config.tolerance_after_ms:
            label = keystroke["key"]
            index += 1
            break
        index += 1

    return label, index


def _match_label_generous(
    current_ms: float,
    keystrokes: List[dict],
    start_index: int,
    config: DatasetBuildConfig,
) -> Tuple[str, int]:
    index = start_index
    while index < len(keystrokes) and current_ms > keystrokes[index]["timestamp_ms"] + config.stale_keystroke_ms:
        index += 1

    label = "nothing"
    if index < len(keystrokes):
        timestamp = keystrokes[index]["timestamp_ms"]
        if timestamp - config.tolerance_before_ms <= current_ms <= timestamp + config.tolerance_after_ms:
            label = keystrokes[index]["key"]

    return label, index


def _match_label(
    current_ms: float,
    keystrokes: List[dict],
    start_index: int,
    config: DatasetBuildConfig,
) -> Tuple[str, int]:
    if config.label_strategy == "generous":
        return _match_label_generous(current_ms, keystrokes, start_index, config)
    return _match_label_original(current_ms, keystrokes, start_index, config)


def process_split(dataset_split, config: DatasetBuildConfig) -> List[Tuple[torch.Tensor, int]]:
    samples: List[Tuple[torch.Tensor, int]] = []

    for entry in dataset_split:
        keystrokes = _clean_keystrokes(entry["keystrokes"])
        keystroke_index = 0
        video = entry["video"]

        if config.include_time_context:
            frame_buffer: deque[Tuple[float, torch.Tensor]] = deque(maxlen=3)

            try:
                first_frame = next(video)
            except StopIteration:
                continue

            first_ms = first_frame["pts"] * 1000.0
            first_tensor = preprocess_frame(first_frame["data"], config)
            frame_buffer.append((first_ms, first_tensor))

            while True:
                try:
                    next_frame = next(video)
                    next_ms = next_frame["pts"] * 1000.0
                    next_tensor = preprocess_frame(next_frame["data"], config)
                    frame_buffer.append((next_ms, next_tensor))
                except StopIteration:
                    next_frame = None

                if len(frame_buffer) == 1:
                    prev_frame = curr_frame = next_frame_tensor = frame_buffer[0][1]
                    current_ms = frame_buffer[0][0]
                elif len(frame_buffer) == 2:
                    prev_frame = curr_frame = frame_buffer[0][1]
                    next_frame_tensor = frame_buffer[1][1]
                    current_ms = frame_buffer[0][0]
                else:
                    _, prev_frame = frame_buffer[-3]
                    current_ms, curr_frame = frame_buffer[-2]
                    _, next_frame_tensor = frame_buffer[-1]

                label, keystroke_index = _match_label(current_ms, keystrokes, keystroke_index, config)
                keep_negative = not config.enable_negative_sampling or random.random() < config.negative_keep_prob
                if label != "nothing" or keep_negative:
                    stacked = torch.cat([prev_frame, curr_frame, next_frame_tensor], dim=0)
                    samples.append((stacked, KEY_MAPPING[label]))

                if next_frame is None:
                    break
        else:
            for frame in video:
                current_ms = frame["pts"] * 1000.0
                label, keystroke_index = _match_label(current_ms, keystrokes, keystroke_index, config)
                keep_negative = not config.enable_negative_sampling or random.random() < config.negative_keep_prob
                if label != "nothing" or keep_negative:
                    image_tensor = preprocess_frame(frame["data"], config)
                    samples.append((image_tensor, KEY_MAPPING[label]))

    return samples


def build_dataset_splits(
    config: DatasetBuildConfig,
    splits: Iterable[str] = ("train", "validation"),
) -> Dict[str, List[Tuple[torch.Tensor, int]]]:
    built: Dict[str, List[Tuple[torch.Tensor, int]]] = {}
    for split in splits:
        dataset_split = load_dataset(config.dataset_name, split=split)
        built[split] = process_split(dataset_split, config)
    return built


def save_dataset_splits(
    output_dir: str | Path,
    split_samples: Dict[str, List[Tuple[torch.Tensor, int]]],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name, samples in split_samples.items():
        torch.save(samples, output_path / f"{split_name}_samples.pth")

