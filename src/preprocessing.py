"""Image preprocessing and augmentation utilities for steering prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf

INPUT_HEIGHT = 66
INPUT_WIDTH = 200
INPUT_CHANNELS = 3
CROP_TOP = 60
CROP_BOTTOM = 25
TRANSFER_HEIGHT = 96
TRANSFER_WIDTH = 160


def load_image_rgb(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_pilotnet(image_rgb: np.ndarray) -> np.ndarray:
    """Crop, resize, convert RGB to YUV, and normalize to [-1, 1]."""
    image = image_rgb[CROP_TOP:image_rgb.shape[0] - CROP_BOTTOM, :, :]
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image.astype(np.float32) / 127.5 - 1.0


def preprocess_mobilenet(image_rgb: np.ndarray) -> np.ndarray:
    """Crop and preprocess an image for MobileNetV2."""
    image = image_rgb[CROP_TOP:image_rgb.shape[0] - CROP_BOTTOM, :, :]
    image = cv2.resize(image, (TRANSFER_WIDTH, TRANSFER_HEIGHT), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.asarray(image, dtype=np.float32)


def random_brightness(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] *= rng.uniform(0.6, 1.4)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_shadow(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = rng.integers(0, width)
    x2 = rng.integers(0, width)
    yy, xx = np.mgrid[0:height, 0:width]
    mask = ((xx - x1) * (height - 1) - (x2 - x1) * yy) > 0
    if rng.random() < 0.5:
        mask = ~mask
    shadowed = image.astype(np.float32)
    shadowed[mask] *= rng.uniform(0.45, 0.75)
    return np.clip(shadowed, 0, 255).astype(np.uint8)


def translate_image_and_steering(
    image: np.ndarray,
    steering: float,
    rng: np.random.Generator,
    max_horizontal_shift: int = 40,
    steering_per_pixel: float = 0.002,
) -> Tuple[np.ndarray, float]:
    horizontal_shift = rng.uniform(-max_horizontal_shift, max_horizontal_shift)
    vertical_shift = rng.uniform(-5, 5)
    matrix = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
    translated = cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE
    )
    steering = float(np.clip(steering + horizontal_shift * steering_per_pixel, -1.0, 1.0))
    return translated, steering


def augment_image(image: np.ndarray, steering: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    image_aug = image.copy()
    steering_aug = float(steering)
    if rng.random() < 0.5:
        image_aug, steering_aug = translate_image_and_steering(image_aug, steering_aug, rng)
    if rng.random() < 0.5:
        image_aug = random_brightness(image_aug, rng)
    if rng.random() < 0.3:
        image_aug = random_shadow(image_aug, rng)
    if rng.random() < 0.5:
        image_aug = cv2.flip(image_aug, 1)
        steering_aug = -steering_aug
    return image_aug, steering_aug
