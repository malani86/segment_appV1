# data_loader.py

import os
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


TRAIN_RESIZE_INTERPOLATION = cv2.INTER_LINEAR


def _normalize_array_to_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array

    if array.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    array = array.astype(np.float32)
    mn = float(np.min(array))
    mx = float(np.max(array))
    if mx - mn < 1e-8:
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = (array - mn) / (mx - mn)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)

def rolling_ball_correction_rgb(image, radius=50):
    """
    Apply Rolling Ball background correction to an RGB image.
    """
    channels = cv2.split(image)
    corrected_channels = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    for channel in channels:
        background = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(channel, background)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        corrected_channels.append(corrected)
    corrected_image = cv2.merge(corrected_channels)
    return corrected_image


def ensure_training_rgb_uint8(image):
    if isinstance(image, (str, os.PathLike, Path)):
        with Image.open(image) as pil_image:
            array = np.array(pil_image)
    elif isinstance(image, Image.Image):
        array = np.array(image)
    else:
        array = np.asarray(image)

    array = _normalize_array_to_uint8(array)
    if array.ndim == 2:
        return np.array(Image.fromarray(array).convert("RGB"), dtype=np.uint8)

    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.shape[-1] == 1:
        return np.array(Image.fromarray(array[..., 0]).convert("RGB"), dtype=np.uint8)

    if array.shape[-1] >= 3:
        return np.array(Image.fromarray(array[..., :3]).convert("RGB"), dtype=np.uint8)

    raise ValueError(f"Unsupported image shape: {array.shape}")


def resize_like_training(image, size):
    resize = A.Resize(
        height=size,
        width=size,
        interpolation=TRAIN_RESIZE_INTERPOLATION,
        mask_interpolation=cv2.INTER_NEAREST,
    )
    return resize(image=image)["image"]


def preprocess_rgb_like_training(image, radius=50, size=512):
    rgb = ensure_training_rgb_uint8(image)
    corrected = rolling_ball_correction_rgb(rgb, radius=radius)
    resized = resize_like_training(corrected, size=size)
    normalized = resized.astype(np.float32) / 255.0
    return normalized, rgb.shape[:2], corrected

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, mask_list, transform=None, return_filename=True, return_orig_size=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform
        # Resize images and masks to 256x256 using Albumentations
        self.resize = A.Resize(
            height=512,
            width=512,
            interpolation=TRAIN_RESIZE_INTERPOLATION,
            mask_interpolation=cv2.INTER_NEAREST,
        )
        self.return_filename = return_filename
        self.return_orig_size = return_orig_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        
        # Load image and apply rolling ball correction
        img = ensure_training_rgb_uint8(img_path)
        orig_h, orig_w = img.shape[:2]
        img = rolling_ball_correction_rgb(img, radius=50)
        
        # Load mask and convert to binary
        mask = np.array(Image.open(mask_path).convert("L"))
        mask[mask > 0] = 1
        
        # Resize
        resized = self.resize(image=img, mask=mask)
        img = resized["image"].astype(np.float32) / 255.0
        mask = resized["mask"]
        
        # Apply additional transforms if provided
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # Ensure mask has a channel dimension
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        if self.return_orig_size and self.return_filename:
            return img, mask, (orig_h, orig_w), self.image_list[idx]
        elif self.return_orig_size:
            return img, mask, (orig_h, orig_w)
        elif self.return_filename:
            return img, mask, self.image_list[idx]
        else:
            return img, mask
