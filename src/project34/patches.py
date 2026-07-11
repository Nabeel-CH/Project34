"""Deterministic patch helpers shared by Step 1.2 and Step 1.3.

Cropping a mask's bounding box, resizing to 224x224, and saving a patch as
.npy plus a normalised .png preview. The Step 1.3 background *sampler* itself
is RNG-sensitive and stays in that notebook; only these deterministic helpers
are shared here.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


def crop_from_mask(image, mask, margin=20):
    """Crop `image` to the mask's bounding box plus a margin.

    Returns (patch, (y0, y1, x0, x1)) or None if the mask is empty.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    H, W = image.shape
    y0 = max(0, y0 - margin); y1 = min(H - 1, y1 + margin)
    x0 = max(0, x0 - margin); x1 = min(W - 1, x1 + margin)
    return image[y0:y1 + 1, x0:x1 + 1], (y0, y1, x0, x1)


def resize_to_224(patch):
    return cv2.resize(
        patch.astype(np.float32),
        (224, 224),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32)


def save_patch(patch01, out_stem):
    """Save a patch as <out_stem>.npy plus a min-max-normalised <out_stem>.png."""
    np.save(str(out_stem.with_suffix(".npy")), patch01.astype(np.float32))

    p = patch01.astype(np.float32)
    p = p - np.nanmin(p)
    if np.nanmax(p) > 0:
        p = p / np.nanmax(p)

    img8 = (np.clip(p, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img8).save(str(out_stem.with_suffix(".png")))
