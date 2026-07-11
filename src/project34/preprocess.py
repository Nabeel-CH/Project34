"""Preprocessing steps for the Step 1.1 pipeline.

Normalisation, breast segmentation, pectoral removal and CLAHE, plus the
`preprocess_one` orchestration. These are the deterministic image operations
extracted verbatim from Step 1.1; the pectoral mask is now passed in rather
than looked up from a notebook-global dict, so the function has no hidden state.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage as ndi

from project34.data import read_dicom


def normalise(img, mode="percentile", pct=(1, 99)):
    """Normalise a raw image to [0, 1].

    "percentile" clips at the 1st/99th percentile before rescaling (robust to
    the outlier pixels in INbreast); "minmax" is a plain min-max, kept for the
    ablation only.
    """
    if mode == "minmax":
        lo, hi = float(img.min()), float(img.max())
    elif mode == "percentile":
        lo, hi = np.percentile(img, pct)
        img = np.clip(img, lo, hi)
    else:
        raise ValueError(f"Unknown norm_mode: {mode!r}")

    img = (img - lo) / (hi - lo + 1e-8)
    return img


def to_uint8(img01):
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)


def make_breast_mask(img01):
    """Segment breast vs background: Otsu -> morphology -> largest CC -> fill."""
    u8 = to_uint8(img01)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    num, labels = cv2.connectedComponents(m)
    if num <= 1:
        mask = (m > 0).astype(np.uint8)
    else:
        sizes = [(labels == i).sum() for i in range(1, num)]
        biggest = 1 + int(np.argmax(sizes))
        mask = (labels == biggest).astype(np.uint8)

    mask = ndi.binary_fill_holes(mask).astype(np.uint8)
    return mask


def remove_pectoral(img01, pect_mask):
    """Zero out the pectoral-muscle region given its (0/1) mask."""
    out = img01.copy()
    out[pect_mask == 1] = 0.0
    return out


def apply_clahe(img01, mask, clip=2.0, tiles=(8, 8)):
    u8 = to_uint8(img01)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    enh = clahe.apply(u8).astype(np.float32) / 255.0
    return enh * mask.astype(np.float32)


def preprocess_one(dicom_path, pect_mask=None, do_contrast=True):
    """Run the full pipeline on one DICOM; return the intermediate + final arrays.

    `pect_mask` is the (0/1) pectoral mask to remove (None -> no removal).
    """
    norm = normalise(read_dicom(dicom_path))

    if pect_mask is None:
        pect_mask = np.zeros_like(norm, dtype=np.uint8)

    no_pect = remove_pectoral(norm, pect_mask)
    breast_mask = make_breast_mask(no_pect)
    bg_removed = no_pect * breast_mask.astype(np.float32)
    final = apply_clahe(bg_removed, breast_mask) if do_contrast else bg_removed

    return {
        "norm": norm,
        "pect_mask": pect_mask,
        "no_pect": no_pect,
        "breast_mask": breast_mask,
        "bg_removed": bg_removed,
        "final": final,
    }
