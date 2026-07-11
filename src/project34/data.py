"""Data loading and annotation helpers for the Project34 Step 0-1 notebooks.

These are the deterministic loaders/parsers that were previously duplicated
across the inspection and preprocessing notebooks: DICOM reading, the OsiriX
plist mass-ROI parser, polygon rasterising, the LabelMe pectoral-mask loader,
the file-id / sheet / index builders, and the BI-RADS / ACR label mappings.
Each is a faithful extraction of the notebook version, so importing it here
instead of redefining it inline produces identical results.
"""

from __future__ import annotations

import json
import re
import plistlib

import numpy as np
import pandas as pd
import cv2
import pydicom
from skimage.draw import polygon


# --------------------------------------------------------------------------- #
# 1. DICOM + file-id helpers
# --------------------------------------------------------------------------- #
def read_dicom(path):
    """Read a DICOM and return its pixel data as a raw float32 array.

    No VOI LUT and no MONOCHROME1 inversion: all 410 INbreast files are
    MONOCHROME2 with no VOI metadata (audited in Step 0.1), so both were no-ops.
    """
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    return img


def find_sheet_with_cols(xls, cols):
    """Return (sheet_name, df) of the first XLS sheet containing all `cols`."""
    for name in xls.sheet_names:
        df = xls.parse(name)
        have = {c.strip().lower() for c in df.columns.astype(str)}
        if all(rc.lower() in have for rc in cols):
            return name, df
    return None, None


def build_dicom_index(dicom_dir):
    """Map int file_id -> DICOM path for every `<digits>_...` file under a dir."""
    idx = {}
    for fp in dicom_dir.rglob("*"):
        if fp.is_dir():
            continue
        # many INbreast dicoms have no extension; accept everything that starts with digits
        m = re.match(r"^(\d+)_", fp.name)
        if m:
            idx[int(m.group(1))] = fp
    return idx


def build_npy_index(folder):
    """Map int file_id -> .npy path for every patch/image .npy in a folder."""
    idx = {}
    for fp in sorted(folder.glob("*.npy")):
        idx[int(re.match(r"(\d+)", fp.name).group(1))] = fp
    return idx


def build_xml_index(folder):
    """Map int file_id -> XML path for every `<digits>.xml` in a folder."""
    idx = {}
    for fp in sorted(folder.glob("*.xml")):
        if fp.stem.isdigit():
            idx[int(fp.stem)] = fp
    return idx


# --------------------------------------------------------------------------- #
# 2. OsiriX plist mass-ROI parser
# --------------------------------------------------------------------------- #
_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_point_any(p):
    if isinstance(p, (list, tuple)) and len(p) == 2:
        try:
            return float(p[0]), float(p[1])
        except Exception:
            return None
    if isinstance(p, str):
        nums = _num_re.findall(p)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    return None


def _coerce_points(pts):
    out = []
    if pts is None:
        return np.zeros((0, 2), dtype=float)
    if isinstance(pts, str):
        nums = _num_re.findall(pts)
        for i in range(0, len(nums) - 1, 2):
            out.append((float(nums[i]), float(nums[i + 1])))
        return np.array(out, dtype=float) if out else np.zeros((0, 2), dtype=float)
    if isinstance(pts, (list, tuple)):
        for p in pts:
            xy = _parse_point_any(p)
            if xy is not None:
                out.append(xy)
            elif isinstance(p, (list, tuple)) and len(p) > 2:
                try:
                    flat = list(p)
                    for i in range(0, len(flat) - 1, 2):
                        out.append((float(flat[i]), float(flat[i + 1])))
                except Exception:
                    pass
    return np.array(out, dtype=float) if out else np.zeros((0, 2), dtype=float)


def load_mass_rois(xml_path):
    """Return the Mass ROIs of an INbreast OsiriX plist as [{name, points}].

    Each entry is a polygon with >=3 (x, y) points; ROIs not named 'Mass'
    are dropped.
    """
    with open(xml_path, "rb") as f:
        data = plistlib.load(f)
    img0 = data["Images"][0]
    rois = img0.get("ROIs", [])
    out = []
    for roi in rois:
        name = str(roi.get("Name", "")).strip()
        pts_raw = roi.get("Point_px", None)
        if pts_raw is None:
            pts_raw = roi.get("Points", None)
        pts = _coerce_points(pts_raw)
        if pts.shape[0] < 3:
            continue
        out.append({"name": name, "points": pts})
    return [r for r in out if r["name"].strip().lower() == "mass"]


# --------------------------------------------------------------------------- #
# 3. Mask rasterising
# --------------------------------------------------------------------------- #
def polygon_to_mask(points, shape_hw):
    """Rasterise an Nx2 (x, y) polygon into a boolean mask of shape (H, W)."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return np.zeros(shape_hw, dtype=bool)
    rr, cc = polygon(pts[:, 1], pts[:, 0], shape_hw)
    mask = np.zeros(shape_hw, dtype=bool)
    mask[rr, cc] = True
    return mask


def load_labelme_mask(json_path, shape_hw):
    """Fill all polygons in a LabelMe-style JSON into a uint8 mask of (H, W)."""
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    shapes = ann.get("shapes", [])
    for s in shapes:
        pts = np.array(s.get("points", []), dtype=np.float32)
        if pts.size == 0:
            continue
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        cv2.fillPoly(mask, [pts], 1)

    return mask


# --------------------------------------------------------------------------- #
# 4. View + label mappings
# --------------------------------------------------------------------------- #
def is_oblique(filename):
    """True for MLO/ML (oblique) views, which carry a pectoral muscle."""
    u = filename.upper()
    return ("_MLO_" in u) or ("_ML_" in u)


def birads_to_int(b):
    """Convert a BI-RADS value like '4a'/'4b'/'4c' to its integer, else None."""
    if pd.isna(b):
        return None
    s = str(b).strip().lower()
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else None


def birads_to_mass_label(b):
    """Map BI-RADS to a mass label: 2-3 benign, 4-6 malignant, 1/None ignored.

    BI-RADS is suspicion, not pathology ground truth; this is the requested rule.
    """
    bi = birads_to_int(b)
    if bi is None:
        return None
    if bi in [1]:
        return None  # normal / ignore for mass classification
    if bi in [2, 3]:
        return "benign"
    if bi in [4, 5, 6]:
        return "malignant"
    return None


def tissue_binary(acr):
    """Binary tissue-density label: ACR 1-2 -> 0 (non-dense), 3-4 -> 1 (dense)."""
    return 1 if acr >= 3 else 0
