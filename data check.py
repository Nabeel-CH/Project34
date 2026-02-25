import random
from pathlib import Path
from collections import Counter

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt

# Point this to your INbreast folder root (the one containing ALL-IMGS / ALL-XML / etc.)
INBREAST_ROOT = Path("data/raw/inbreast")
RAW = INBREAST_ROOT / "ALL-IMGS"


# ----------------------------
# Helpers
# ----------------------------
def iter_dicom_paths(img_dir: Path):
    """Return a list of paths that look like DICOMs (prefer *.dcm, else try extensionless files)."""
    dicoms = list(img_dir.rglob("*.dcm"))
    if dicoms:
        return dicoms

    # fallback: some datasets use no extension for DICOMs; try to read only valid ones
    candidates = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix == ""]
    valid = []
    for p in candidates:
        try:
            pydicom.dcmread(str(p), stop_before_pixels=True)
            valid.append(p)
        except Exception:
            pass
    return valid


def load_dicom_for_view(path: Path) -> np.ndarray:
    """Load DICOM pixel data and normalize to [0,1] for visualization."""
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)

    # Multi-frame guard (rare for mammography but harmless)
    if img.ndim == 3:
        img = img[0]

    # apply rescale if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # apply VOI LUT if available (often improves contrast)
    try:
        img = apply_voi_lut(img, ds).astype(np.float32)
    except Exception:
        pass

    # handle inversion
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    # robust normalization for viewing
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)

    return img


def summarize_dataset(inbreast_root: Path, img_dir: Path, dicom_paths: list[Path], n_header_samples: int = 25):
    """Print a summary: file counts, DICOM header stats, and what label/annotation files exist."""
    print("\n==============================")
    print("INBREAST DATASET SUMMARY")
    print("==============================")

    # 1) Count file types under the INbreast root
    all_files = [p for p in inbreast_root.rglob("*") if p.is_file()]
    ext_counts = Counter([p.suffix.lower() if p.suffix else "<no_ext>" for p in all_files])

    print(f"\nRoot: {inbreast_root.resolve()}")
    print(f"Image dir: {img_dir.resolve()}")

    print("\nFile type counts (top):")
    for ext, cnt in ext_counts.most_common(12):
        print(f"  {ext:10s}  {cnt}")

    # 2) DICOM count
    print(f"\nDICOM candidates found in {img_dir.name}: {len(dicom_paths)}")

    # 3) Look for annotation/label files
    label_files = []
    for pat in ("*.xml", "*.csv", "*.xlsx", "*.json", "*.txt"):
        label_files.extend(inbreast_root.rglob(pat))
    label_files = sorted({p.resolve() for p in label_files})

    print("\nPotential label/annotation files found:")
    if not label_files:
        print("  (none found under this root)")
    else:
        for p in label_files[:30]:
            rel = p.relative_to(inbreast_root.resolve())
            print(f"  {rel}")
        if len(label_files) > 30:
            print(f"  ... and {len(label_files) - 30} more")

    # 4) Sample DICOM headers to understand what's inside
    if not dicom_paths:
        print("\nNo DICOMs found to inspect headers.")
        return

    header_sample = random.sample(dicom_paths, k=min(n_header_samples, len(dicom_paths)))

    modality = Counter()
    viewpos = Counter()
    laterality = Counter()
    photo = Counter()
    shapes = Counter()
    frames_counter = Counter()

    # check for any breast-density-ish tags via keywords (often absent in INbreast DICOMs)
    density_tag_hits = 0

    for p in header_sample:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)

        modality[str(getattr(ds, "Modality", None))] += 1
        viewpos[str(getattr(ds, "ViewPosition", None))] += 1
        laterality[str(getattr(ds, "ImageLaterality", None))] += 1
        photo[str(getattr(ds, "PhotometricInterpretation", None))] += 1

        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        shapes[str((rows, cols))] += 1

        nframes = getattr(ds, "NumberOfFrames", None)
        frames_counter[str(nframes if nframes is not None else 1)] += 1

        # keyword-based scan for density/BIRADS tags (not guaranteed)
        for kw in ("Breast", "Composition", "Density", "BIRADS", "BI-RADS"):
            # scan DICOM elements by keyword
            for elem in ds:
                if elem.keyword and kw.lower() in elem.keyword.lower():
                    density_tag_hits += 1
                    break

    print("\nDICOM header sample stats (from a random subset):")
    print("  Modality:", dict(modality))
    print("  ViewPosition:", dict(viewpos))
    print("  ImageLaterality:", dict(laterality))
    print("  PhotometricInterpretation:", dict(photo))
    print("  Common shapes (Rows, Cols):", dict(shapes.most_common(5)))
    print("  NumberOfFrames (mostly 1 in mammography):", dict(frames_counter))

    if density_tag_hits == 0:
        print("\nLabel note:")
        print("  I didn't see obvious density/BIRADS-related tags in the sampled DICOM headers.")
        print("  In INbreast, labels/annotations are typically stored in separate XML/CSV files.")
    else:
        print("\nLabel note:")
        print("  Some sampled headers contained keywords related to density/BIRADS-like fields.")
        print("  (Still, INbreast often uses separate annotation files—check the XML/CSV list above.)")

    # 5) Show one example header (lightweight)
    example = pydicom.dcmread(str(header_sample[0]), stop_before_pixels=True)
    print("\nExample DICOM (header fields):")
    fields = [
        "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
        "Modality", "ViewPosition", "ImageLaterality", "Rows", "Columns",
        "PixelSpacing", "BitsStored", "RescaleSlope", "RescaleIntercept",
        "PhotometricInterpretation"
    ]
    for f in fields:
        print(f"  {f:22s}: {getattr(example, f, None)}")


# ----------------------------
# Main
# ----------------------------
dicoms = iter_dicom_paths(RAW)

# Print summary BEFORE showing any images
summarize_dataset(INBREAST_ROOT, RAW, dicoms, n_header_samples=25)

# Visual sanity check: show 5 random images
print("\nShowing 5 random mammograms for sanity check...")
sample = random.sample(dicoms, k=min(5, len(dicoms)))

plt.figure(figsize=(12, 6))
for i, p in enumerate(sample, 1):
    img = load_dicom_for_view(p)
    plt.subplot(1, len(sample), i)
    plt.imshow(img, cmap="gray")
    plt.title(p.name[:12])
    plt.axis("off")
plt.tight_layout()
plt.show()
