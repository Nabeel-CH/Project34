#!/usr/bin/env python3
"""Resume FINAL Step 2.7 sections 3–6 from cached section 1–2 outputs."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kymatio.scattering2d.frontend.numpy_frontend import ScatteringNumPy2D as Scattering2D
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from project34.protocol import SEEDS  # noqa: E402

DATA = PROJECT / "data"
OUT = DATA / "outputs" / "tissue_preprocessing_methodology_checks"
IMG_SIZE = 224
LABEL_MAP = {"fatty": 0, "fibroglandular": 1}
VARIANTS = ["current_full_preprocessed", "raw_minmax", "raw_percentile", "masked_no_clahe"]
N_REJECT_PER_LABEL = 60

_final_cache: dict = {}
_mask_cache: dict = {}


def resolve_project_path(path_like) -> Path:
    s = str(path_like).replace("\\", "/")
    p = Path(s)
    if p.is_absolute():
        return p
    marker = "data/"
    idx = s.find(marker)
    if idx >= 0:
        return PROJECT / s[idx:]
    return (PROJECT / p).resolve()


def resize_to_224(patch: np.ndarray) -> np.ndarray:
    return cv2.resize(patch.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_npy_cached(path_like, cache) -> np.ndarray:
    path = resolve_project_path(path_like)
    key = str(path)
    if key not in cache:
        cache[key] = np.load(path).astype(np.float32)
    return cache[key]


def radial_power_spectrum(img: np.ndarray, n_bins=40) -> np.ndarray:
    arr = img.astype(np.float32) - float(np.mean(img))
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    h, w = power.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    bins = np.linspace(0, rr.max(), n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = (rr >= bins[i]) & (rr < bins[i + 1])
        out[i] = power[m].mean() if np.any(m) else 0.0
    return np.log1p(out).astype(np.float32)


def patch_stats(images: np.ndarray, features: np.ndarray | None = None) -> pd.DataFrame:
    flat = images.reshape(len(images), -1)
    out = pd.DataFrame({
        "mean": flat.mean(axis=1),
        "std": flat.std(axis=1),
        "p05": np.percentile(flat, 5, axis=1),
        "p50": np.percentile(flat, 50, axis=1),
        "p95": np.percentile(flat, 95, axis=1),
    })
    spectra = np.stack([radial_power_spectrum(im) for im in images])
    out["fft_low"] = spectra[:, :8].mean(axis=1)
    out["fft_mid"] = spectra[:, 8:24].mean(axis=1)
    out["fft_high"] = spectra[:, 24:].mean(axis=1)
    if features is not None:
        out["ws_energy"] = np.linalg.norm(features, axis=1)
    return out


def summarize_mean_diff(df, group_col, value_cols):
    groups = list(df[group_col].dropna().unique())
    if len(groups) != 2:
        return pd.DataFrame()
    a, b = groups
    A, B = df[df[group_col] == a], df[df[group_col] == b]
    rows = []
    for c in value_cols:
        rows.append({
            "metric": c,
            f"mean_{a}": A[c].mean(),
            f"mean_{b}": B[c].mean(),
            "abs_diff": abs(A[c].mean() - B[c].mean()),
            "wasserstein": wasserstein_distance(A[c].dropna(), B[c].dropna()),
        })
    return pd.DataFrame(rows).sort_values("wasserstein", ascending=False)


def purity_against_labels(y_true, cluster_labels) -> float:
    score = 0
    for c in np.unique(cluster_labels):
        _, counts = np.unique(y_true[cluster_labels == c], return_counts=True)
        score += counts.max()
    return score / len(y_true)


def sample_rejected_candidates(manifest, final_to_source, n_per_label=N_REJECT_PER_LABEL, seed=34):
    rng = np.random.default_rng(seed)
    records = []
    by_label = {"fatty": 0, "fibroglandular": 0}
    rows = manifest.sample(frac=1, random_state=seed).reset_index(drop=True)
    trial = 0
    while min(by_label.values()) < n_per_label and trial < 120000:
        trial += 1
        row = rows.iloc[int(rng.integers(0, len(rows)))]
        label = row.label
        if by_label[label] >= n_per_label:
            continue
        source = final_to_source[str(resolve_project_path(row.source_final_npy))]
        img = load_npy_cached(row.source_final_npy, _final_cache)
        breast = load_npy_cached(source["breast_mask"], _mask_cache) > 0
        pect = load_npy_cached(source["pect_mask"], _mask_cache) > 0
        allowed = breast & (~pect)
        s = int(row.orig_size)
        if img.shape[0] <= s or img.shape[1] <= s:
            continue
        y0 = int(rng.integers(0, img.shape[0] - s))
        x0 = int(rng.integers(0, img.shape[1] - s))
        y1, x1 = y0 + s, x0 + s
        patch_allowed = allowed[y0:y1, x0:x1]
        if patch_allowed.mean() < 0.90:
            continue
        vals = img[allowed]
        if vals.size < 10:
            continue
        thr = float(row.threshold) if not pd.isna(row.threshold) else float(np.percentile(vals, 50))
        patch = img[y0:y1, x0:x1].copy()
        if label == "fatty":
            target = (patch <= max(0.0, thr - 0.03)) & patch_allowed
        else:
            target = (patch >= min(1.0, thr + 0.03)) & patch_allowed
        target_frac = target.sum() / max(1, patch_allowed.sum())
        if target_frac >= 0.65:
            continue
        patch[~patch_allowed] = 0.0
        patch224 = resize_to_224(patch)
        npy = OUT / "rejected_candidate_patches" / label / f"{int(row.file_id)}_{label}_reject_{by_label[label]:03d}.npy"
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, patch224.astype(np.float32))
        records.append({
            "file_id": row.file_id,
            "label": label,
            "acr": row.acr,
            "orig_size": s,
            "y0": y0, "y1": y1 - 1, "x0": x0, "x1": x1 - 1,
            "allowed_frac": float(patch_allowed.mean()),
            "target_frac": float(target_frac),
            "threshold": thr,
            "patch_npy": str(npy.relative_to(PROJECT)),
        })
        by_label[label] += 1
    print("Rejected candidates:", by_label, "trials:", trial)
    return pd.DataFrame(records)


def load_cached_state():
    manifest = pd.read_csv(OUT / "tissue_methodology_manifest.csv")
    manifest["patch_npy"] = manifest["patch_npy"].map(lambda p: str(resolve_project_path(p)))
    manifest["source_final_npy"] = manifest["source_final_npy"].map(lambda p: str(resolve_project_path(p)))
    y = manifest["class_id"].to_numpy(np.int64)

    preproc = pd.read_csv(DATA / "outputs" / "preprocessed" / "preproc_index.csv")
    preproc["final_resolved"] = preproc["final_npy"].map(lambda p: str(resolve_project_path(p)))
    final_to_source = {
        row.final_resolved: {
            "dicom": str(resolve_project_path(row.dicom)),
            "breast_mask": str(resolve_project_path(row.breast_mask_npy)),
            "pect_mask": str(resolve_project_path(row.pect_mask_npy)),
        }
        for row in preproc.itertuples(index=False)
    }

    patch_arrays = {v: np.load(OUT / f"tissue_{v}_patches_224.npy") for v in VARIANTS}
    ws_by_variant = {v: np.load(OUT / f"tissue_{v}_ws_J6_L5_avg_features.npy") for v in VARIANTS}
    ws_summary = pd.read_csv(OUT / "tissue_ws_pixel_source_summary.csv")
    return manifest, y, final_to_source, patch_arrays, ws_by_variant, ws_summary


def run_section_3(manifest, final_to_source, patch_arrays, ws_by_variant):
    rejected_manifest_path = OUT / "rejected_candidate_manifest.csv"
    if rejected_manifest_path.exists():
        rejected_manifest = pd.read_csv(rejected_manifest_path)
        print("Loaded rejected manifest:", rejected_manifest.shape)
    else:
        rejected_manifest = sample_rejected_candidates(manifest, final_to_source)
        rejected_manifest.to_csv(rejected_manifest_path, index=False)

    rejected_patches_path = OUT / "rejected_candidate_patches_224.npy"
    if rejected_patches_path.exists():
        rejected_patches = np.load(rejected_patches_path)
    else:
        rejected_patches = np.stack([np.load(PROJECT / p).astype(np.float32) for p in rejected_manifest["patch_npy"]])
        np.save(rejected_patches_path, rejected_patches)

    rejected_ws_path = OUT / "rejected_candidate_ws_J6_L5_avg_features.npy"
    if rejected_ws_path.exists():
        rejected_ws = np.load(rejected_ws_path)
    else:
        scattering = Scattering2D(J=6, shape=(IMG_SIZE, IMG_SIZE), L=5, max_order=2)
        rejected_ws = []
        for i, img in enumerate(rejected_patches, start=1):
            S = scattering(img.astype(np.float32))
            rejected_ws.append(S.reshape(S.shape[0], -1).mean(axis=1))
            if i % 25 == 0 or i == len(rejected_patches):
                print(f"rejected WS {i}/{len(rejected_patches)}")
        rejected_ws = np.stack(rejected_ws).astype(np.float32)
        np.save(rejected_ws_path, rejected_ws)

    accepted = patch_stats(patch_arrays["current_full_preprocessed"], ws_by_variant["current_full_preprocessed"])
    accepted["selection"] = "accepted"
    accepted["label"] = manifest["label"].to_numpy()
    accepted["file_id"] = manifest["file_id"].astype(str).to_numpy()
    accepted["x_center_norm"] = ((manifest["x0"] + manifest["x1"]) / 2) / manifest.groupby("file_id")["x1"].transform("max").replace(0, np.nan)
    accepted["y_center_norm"] = ((manifest["y0"] + manifest["y1"]) / 2) / manifest.groupby("file_id")["y1"].transform("max").replace(0, np.nan)
    accepted["orig_size"] = manifest["orig_size"].to_numpy()
    accepted["target_frac"] = manifest["target_frac"].to_numpy()
    accepted["allowed_frac"] = manifest["allowed_frac"].to_numpy()

    rejected = patch_stats(rejected_patches, rejected_ws)
    rejected["selection"] = "rejected_candidate"
    rejected["label"] = rejected_manifest["label"].to_numpy()
    rejected["file_id"] = rejected_manifest["file_id"].astype(str).to_numpy()
    rejected["x_center_norm"] = ((rejected_manifest["x0"] + rejected_manifest["x1"]) / 2) / rejected_manifest.groupby("file_id")["x1"].transform("max").replace(0, np.nan)
    rejected["y_center_norm"] = ((rejected_manifest["y0"] + rejected_manifest["y1"]) / 2) / rejected_manifest.groupby("file_id")["y1"].transform("max").replace(0, np.nan)
    rejected["orig_size"] = rejected_manifest["orig_size"].to_numpy()
    rejected["target_frac"] = rejected_manifest["target_frac"].to_numpy()
    rejected["allowed_frac"] = rejected_manifest["allowed_frac"].to_numpy()

    audit_stats = pd.concat([accepted, rejected], ignore_index=True)
    audit_stats.to_csv(OUT / "otsu_selection_bias_patch_stats.csv", index=False)
    bias_metrics = summarize_mean_diff(
        audit_stats, "selection",
        ["mean", "std", "p05", "p50", "p95", "fft_low", "fft_mid", "fft_high", "ws_energy",
         "x_center_norm", "y_center_norm", "orig_size", "target_frac"],
    )
    bias_metrics.to_csv(OUT / "otsu_selection_bias_metric_differences.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.ravel(), ["mean", "std", "fft_low", "fft_mid", "fft_high", "ws_energy"]):
        for selection, grp in audit_stats.groupby("selection"):
            ax.hist(grp[metric], bins=30, alpha=0.55, density=True, label=selection)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    plt.suptitle("Otsu-selection audit: accepted vs approximate rejected candidates")
    plt.tight_layout()
    plt.savefig(OUT / "otsu_selection_bias_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Section 3 done.")


def run_section_4(y, ws_by_variant):
    cluster_rows = []
    for variant, Xws in ws_by_variant.items():
        Z = StandardScaler().fit_transform(Xws)
        for seed in SEEDS:
            km = KMeans(n_clusters=2, random_state=seed, n_init=25)
            clusters = km.fit_predict(Z)
            cluster_rows.append({
                "variant": variant, "seed": seed,
                "purity": purity_against_labels(y, clusters),
                "adjusted_rand": adjusted_rand_score(y, clusters),
                "silhouette": silhouette_score(Z, clusters),
            })
    cluster_per_seed = pd.DataFrame(cluster_rows)
    cluster_summary = cluster_per_seed.groupby("variant").agg(
        purity=("purity", "mean"), purity_sd=("purity", "std"),
        adjusted_rand=("adjusted_rand", "mean"), adjusted_rand_sd=("adjusted_rand", "std"),
        silhouette=("silhouette", "mean"), silhouette_sd=("silhouette", "std"),
    ).reset_index().sort_values("adjusted_rand", ascending=False)
    cluster_per_seed.to_csv(OUT / "unsupervised_ws_cluster_per_seed.csv", index=False)
    cluster_summary.to_csv(OUT / "unsupervised_ws_cluster_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot = cluster_summary.sort_values("adjusted_rand")
    ax.barh(plot["variant"], plot["adjusted_rand"], xerr=plot["adjusted_rand_sd"].fillna(0), alpha=0.85)
    ax.set_xlabel("Adjusted Rand index vs ACR-derived patch labels")
    ax.set_title("Unsupervised WS two-cluster sanity check")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "unsupervised_ws_cluster_ari.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Section 4 done.")


def run_section_5(y, patch_arrays):
    spectra = {}
    freq_rows = []
    for variant, imgs in patch_arrays.items():
        spec_path = OUT / f"tissue_{variant}_radial_fft_spectra.npy"
        if spec_path.exists():
            spec = np.load(spec_path)
        else:
            spec = np.stack([radial_power_spectrum(im) for im in imgs])
            np.save(spec_path, spec)
        spectra[variant] = spec
        for label_name, class_id in LABEL_MAP.items():
            m = y == class_id
            for bin_idx, val in enumerate(spec[m].mean(axis=0)):
                freq_rows.append({"variant": variant, "label": label_name, "freq_bin": bin_idx, "log_power": val})

    pd.DataFrame(freq_rows).to_csv(OUT / "tissue_radial_fft_mean_spectra_long.csv", index=False)
    freq_sep_rows = []
    for variant, spec in spectra.items():
        fatty = spec[y == LABEL_MAP["fatty"]]
        fibro = spec[y == LABEL_MAP["fibroglandular"]]
        diff = fibro.mean(axis=0) - fatty.mean(axis=0)
        freq_sep_rows.append({
            "variant": variant,
            "mean_abs_spectral_gap": float(np.mean(np.abs(diff))),
            "low_band_gap": float(np.mean(np.abs(diff[:8]))),
            "mid_band_gap": float(np.mean(np.abs(diff[8:24]))),
            "high_band_gap": float(np.mean(np.abs(diff[24:]))),
            "wasserstein_all_bins": float(np.mean([wasserstein_distance(fatty[:, i], fibro[:, i]) for i in range(spec.shape[1])])),
        })
    freq_summary = pd.DataFrame(freq_sep_rows).sort_values("mean_abs_spectral_gap", ascending=False)
    freq_summary.to_csv(OUT / "tissue_frequency_separation_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, variant in zip(axes.ravel(), VARIANTS):
        spec = spectra[variant]
        for label_name, class_id, color in [("fatty", 0, "tab:blue"), ("fibroglandular", 1, "tab:orange")]:
            m = y == class_id
            mean, sd = spec[m].mean(axis=0), spec[m].std(axis=0)
            x = np.arange(len(mean))
            ax.plot(x, mean, label=label_name, color=color)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15)
        ax.set_title(variant)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("radial frequency bin")
    axes[0, 0].set_ylabel("log radial power")
    axes[0, 0].legend(fontsize=8)
    plt.suptitle("Fatty vs fibroglandular radial Fourier spectra by pixel source")
    plt.tight_layout()
    plt.savefig(OUT / "tissue_radial_fft_by_variant.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Section 5 done.")


def run_section_6(ws_summary, bias_metrics, cluster_summary, freq_summary):
    current_ws = ws_summary.loc[ws_summary["variant"] == "current_full_preprocessed"].iloc[0]
    best_ws = ws_summary.sort_values("auroc", ascending=False).iloc[0]
    masked_ws = ws_summary.loc[ws_summary["variant"] == "masked_no_clahe"].iloc[0]
    largest_bias = bias_metrics.iloc[0]
    best_cluster = cluster_summary.sort_values("adjusted_rand", ascending=False).iloc[0]
    best_freq = freq_summary.sort_values("mean_abs_spectral_gap", ascending=False).iloc[0]

    recommendations = [
        {
            "check": "1. Raw-coordinate tissue patch extraction",
            "headline": f"Best WS AUROC was {best_ws.variant} ({best_ws.auroc:.3f}); current preprocessing was {current_ws.auroc:.3f}.",
            "interpretation": "Raw-coordinate extraction is useful as a sensitivity test, but it does not automatically improve tissue WS.",
            "recommendation": "Keep as methodology sensitivity / appendix unless it materially beats the current row.",
        },
        {
            "check": "2. Otsu-selection bias audit",
            "headline": f"Largest accepted-vs-rejected shift was {largest_bias.metric} (Wasserstein {largest_bias.wasserstein:.3f}).",
            "interpretation": "The Otsu purity rule is not neutral if accepted and rejected candidates differ in intensity/frequency/WS-energy statistics.",
            "recommendation": "Keep this audit if the shifts are visible; it directly answers the supervisor's sampling-bias concern.",
        },
        {
            "check": "3. No-CLAHE tissue sensitivity",
            "headline": f"masked/no-CLAHE WS AUROC {masked_ws.auroc:.3f} vs current {current_ws.auroc:.3f}.",
            "interpretation": "No-CLAHE should only change the report narrative if it consistently improves or moves tissue closer to Razali.",
            "recommendation": "Use as a negative/control result if it is similar or worse; do not replace the main tissue pipeline on this alone.",
        },
        {
            "check": "4. Unsupervised WS tissue sanity check",
            "headline": f"Best ARI was {best_cluster.adjusted_rand:.3f} for {best_cluster.variant}; purity {best_cluster.purity:.3f}.",
            "interpretation": "This checks whether WS features naturally align with ACR-derived tissue labels without supervised fitting.",
            "recommendation": "Appendix/future-work unless ARI is clearly strong; do not treat it as a classifier replacement.",
        },
        {
            "check": "5. Frequency-domain tissue diagnostic",
            "headline": f"Largest fatty/fibro spectral gap was {best_freq.variant} (mean abs gap {best_freq.mean_abs_spectral_gap:.3f}).",
            "interpretation": "This is the most direct answer to the frequency-content concern, independent of classifier choice.",
            "recommendation": "Keep as a qualitative/diagnostic figure if it shows preprocessing changes spectral separation.",
        },
    ]
    pd.DataFrame(recommendations).to_csv(OUT / "tissue_methodology_recommendations.csv", index=False)
    with open(OUT / "tissue_methodology_summary.json", "w") as f:
        json.dump({
            "best_ws_variant": best_ws.to_dict(),
            "current_ws_variant": current_ws.to_dict(),
            "masked_no_clahe_variant": masked_ws.to_dict(),
            "largest_otsu_bias_metric": largest_bias.to_dict(),
            "best_unsupervised_cluster_variant": best_cluster.to_dict(),
            "best_frequency_variant": best_freq.to_dict(),
            "n_reject_per_label": N_REJECT_PER_LABEL,
        }, f, indent=2)
    print("Section 6 done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, choices=[3, 4, 5, 6], required=True)
    args = parser.parse_args()

    manifest, y, final_to_source, patch_arrays, ws_by_variant, ws_summary = load_cached_state()

    if args.section == 3:
        run_section_3(manifest, final_to_source, patch_arrays, ws_by_variant)
    elif args.section == 4:
        run_section_4(y, ws_by_variant)
    elif args.section == 5:
        run_section_5(y, patch_arrays)
    elif args.section == 6:
        bias = pd.read_csv(OUT / "otsu_selection_bias_metric_differences.csv")
        cluster = pd.read_csv(OUT / "unsupervised_ws_cluster_summary.csv")
        freq = pd.read_csv(OUT / "tissue_frequency_separation_summary.csv")
        run_section_6(ws_summary, bias, cluster, freq)


if __name__ == "__main__":
    main()
