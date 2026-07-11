"""Shared evaluation *protocol* for the Project34 Step 2 experiments.

This module is the single source of truth for the locked evaluation protocol
that the FINAL Step 2 notebooks previously duplicated inline:

* reproducibility   -- :data:`SEEDS`, :data:`DIAG_SEED`, :func:`set_seed`
* project location  -- :func:`find_project_root`, :func:`standard_paths`
* classifiers       -- :class:`SubspaceKNN`, :class:`AdaptivePCAKNN`
* pipeline factories-- :func:`sknn_pipe`, :func:`logreg_pipe`, :func:`svm_pipe`,
                       :func:`pca32_logreg`
* splitting         -- :func:`image_split`  (the leakage-free, group-aware split)
* evaluators        -- :func:`holdout5` (repeated-seed hold-out, AUROC-led),
                       :func:`cv_seed34` (seed-34 group-aware CV, paper-comparison)
* reporting helpers -- :func:`show`, :func:`record`, :func:`build_results_table`

Every function/class here is a *faithful, byte-equivalent* extraction of the
implementation that previously lived in the notebooks (primarily FINAL Step 2.3
and 2.5), so importing from this module instead of redefining inline produces
numerically identical results.

Import-light by design: only NumPy, pandas, scikit-learn, and torch (the latter
solely for :func:`set_seed`). It deliberately does **not** import Kymatio,
torchvision, OpenCV, pydicom, or any feature-extraction / data-loading code --
those belong in future ``data.py`` / ``features.py`` modules or stay in the
notebooks.

Note on seeds: the canonical :data:`SEEDS` order is ``[1, 7, 34, 42, 99]``, and all
FINAL Step 2 notebooks (2.1--2.7) import it from here, so the seed set is defined in
exactly one place. (Step 2.4 previously kept a local ``[34, 1, 7, 42, 99]`` order to
preserve its per-seed CSV row order, but was later realigned to import the canonical
``SEEDS``; ``99`` replaced an earlier ``1234`` as the fifth seed across the project.)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

__all__ = [
    "SEEDS", "DIAG_SEED", "set_seed", "find_project_root", "standard_paths",
    "SubspaceKNN", "AdaptivePCAKNN",
    "sknn_pipe", "logreg_pipe", "svm_pipe", "pca32_logreg",
    "image_split", "holdout5", "cv_seed34",
    "show", "record", "build_results_table",
]

# --------------------------------------------------------------------------- #
# 1. Reproducibility / seeds
# --------------------------------------------------------------------------- #
SEEDS = [1, 7, 34, 42, 99]     # canonical repeated-hold-out seeds
DIAG_SEED = 34                 # canonical diagnostic / group-CV seed


def set_seed(seed):
    """Seed Python, NumPy and torch for deterministic runs.

    Byte-equivalent to the ``set_seed`` previously defined in FINAL Step 2.3/2.4/2.5.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_project_root():
    """Locate the Project34 root whether the kernel starts at the root or in Step 2."""
    cur = Path.cwd().resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "data").exists() and (cand / "Step 2 - experiments NOTEBOOKS").exists():
            return cand
    raise FileNotFoundError("project root not found")


def standard_paths(root=None):
    """Canonical input locations (optional convenience).

    Notebooks may still define their own per-notebook ``OUT`` directories; this
    only centralises the stable *input* manifests / preprocessed-image folder.
    """
    root = Path(root) if root is not None else find_project_root()
    return dict(
        ROOT=root,
        TISSUE_INDEX=root / "data" / "outputs" / "background_patches" / "patches_index.csv",
        MASS_INDEX=root / "data" / "outputs" / "masses" / "mass_index.csv",
        PREPROC_FINAL_DIR=root / "data" / "outputs" / "preprocessed" / "final",
    )


# --------------------------------------------------------------------------- #
# 2. Classifiers (paper-style)
# --------------------------------------------------------------------------- #
class SubspaceKNN(ClassifierMixin, BaseEstimator):
    """Paper's classifier: 80 learners, 1-NN on random 190-d subspaces, soft vote."""
    _estimator_type = "classifier"

    def __init__(self, n_estimators=80, subspace_dim=190, n_neighbors=1, random_state=34):
        self.n_estimators = n_estimators
        self.subspace_dim = subspace_dim
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        self.classes_ = np.unique(y); self.models_ = []; self.subspaces_ = []
        rng = np.random.RandomState(self.random_state); nf = X.shape[1]; d = min(self.subspace_dim, nf)
        for _ in range(self.n_estimators):
            c = rng.choice(nf, size=d, replace=False)
            self.models_.append(KNeighborsClassifier(self.n_neighbors).fit(X[:, c], y)); self.subspaces_.append(c)
        return self

    def predict_proba(self, X):
        X = np.asarray(X); pr = np.zeros((X.shape[0], len(self.classes_)))
        for m, c in zip(self.models_, self.subspaces_):
            p = m.predict_proba(X[:, c]); a = np.zeros_like(pr)
            for j, cl in enumerate(m.classes_):
                a[:, np.where(self.classes_ == cl)[0][0]] = p[:, j]
            pr += a
        return pr / len(self.models_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class AdaptivePCAKNN(ClassifierMixin, BaseEstimator):
    """Scale -> PCA (component count capped to the fold) -> Subspace k-NN. Mirrors old Step 2.6."""
    _estimator_type = "classifier"

    def __init__(self, n_components=200, random_state=34):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X); nc = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        self.sc_ = StandardScaler().fit(X); Xp = self.sc_.transform(X)
        self.pca_ = PCA(n_components=nc, random_state=self.random_state).fit(Xp)
        self.clf_ = SubspaceKNN(80, min(190, nc), 1, self.random_state).fit(self.pca_.transform(Xp), y)
        self.classes_ = self.clf_.classes_; return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.pca_.transform(self.sc_.transform(np.asarray(X))))

    def predict(self, X):
        return self.clf_.predict(self.pca_.transform(self.sc_.transform(np.asarray(X))))


# --------------------------------------------------------------------------- #
# 3. Pipeline factories (scaler/PCA fit train-fold only inside the Pipeline)
# --------------------------------------------------------------------------- #
def sknn_pipe(seed=34):
    return Pipeline([("sc", StandardScaler()), ("clf", SubspaceKNN(80, 190, 1, seed))])


def logreg_pipe(seed=34):
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])


def svm_pipe(seed=34):
    return Pipeline([("sc", StandardScaler()),
                     ("clf", SVC(kernel="rbf", C=4, gamma="scale", probability=True, random_state=seed))])


def pca32_logreg(seed=34):
    return Pipeline([("sc", StandardScaler()), ("pca", PCA(32, random_state=seed)),
                     ("clf", LogisticRegression(max_iter=3000))])


# --------------------------------------------------------------------------- #
# 4. Splitting + evaluators
# --------------------------------------------------------------------------- #
def image_split(groups, strat_by_img, seed, test_size=0.2):
    """Leakage-free image-level (group-aware) split.

    ``groups`` is a per-sample array of image ids (``file_id``); ``strat_by_img``
    is a per-image Series (indexed by ``file_id``) giving the stratification
    label. Returns ``(train_idx, test_idx)`` integer index arrays. No image's
    patches can straddle the train/test boundary.
    """
    meta = strat_by_img.reset_index(); meta.columns = ["file_id", "strat"]
    _, te = train_test_split(meta["file_id"], test_size=test_size, stratify=meta["strat"],
                             random_state=seed, shuffle=True)
    is_test = pd.Series(groups).isin(set(te)).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]


def holdout5(X, y, groups, strat, est_factory, seeds=SEEDS):
    """Repeated image-level hold-out over ``seeds``; AUROC-led summary.

    ``est_factory(seed) -> fresh estimator`` is refit cleanly on each seed's
    train fold (never reusing fitted state). Returns
    ``dict(auroc, auroc_sd, acc, acc_sd, f1)``.
    """
    au, ac, f1 = [], [], []
    for seed in seeds:
        tr, te = image_split(groups, strat, seed); est = est_factory(seed); est.fit(X[tr], y[tr])
        yp = est.predict(X[te]); pr = est.predict_proba(X[te])[:, 1]
        au.append(roc_auc_score(y[te], pr) if len(np.unique(y[te])) > 1 else np.nan)
        ac.append(accuracy_score(y[te], yp)); f1.append(f1_score(y[te], yp, zero_division=0))
    return dict(auroc=np.nanmean(au), auroc_sd=np.nanstd(au, ddof=1), acc=np.mean(ac), acc_sd=np.std(ac, ddof=1), f1=np.mean(f1))


def cv_seed34(X, y, groups, strat, est_factory, n_splits=10):
    """Seed-34 group-aware CV accuracy on the seed-34 TRAIN split (paper-comparison metric).

    Returns ``(mean, std)`` of the per-fold accuracy.
    """
    tr, _ = image_split(groups, strat, 34)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=34)
    s = cross_val_score(est_factory(34), X[tr], y[tr], cv=cv, groups=groups[tr], scoring="accuracy", n_jobs=1)
    return s.mean(), s.std()


# --------------------------------------------------------------------------- #
# 5. Reporting helpers
# --------------------------------------------------------------------------- #
def show(tag, res, extra="", width=44):
    """Pretty-print a holdout5-style result dict. ``width`` is the tag column width."""
    print("  %-*s AUROC %.3f±%.3f | acc %.3f±%.3f | F1 %.3f %s" % (
        width, tag, res["auroc"], res["auroc_sd"], res["acc"], res["acc_sd"], res["f1"], extra))


def record(rows, method, task, res, role=None, reason=None, **extra):
    """Generic results-row collector: append a row dict to ``rows`` and return it.

    Provided for new code. The FINAL 2.3 / 2.5 notebooks keep their own bespoke
    ``record`` (incompatible column schemas -- role/reason/cv vs
    provenance/status/reason/source), so this generic form is intentionally not
    adopted there, to keep their consolidated-table CSV columns byte-stable.
    """
    row = dict(method=method, task=task,
               auroc=res.get("auroc"), auroc_sd=res.get("auroc_sd"),
               test_acc=res.get("acc"), test_acc_sd=res.get("acc_sd"), f1=res.get("f1"))
    if role is not None:
        row["role"] = role
    if reason is not None:
        row["reason"] = reason
    row.update(extra)
    rows.append(row)
    return row


def build_results_table(rows, save_path=None):
    """Build a DataFrame from collected ``rows`` and optionally save it to CSV."""
    df = pd.DataFrame(rows)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df
