"""Unit tests for project34.protocol --- the shared evaluation protocol.

Runnable either with pytest (``pytest tests/test_protocol.py``) or standalone
(``python tests/test_protocol.py``) so it does not require pytest to be installed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src layout, so `import project34` works standalone

import numpy as np
import pandas as pd
import torch

from project34.protocol import (
    SEEDS, DIAG_SEED, set_seed, SubspaceKNN, sknn_pipe,
    image_split, holdout5, cv_seed34,
)


def _toy(n_img=24, per=3, dim=20, seed=0):
    """A small grouped toy dataset: n_img images, `per` patches each, mild class signal."""
    rng = np.random.RandomState(seed)
    groups = np.repeat(np.arange(n_img), per)
    y_img = np.array([0, 1] * (n_img // 2))           # balanced per image
    y = np.repeat(y_img, per)
    X = (rng.rand(n_img * per, dim) + y[:, None] * 0.4).astype(np.float32)
    strat = pd.Series(y_img, index=np.arange(n_img))  # per-image strat label
    return X, y, groups, strat


def test_image_split_no_leakage():
    """No image's patches may straddle the train/test boundary, and it must partition."""
    X, y, groups, strat = _toy()
    for seed in SEEDS:
        tr, te = image_split(groups, strat, seed)
        assert set(groups[tr]).isdisjoint(set(groups[te])), f"group leakage at seed {seed}"
        assert set(tr.tolist()).isdisjoint(set(te.tolist()))
        assert len(tr) + len(te) == len(y)            # exhaustive partition


def test_set_seed_determinism():
    """set_seed makes NumPy + torch draws reproducible and pins cudnn determinism."""
    set_seed(DIAG_SEED); a = np.random.rand(5); t = torch.rand(5)
    set_seed(DIAG_SEED); b = np.random.rand(5); u = torch.rand(5)
    assert np.array_equal(a, b)
    assert torch.equal(t, u)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_holdout5_repeatable():
    """holdout5 is fully deterministic: two calls give identical numbers."""
    X, y, groups, strat = _toy()
    r1 = holdout5(X, y, groups, strat, sknn_pipe)
    r2 = holdout5(X, y, groups, strat, sknn_pipe)
    for k in r1:
        assert np.allclose(r1[k], r2[k], rtol=0, atol=0, equal_nan=True), f"non-deterministic key {k}"


def test_holdout5_contract():
    X, y, groups, strat = _toy()
    r = holdout5(X, y, groups, strat, sknn_pipe)
    assert set(r.keys()) == {"auroc", "auroc_sd", "acc", "acc_sd", "f1"}


def test_cv_seed34_contract():
    X, y, groups, strat = _toy()
    out = cv_seed34(X, y, groups, strat, sknn_pipe)
    assert isinstance(out, tuple) and len(out) == 2
    assert all(np.isscalar(v) or np.ndim(v) == 0 for v in out)


def test_sknn_pipe_fits_train_only():
    """The scaler inside sknn_pipe must fit on the TRAIN fold only, not the full data."""
    X, y, groups, strat = _toy()
    tr, te = image_split(groups, strat, DIAG_SEED)
    pipe = sknn_pipe(DIAG_SEED).fit(X[tr], y[tr])
    sc = pipe.named_steps["sc"]
    assert np.allclose(sc.mean_, X[tr].mean(axis=0)), "scaler mean != train mean"
    assert not np.allclose(sc.mean_, X.mean(axis=0)), "scaler appears fit on full data (leakage)"


def test_subspaceknn_seeded_reproducible():
    """SubspaceKNN with a fixed random_state is reproducible; different seeds differ."""
    X, y, _, _ = _toy()
    p1 = SubspaceKNN(80, min(190, X.shape[1]), 1, 34).fit(X, y).predict_proba(X)
    p2 = SubspaceKNN(80, min(190, X.shape[1]), 1, 34).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print("ALL %d TESTS PASSED" % len(fns))
