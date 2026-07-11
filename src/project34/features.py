"""Feature-extraction helpers for the Step 2 replication notebooks.

Loading a saved patch as a [0, 1] image and the averaged wavelet-scattering
descriptor. Both are deterministic and shared across the Step 2 notebooks. Kept
out of protocol.py, which stays kymatio/torch-free.
"""

from __future__ import annotations

import numpy as np
from kymatio.scattering2d.frontend.numpy_frontend import ScatteringNumPy2D as Scattering2D


def load_patch01(p):
    """Load a saved patch as float32 clipped to [0, 1] (NaN/inf -> 0/1)."""
    a = np.load(p).astype(np.float32); a = np.nan_to_num(a, nan=0., posinf=1., neginf=0.)
    return np.clip(a, 0., 1.)


def averaged_ws_features(patches, J=6, L=5, shape=(224, 224)):
    """Order <= 2 scattering features, spatially averaged: one row per patch.

    ``patches`` is an iterable of [0, 1] arrays; returns an (N, 406) float32
    array for J=6, L=5.
    """
    scat = Scattering2D(J=J, shape=shape, L=L, max_order=2)
    return np.stack([scat(im).mean(axis=(1, 2)).astype(np.float32) for im in patches])
