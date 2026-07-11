"""Project34 --- shared library for the breast-cancer screening replication.

This is the project's first shared Python package. The locked evaluation
*protocol* (seeds, image-level/group-aware splitting, the paper's Subspace
k-NN classifier, pipeline factories, and the repeated-seed / group-CV
evaluators) lives in :mod:`project34.protocol`.

Notebooks import the protocol explicitly, e.g.::

    from project34.protocol import set_seed, SEEDS, SubspaceKNN, image_split, holdout5

The module is a *faithful, byte-equivalent* extraction of the helpers that
previously lived (duplicated) inside the FINAL Step 2 notebooks; importing it
instead of redefining those helpers does not change any numerical result.
"""

__all__ = ["protocol", "data", "preprocess", "patches"]
