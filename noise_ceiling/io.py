"""Reading and writing results."""

from __future__ import annotations

import os
import pickle

import numpy as np

from .style import LEGACY_NAMES


def ensure_dir(path):
    """Create the directory containing `path` if it does not exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def rename_legacy(results):
    """Map the long estimator names used by older result files to short keys."""
    return {LEGACY_NAMES.get(key, key): value for key, value in results.items()}


def save_results(path, results):
    """Save a dict of arrays as a .npy file."""
    ensure_dir(path)
    np.save(path, results)


def load_results(path):
    """Load a dict of arrays written by `save_results`."""
    return rename_legacy(np.load(path, allow_pickle=True).item())


def save_pickle(path, obj):
    ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
