"""Loading the three recorded datasets.

Every loader returns an array of shape (n_trials, n_units, n_samples). Trials
are separate subjects for Sherlock and repeated presentations for THINGS and
THINGS-EEG. The last axis is the one correlations are computed across: time
points for the continuous Sherlock stimulus, stimuli for the other two.

DATASETS holds everything that differs between them, so that the differences are
visible in one place instead of being spread over the run scripts: paths, the
arguments of `preprocessing.clean`, and whether LOOP is run.
"""

from __future__ import annotations

import numpy as np


def load_sherlock(roi, root="data/sherlock"):
    """Chen et al. (2017), one ROI. Two runs concatenated along the last axis."""
    return np.load(f"{root}/sherlock_{roi}.npy")


def load_things(path="data/THINGS_fMRI1_V1.npy"):
    """THINGS fMRI, repeated presentations of the test stimuli."""
    return np.load(path)


def load_eeg(path="data/THINGS_EEG2.npy"):
    """THINGS-EEG, repeated presentations of the test stimuli.

    The stored array is (stimuli, repetitions, channels, time); channels and
    time are folded into a single unit axis.
    """
    raw = np.load(path, allow_pickle=True)
    data = raw.item()["preprocessed_eeg_data"].transpose(1, 2, 3, 0)
    return data.reshape(data.shape[0], -1, data.shape[-1])


DATASETS = {
    "sherlock": {
        "loader": load_sherlock,
        "rois": ["V1", "A1", "IT", "amygdala", "hippocampus"],
        # 1976 time points, two runs, cleaned separately; LOOP holds out each run
        "clean": {"tr": 1.5, "cutoff_s": 140.0, "run_lengths": [946, 1030]},
        "loop": True,
        "n_bootstraps": 50,
        "results": "results/sherlock_results_{roi}.npy",
    },
    "things": {
        "loader": load_things,
        "rois": None,
        # the last axis indexes stimuli, which show no autocorrelation: each
        # series is only centred and z-scored. Too few stimuli for a LOOP test set.
        "clean": {"cutoff_s": None},
        "loop": False,
        "n_bootstraps": 50,
        "results": "results/THINGS_results.npy",
    },
    "eeg": {
        "loader": load_eeg,
        "rois": None,
        "clean": {"cutoff_s": None},
        "loop": False,
        "n_bootstraps": 50,
        "results": "results/eeg_results.npy",
    },
}
