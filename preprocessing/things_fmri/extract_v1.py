"""THINGS-fMRI V1 responses to the repeated test images.

    python preprocessing/things_fmri/extract_v1.py /path/to/betas_csv data/THINGS_fMRI1_V1.npy

Reads the single-trial responses of subject 1 from the table-format release of
THINGS-data, keeps the V1 voxels and the 100 test images, each shown 12 times,
and writes an array of shape (12 repetitions, voxels, 100 images): the file
distributed on OSF. Repetitions are in the order they were presented and
images in alphabetical order of their file names. The responses are used as
released, with no further processing.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import pandas as pd

SUBJECT = "01"
ROI = "V1"
N_REPS = 12


def extract(betas_dir, subject=SUBJECT, roi=ROI):
    """Array (repetitions, voxels, images)."""
    trials = pd.read_csv(f"{betas_dir}/sub-{subject}_StimulusMetadata.csv")
    voxels = pd.read_csv(f"{betas_dir}/sub-{subject}_VoxelMetadata.csv", usecols=[roi])
    test = trials[trials["trial_type"] == "test"]
    rows = np.flatnonzero(voxels[roi] == 1)

    # one row per voxel and one column per trial, in the order of the two
    # metadata tables; only the ROI rows are read
    with h5py.File(f"{betas_dir}/sub-{subject}_ResponseData.h5", "r") as f:
        responses = f["ResponseData/block0_values"]
        assert responses.shape == (len(voxels), len(trials)), responses.shape
        responses = responses[rows]

    images = np.unique(test["stimulus"])
    out = np.empty((N_REPS, len(rows), len(images)))
    for i, image in enumerate(images):
        shown = test.index[test["stimulus"] == image]
        assert len(shown) == N_REPS, (image, len(shown))
        out[:, :, i] = responses[:, shown].T
    return out


def main(betas_dir, out):
    data = extract(betas_dir)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, data)
    print(f"wrote {out}: {data.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("betas_dir", help="folder holding the sub-01_* files of the release")
    parser.add_argument("out", nargs="?", default="data/THINGS_fMRI1_V1.npy")
    args = parser.parse_args()
    main(args.betas_dir, args.out)
