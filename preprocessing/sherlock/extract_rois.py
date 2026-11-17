"""Sherlock ROI time series from the fMRIPrep outputs.

    python preprocessing/sherlock/extract_rois.py /path/to/ds001132/derivatives data/sherlock

For every subject and each of the two runs, the voxels of each ROI are read from
the preprocessed BOLD series in MNI152NLin2009cAsym space, z-scored over time and
trimmed to the movie; the two runs are then concatenated. This writes
sherlock_{roi}.npy, of shape (16 subjects, voxels, 946 + 1030 TRs), for every
ROI in ROIS: the files distributed on OSF. Nothing else is done to the data: no
confound regression and no smoothing. The drift removal applied before the
estimators are computed is in noise_ceiling/datasets/preprocessing.py.
"""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np
from nilearn import datasets, image

SUBJECTS = [f"{i:02d}" for i in range(1, 17)]
RUN_LENGTHS = {1: 946, 2: 1030}
SPACE = "MNI152NLin2009cAsym"

# Harvard-Oxford maximum-probability atlases, thresholded at 25%, 2 mm:
# (atlas, label values)
ROIS = {
    "V1": ("cort-maxprob-thr25-2mm", [24]),            # intracalcarine cortex
    "A1": ("cort-maxprob-thr25-2mm", [45]),            # Heschl's gyrus
    "IT": ("cort-maxprob-thr25-2mm", [16]),            # inferior temporal gyrus, temporo-occipital part
    "amygdala": ("sub-maxprob-thr25-2mm", [10, 20]),   # left and right
    "hippocampus": ("sub-maxprob-thr25-2mm", [9, 19]),  # left and right
}

# TRs dropped at the start and at the end of each run so that every subject
# keeps exactly the TRs of the movie: TRIM[run][subject] = (start, end). From
# the README of OpenNeuro ds001132 v1.0.0.
TRIM = {
    1: {"01": (20, 7), "02": (20, 7), "03": (20, 6), "04": (6, 13),
        "05": (20, 8), "06": (20, 7), "07": (20, 6), "08": (20, 5),
        "09": (20, 4), "10": (20, 6), "11": (20, 3), "12": (20, 5),
        "13": (20, 5), "14": (20, 6), "15": (20, 3), "16": (20, 5)},
    2: {"01": (20, 11), "02": (20, 7), "03": (20, 7), "04": (6, 5),
        "05": (20, 7), "06": (20, 7), "07": (20, 10), "08": (20, 7),
        "09": (20, 8), "10": (20, 8), "11": (20, 6), "12": (20, 7),
        "13": (20, 6), "14": (20, 4), "15": (20, 4), "16": (20, 6)},
}


def _func(derivatives, subject, run, suffix):
    return (f"{derivatives}/sub-{subject}/func/"
            f"sub-{subject}_task-sherlockPart{run}_space-{SPACE}_{suffix}.nii.gz")


def roi_masks(reference):
    """Boolean mask of every ROI on the voxel grid of `reference`."""
    masks = {}
    for roi, (atlas_name, labels) in ROIS.items():
        atlas = datasets.fetch_atlas_harvard_oxford(atlas_name).maps
        region = image.new_img_like(atlas, np.isin(atlas.get_fdata(), labels).astype(np.float32))
        masks[roi] = image.resample_to_img(region, reference, interpolation="nearest"
                                           ).get_fdata().astype(bool)
    return masks


def extract(derivatives, subjects=SUBJECTS):
    """{roi: array (subjects, voxels, TRs)}."""
    reference = nib.load(_func(derivatives, subjects[0], 1, "desc-brain_mask"))
    masks = roi_masks(reference)

    data = {roi: [] for roi in ROIS}
    for subject in subjects:
        runs = {roi: [] for roi in ROIS}
        for run in RUN_LENGTHS:
            img = nib.load(_func(derivatives, subject, run, "desc-preproc_bold"))
            assert np.allclose(img.affine, reference.affine) and img.shape[:3] == reference.shape, \
                f"sub-{subject} run {run} is not on the grid of sub-{subjects[0]}"
            bold = img.get_fdata()
            start, end = TRIM[run][subject]
            for roi, mask in masks.items():
                x = bold[mask]                                  # (voxels, TRs)
                x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)
                x = x[:, start:x.shape[-1] - end]
                assert x.shape[-1] == RUN_LENGTHS[run], (subject, run, x.shape)
                runs[roi].append(x)
        for roi in ROIS:
            data[roi].append(np.concatenate(runs[roi], axis=-1))
        print(f"sub-{subject} done")
    return {roi: np.array(x) for roi, x in data.items()}


def main(derivatives, out_dir):
    data = extract(derivatives)
    os.makedirs(out_dir, exist_ok=True)
    for roi in ROIS:
        np.save(f"{out_dir}/sherlock_{roi}.npy", data[roi])
        print(f"{roi}: {data[roi].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("derivatives", help="fMRIPrep output directory, holding sub-01/ ... sub-16/")
    parser.add_argument("out_dir", nargs="?", default="data/sherlock")
    args = parser.parse_args()
    main(args.derivatives, args.out_dir)
