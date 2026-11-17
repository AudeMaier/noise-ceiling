# Sherlock preprocessing

How the files in `sherlock.zip` on [OSF](https://osf.io/2btpr/) were made. To reproduce the results of the paper, you only need to download those files. The steps below are for anyone who wants to redo the preprocessing.

The data are from Chen et al. (2017): 16 subjects watched the first episode of *Sherlock*, split into two runs. The raw data are on OpenNeuro as [ds001132, version 1.0.0](https://openneuro.org/datasets/ds001132/versions/1.0.0) (doi:10.18112/openneuro.ds001132.v1.0.0).

## 1. fMRIPrep

```bash
bash preprocessing/sherlock/run_fmriprep.sh /path/to/ds001132 /path/to/freesurfer/license.txt
```

This runs fMRIPrep 25.2.3 with its default settings on every subject. It was run from the `nipreps/fmriprep:25.2.3` Docker image. The only outputs used afterwards are the preprocessed BOLD series in `MNI152NLin2009cAsym` space (`*_desc-preproc_bold.nii.gz`) and one brain mask (`*_desc-brain_mask.nii.gz`), which only provides the voxel grid.

## 2. ROI time series

```bash
python preprocessing/sherlock/extract_rois.py /path/to/ds001132/derivatives data/sherlock
```

The script does the following for every subject and run:

1. **Select the ROI voxels.** Each ROI is defined from a Harvard-Oxford maximum-probability atlas (threshold 25%, 2 mm) and resampled to the BOLD grid by nearest neighbour.

   | ROI | Atlas | Labels |
   | --- | --- | --- |
   | V1 | cortical | intracalcarine cortex |
   | A1 | cortical | Heschl's gyrus |
   | IT | cortical | inferior temporal gyrus, temporo-occipital part |
   | amygdala | subcortical | left and right amygdala |
   | hippocampus | subcortical | left and right hippocampus |

2. **Z-score** every voxel over the time points of the run.
3. **Trim the run** so that every subject keeps the same time points of the movie. The number of TRs dropped at the start and at the end of each run is listed per subject in `TRIM`. These offsets are taken from the README of the OpenNeuro dataset.
4. **Concatenate the two runs**, which gives 946 + 1030 = 1976 time points.

The output is `sherlock_{roi}.npy`, of shape `(16 subjects, voxels, 1976)`.

No confounds are regressed out and no spatial smoothing is applied. The only further processing is done when the estimators are computed (`noise_ceiling/datasets/preprocessing.py`). There, each run is high-pass filtered with a cutoff at 140 s and z-scored again.

The script was run with Python 3.8, nilearn 0.10.4, nibabel 5.2.1 and numpy 1.24.4. It also runs with nilearn 0.12.
