# THINGS-fMRI preprocessing

How the file in `things.zip` on [OSF](https://osf.io/2btpr/) was made. To reproduce the results of the paper, you only need to download that file. The steps below are for anyone who wants to rebuild it.

The data are from THINGS-fMRI (Hebart et al., 2023). We start from the single-trial responses that the authors released in table format: [THINGS-data: fMRI Single Trial Responses (table format)](https://doi.org/10.25452/figshare.plus.20492835.v2), figshare, version 2. Only subject 1 is used, so the three `sub-01_*` files of `betas_csv.zip` are all you need:

- `sub-01_ResponseData.h5`
- `sub-01_StimulusMetadata.csv`
- `sub-01_VoxelMetadata.csv`

## Extraction

```bash
python preprocessing/things_fmri/extract_v1.py /path/to/betas_csv data/THINGS_fMRI1_V1.npy
```

The script makes two selections from the released responses:

- **Voxels:** those labelled V1 in `sub-01_VoxelMetadata.csv`, which gives 1049 voxels.
- **Trials:** those of the 100 test images, each shown 12 times over the sessions.

The output has shape `(12 repetitions, 1049 voxels, 100 images)`. The repetitions are in the order they were presented, and the images are in alphabetical order of their file names.

The responses are not processed any further here. When the estimators are computed, the responses of each voxel in each repetition are only centred and z-scored across images (`noise_ceiling/datasets/loaders.py`).
