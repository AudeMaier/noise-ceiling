# THINGS-EEG preprocessing

`THINGS_EEG2.npy` in `eeg.zip` on [OSF](https://osf.io/2btpr/) is a file from the THINGS-EEG2 dataset (Gifford et al., 2022), copied without any change. It is `sub-01/preprocessed_eeg_test.npy` from the [preprocessed EEG data](https://osf.io/anp5v/) of the dataset's OSF project ([osf.io/3jk45](https://osf.io/3jk45/)). Only subject 1 is used.

No preprocessing is done in this repository. The file is the authors' preprocessed release; their paper and the dataset's OSF project describe how it was made.

## Contents

The file holds a dictionary. `preprocessed_eeg_data` is the array used here, of shape `(200 test images, 80 repetitions, 17 channels, 100 time points)`.

- The 17 channels are the occipital and parietal ones listed in `ch_names`.
- The time points run from −0.2 s to 0.79 s around image onset, in steps of 10 ms (`times`).

`load_eeg` in `noise_ceiling/datasets/loaders.py` rearranges the array to `(80 repetitions, 17 × 100 units, 200 images)`. A unit is a channel at one time point. Correlations are computed across images.
