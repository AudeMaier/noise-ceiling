# Accuracy of Noise Ceiling Estimators for Brain Scores

Code for the paper *Accuracy of Noise Ceiling Estimators for Brain Scores*, by Aude Maier, Lucas Gruaz, Martin Schrimpf and Johanni Brea. Noise ceiling estimators are compared against each other and against their closed forms, on synthetic data and on three brain datasets.

## Install

Use Python 3.12 with the versions pinned in `requirements.txt`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                           # makes `import noise_ceiling` work
```

`pip install -e .` is what lets the scripts in `experiments/` find the package; without it, run them with `PYTHONPATH=.`.

Only the LOOP estimator needs a GPU. Everything else runs on CPU.

## Data

The data files are too large for this repository. They are on OSF, [osf.io/2btpr](https://osf.io/2btpr/), as `sherlock.zip`, `things.zip` and `eeg.zip` (about 500 MB in all). To download and unpack them:

```bash
python scripts/download_data.py                    # all three
python scripts/download_data.py --only things.zip  # or some of them
```

Every archive is checked against `data_checksums.txt` before it is unpacked. The files then sit where the code expects them; to keep them elsewhere, edit the paths in `noise_ceiling/datasets/loaders.py`:

```
data/sherlock/sherlock_{V1,A1,IT,amygdala,hippocampus}.npy
data/THINGS_fMRI1_V1.npy
data/THINGS_EEG2.npy
```

How each file was made from the public releases of the datasets is described in `preprocessing/`, with the scripts to rebuild it:

- Sherlock (Chen et al., 2017): `preprocessing/sherlock/`
- THINGS-fMRI (Hebart et al., 2023): `preprocessing/things_fmri/`
- THINGS-EEG2 (Gifford et al., 2022): `preprocessing/things_eeg/`

Every dataset is an array of shape `(n_trials, n_units, n_samples)`. Trials are separate subjects for Sherlock and repeated presentations for THINGS and THINGS-EEG. The last axis is the one correlations are computed across: time points for the continuous Sherlock stimulus, stimuli for the other two.

## Running

Every paper figure, from scratch:

```bash
python experiments/run_synthetic.py                  # the four synthetic regimes
python experiments/run_empirical.py sherlock --loop  # all five ROIs, LOOP included (GPU recommended)
python experiments/run_empirical.py things
python experiments/run_empirical.py eeg
python experiments/make_figures.py                   # Figures 2 and 3
python experiments/run_T_convergence_sherlock.py     # Figure 4
```

The run scripts write to `results/`, the figure scripts to `figures/figures_paper/`. Useful flags:

- `run_synthetic.py configs/synthetic/within_subject.yaml` runs one regime.
- `run_empirical.py sherlock --roi V1` restricts Sherlock to one ROI.

On Sherlock, `--loop` does more than add the LOOP estimator computation: LOOP holds out each of the two runs in turn, so every other estimator is then also computed on each run separately and averaged over the runs, so that all of them see the same data. The paper uses `--loop`. Without it, LOOP is skipped and the other estimators are computed on the whole recording, so their values differ slightly from the paper's. LOOP runs on CPU too, only much more slowly.

`make_figures.py` keeps the mean curves it reads in `results/figure_curves.pkl` and refreshes them when a result file is newer. `run_T_convergence_sherlock.py` caches its measurements in `results/finite_T_sherlock_{roi}.pkl`. Pass `--recompute` to either to start over.

## Configuration

**Synthetic data.** Each synthetic regime has a config file in `configs/synthetic/`.

**Recorded data.** The settings that differ between Sherlock, THINGS and THINGS-EEG are all in the `DATASETS` dictionary of `noise_ceiling/datasets/loaders.py`: where the data and results are stored, the ROIs, how each time series is cleaned before the estimators are computed, how many trial subsets are drawn, and whether LOOP is run.

## Layout

```
noise_ceiling/
  estimators/
    theoretical_models.py     closed forms
    empirical/
      estimators.py           SH, SHSB, ATV, LOOC, TTMC, and the true ceilings
      loop.py                 LOOP, and the train/test splits it needs
      ridge_GPU/              ridge regression on the GPU
  datasets/
    synthetic.py              SyntheticDataset: estimates, closed forms, sweeps
    generation.py             the generative model
    empirical.py              EmpiricalDataset: loading, sweeps
    loaders.py                per-dataset paths and settings
    preprocessing.py          trend removal and standardisation
  analysis.py                 fit of the finite-T trend, a - b T^-p
  plotting.py, style.py       figures
  config.py, io.py            configuration and result files

experiments/                  entry points
configs/                      synthetic parameters
preprocessing/                how the data files on OSF were made
scripts/download_data.py      downloads them, checked against data_checksums.txt
tests/                        sanity checks, run with pytest
```

Estimators are referred to by short keys throughout: `SH`, `SHSB`, `ATV`, `LOOC`, `TTMC`, `LOOP`, and `NC_mean` / `NC_single` for the two true ceilings, which are only available on synthetic data.

## Tests

```bash
pytest
```
