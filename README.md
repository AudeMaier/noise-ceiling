# Noise ceiling estimators

Code for the paper "Accuracy of Noise Ceiling Estimators for Brain Scores". Noise ceiling estimators are compared against each other and against their closed forms, on synthetic data and on three brain datasets.

## Install

The code runs on Python 3.8 and 3.12. Two pinned environments are provided,
matching the machines the results were produced on:

```bash
pip install -r requirements-py38-gpu.txt   # GPU runs, Python 3.8.13, CUDA 12.1
pip install -r requirements-py312.txt      # CPU runs, Python 3.12.3
pip install -e .                           # makes `import noise_ceiling` work
```

`pip install -e .` is what lets the scripts in `experiments/` find the package;
without it, run them with `PYTHONPATH=.`.

Only the LOOP estimator needs a GPU. Everything else runs on CPU.

## Data
The three datasets are:
- Sherlock: (Chen et al., 2016)
- THINGS fMRI1 (Hebart et al., 2023)
- THINGS EEG2 (Gifford et al., 2022)

They are not included in the repository. Download and unpack it with:
 
```bash
python scripts/download_data.py --view-only 6a9a67ebc6aa4426a48321c880fa41be
```
 
The data is hosted on OSF and can also be browsed and downloaded by hand at
<https://osf.io/2btpr/?view_only=6a9a67ebc6aa4426a48321c880fa41be>, unpacking
the archives into `data/`.
 
This lays the files out as the loaders expect:
 
```
data/sherlock/sherlock_{V1,A1,IT,Amygdala,Hippocampus}.npy
data/THINGS_fMRI1_V1.npy
data/THINGS_EEG2.npy
```

## Running

```bash
python experiments/run_synthetic.py                  # all three synthetic regimes
python experiments/run_synthetic.py configs/synthetic/within_subject.yaml
python experiments/run_T_convergence.py              # finite-T bias on synthetic data
python experiments/run_T_convergence_sherlock.py     # finite-T bias on sherlock
python experiments/run_empirical.py sherlock         # all five ROIs
python experiments/run_empirical.py things
python experiments/run_empirical.py eeg
python experiments/make_figures.py                   # paper figures
python experiments/retention_curve_aic.py            # validation of AIC selection
```

Useful flags for `run_empirical.py`:

- `--roi V1` restricts Sherlock to one ROI.
- `--loop` also computes the LOOP estimator. Off by default: it is much slower
  than the others and needs a GPU to be practical.
- `--recompute-aic` redoes the unit selection instead of reusing the cached
  Delta AIC distribution in `results/AIC_dist/`. The procedure is stochastic, so
  recomputing generally selects a slightly different set of units.

Which figure comes from which command:

| Figure | Needs |
| --- | --- |
| 2 | `run_synthetic.py` (within_subject), `run_empirical.py things` |
| 3 | `run_synthetic.py` (sensory_areas, cognitive_areas), `run_empirical.py sherlock --roi V1 --roi IT` |
| 4 | `run_empirical.py sherlock` |
| finite-T synthetic | `run_T_convergence.py` |
| finite-T Sherlock | `run_T_convergence_sherlock.py` |
| AIC retention curves | `retention_curve_aic.py` |

`run_T_convergence.py` caches its simulations in
`data/finite_T_bias/finite_T_bias_tmax.pkl` and reuses them on the next run;
delete the file to resimulate.

## Configuration

Synthetic parameters live in `configs/synthetic/*.yaml`; the `model` block maps
one to one onto `generation.generate_voxel_grids`. Everything that differs
between the recorded datasets — paths, cleaning, AIC threshold, whether test
chunks have to be purged — is in `DATASETS` in
`noise_ceiling/datasets/loaders.py`.

## Layout

```
noise_ceiling/
  estimators/
    theoretical_models.py     closed forms
    empirical/
      estimators.py           SH, SHSB, SNR, LOOC, TTMC, and the true ceilings
      loop.py                 LOOP, and the train/test splits it needs
      ridge_GPU/              ridge regression on the GPU
  datasets/
    synthetic.py              SyntheticDataset: estimates, closed forms, sweeps
    generation.py             the generative model
    empirical.py              EmpiricalDataset: loading, selection, sweeps
    loaders.py                per-dataset paths and settings
    preprocessing.py          trend removal and standardisation
  selection.py                AIC selection of units carrying signal
  analysis.py                 bias and the power-law fit
  plotting.py, style.py       figures
  config.py, io.py            configuration and result files

experiments/                  entry points
configs/                      synthetic parameters
tests/                        sanity checks, run with pytest
```

Estimators are referred to by short keys throughout: `SH`, `SHSB`, `SNR`,
`LOOC`, `TTMC`, `LOOP`, and `NC_mean` / `NC_single` for the two true ceilings,
which are only available on synthetic data. Result files written before this
naming used long names; `noise_ceiling.io.load_results` converts them.
