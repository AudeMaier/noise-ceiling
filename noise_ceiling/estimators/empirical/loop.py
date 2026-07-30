"""LOOP: leave-one-out prediction.

One trial is predicted from all the others with a ridge encoding model, and the
held-out correlation is the noise ceiling estimate. Unlike the correlation-based
estimators, LOOP needs a train/test split; `make_split` builds it, and the same
split can be handed to the other estimators so that all of them are evaluated on
the same held-out samples.
"""

from __future__ import annotations

import numpy as np
import torch

from .ridge_GPU.ridge_regression import ridge_corr_pred_torch, ridge_corr_torch


def purged_test_chunks(data_length, n_chunks=5, test_fraction=0.25, buffer_size=50):
    """Contiguous test chunks, one per equal-width bin.

    A buffer of `buffer_size` samples on each side of a chunk is excluded from
    training, so that autocorrelation cannot leak across the split.
    """
    chunk_size = int(data_length * test_fraction) // n_chunks
    bin_size = data_length // n_chunks

    if chunk_size < 1:
        raise ValueError(
            f"test_fraction={test_fraction} with n_chunks={n_chunks} gives chunks "
            f"of less than one sample for data_length={data_length}")
    if bin_size - chunk_size < 1:
        raise ValueError(
            f"bins of {bin_size} samples cannot hold a {chunk_size}-sample chunk; "
            f"reduce n_chunks or test_fraction")

    test_indices, buffered = [], []
    for i in range(n_chunks):
        start = np.random.randint(i * bin_size, (i + 1) * bin_size - chunk_size)
        end = start + chunk_size
        test_indices.append(np.arange(start, end))
        buffered.append(np.arange(max(0, start - buffer_size),
                                  min(data_length, end + buffer_size)))

    test_idx = np.concatenate(test_indices)
    keep = np.ones(data_length, dtype=bool)
    keep[np.concatenate(buffered)] = False
    return np.where(keep)[0], test_idx


def random_split(data_length, test_fraction=0.25):
    """Test samples drawn uniformly at random, with no buffer."""
    test_idx = np.random.choice(data_length, size=int(test_fraction * data_length),
                                replace=False)
    return np.setdiff1d(np.arange(data_length), test_idx), test_idx


def make_split(data_length, chunked=True, n_chunks=5, test_fraction=0.25,
               buffer_size=50):
    """Train/test split as (train_idx, test_idx).

    chunked=True gives purged contiguous chunks, appropriate when neighbouring
    samples are correlated (a continuous stimulus). chunked=False gives a random
    split, appropriate when the last axis indexes unrelated stimuli.
    """
    if chunked:
        return purged_test_chunks(data_length, n_chunks=n_chunks,
                                  test_fraction=test_fraction,
                                  buffer_size=buffer_size)
    return random_split(data_length, test_fraction=test_fraction)


def _standardize(train, *others, eps=1e-8):
    """Z-score every array using the training statistics only."""
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return [(a - mu) / sd for a in (train,) + others]


def ridge_prediction(variables, target, split, device="cpu",
                     alphas=np.logspace(-3, 3, 7), n_validations=3,
                     val_fraction=0.1, val_chunk=True, n_chunk_val=3,
                     val_buffer=50, standardize="train", verbose=False,
                     return_alphas=False):
    """Ridge prediction of `target` (T, n_targets) from `variables` (T, n_features).

    The ridge parameter is selected per target on validation splits drawn from
    the training samples, then the model is refit on the full training set.

    split       : (train_idx, test_idx), from `make_split`.
    standardize : "train" uses training statistics only; "all" uses the whole
                  recording, which lets held-out samples inform the penalty.

    Returns the test correlation per target, and the selected ridge parameters
    if `return_alphas`.
    """
    log = print if verbose else (lambda *args, **kwargs: None)

    if standardize == "all":
        variables = (variables - variables.mean(0)) / variables.std(0)
        target = (target - target.mean(0)) / target.std(0)

    train_idx, test_idx = split
    X_train, X_test = variables[train_idx], variables[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]

    if standardize == "train":
        X_train, X_test = _standardize(X_train, X_test)
        y_train, y_test = _standardize(y_train, y_test)

    def to_tensor(array):
        return torch.tensor(array).to(device).to(float)

    X_train_t, y_train_t = to_tensor(X_train), to_tensor(y_train)

    scores = torch.zeros((len(alphas), y_test.shape[1])).to(device)
    for _ in range(n_validations):
        sub_train, sub_val = make_split(X_train.shape[0], chunked=val_chunk,
                                        n_chunks=n_chunk_val,
                                        test_fraction=val_fraction,
                                        buffer_size=val_buffer)
        scores += ridge_corr_torch(X_train_t[sub_train], X_train_t[sub_val],
                                   y_train_t[sub_train], y_train_t[sub_val],
                                   alphas) / n_validations

    best_idx = torch.argmax(scores.cpu(), dim=0).numpy()
    best_alpha = np.asarray(alphas)[best_idx]

    n_low = int(np.sum(best_idx == 0))
    n_high = int(np.sum(best_idx == len(alphas) - 1))
    if n_low or n_high:
        print(f"[loop] {n_low} targets selected the smallest and {n_high} the "
              f"largest of {len(alphas)} ridge parameters; widen `alphas`.")

    out = ridge_corr_pred_torch(X_train_t, to_tensor(X_test), y_train_t,
                                to_tensor(y_test), to_tensor(best_alpha),
                                normalpha=False)
    log(out.mean().item())
    out = out.cpu().numpy()
    return (out, best_alpha) if return_alphas else out


def loop_estimate(data, split, alphas, index=0, device="cpu", n_validations=10,
                  val_chunk=True, **kwargs):
    """LOOP estimate for one held-out trial.

    Predicts every unit of trial `index` from all units of the remaining trials.

    data : (n_trials, n_units, n_samples).

    Returns the held-out correlation per unit.
    """
    n_samples = data.shape[-1]
    variables = np.delete(data, index, axis=0).reshape(-1, n_samples).T
    target = data[index].T
    return ridge_prediction(variables, target, split, device=device, alphas=alphas,
                            n_validations=n_validations, val_chunk=val_chunk,
                            **kwargs)
