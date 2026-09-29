"""
One reader for the contaminated training files, shared by every consumer.

inject_contamination.py writes three different CSV layouts, mirroring how each
benchmark stores its own training data:

  SMD, SWaT   .npy, features only
  PSM, LTDB   .csv written with index=False; column 0 is a timestamp or id and
              columns 1.. are the features
  WADI        .csv written with index=True; column 0 is the index, the last
              column is the original label, and the features sit in between

Reading any of those with a plain `pd.read_csv(...).select_dtypes(number)` picks
up the index and the label as if they were channels, which is how WADI turned
into 129 channels against a model built for 127. This module resolves the layout
from the expected channel count instead of guessing per dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_contaminated(path, expected_c: int) -> np.ndarray:
    """Return the [L, expected_c] feature matrix of a contaminated training file.

    `expected_c` comes from the dataset config (`input_c`), so the layout is
    resolved by counting rather than by hard-coding a rule per dataset.
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    else:
        df = pd.read_csv(path)
        num = df.select_dtypes(include=[np.number])
        extra = num.shape[1] - expected_c
        if extra == 0:
            arr = num.to_numpy(np.float32)              # features only
        elif extra == 1:
            arr = num.iloc[:, 1:].to_numpy(np.float32)  # leading meta column
        elif extra == 2:
            arr = num.iloc[:, 1:-1].to_numpy(np.float32)  # index and trailing label
        else:
            raise ValueError(
                f"{path.name}: found {num.shape[1]} numeric columns for a model with "
                f"{expected_c} channels. Expected the difference to be 0, 1 or 2.")
    if arr.ndim != 2 or arr.shape[1] != expected_c:
        raise ValueError(f"{path.name}: resolved shape {arr.shape}, "
                         f"expected (*, {expected_c})")
    return arr


def standardise(arr: np.ndarray):
    """Zero-mean unit-variance per channel, matching sklearn's StandardScaler.

    NaN-safe on both sides. A plain mean/std turns one missing value into a whole
    NaN channel: PSM carried 9,548 NaN and the naive version produced 1,510,284.
    Statistics are computed over the finite entries and any residual hole is
    filled with the channel mean, so the model never sees NaN.
    """
    arr = np.asarray(arr, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(arr, axis=0, keepdims=True)
        sd = np.nanstd(arr, axis=0, keepdims=True)
    mu = np.nan_to_num(mu, nan=0.0)              # a wholly missing channel -> 0
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    out = (np.where(np.isnan(arr), mu, arr) - mu) / sd
    return out.astype(np.float32)


def conf_value(cf, option, cast=str):
    """Read one option from a CP-MAE .conf without hard-coding its section.

    The keys are spread over [model], [data], [train] and [param], and they have
    moved between sections before. Looking the option up by name across every
    section removes a whole class of NoOptionError.
    """
    for sec in cf.sections():
        if cf.has_option(sec, option):
            raw = cf.get(sec, option)
            if cast is bool:
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return cast(raw)
    raise KeyError(f"option {option!r} is in none of {cf.sections()}")
