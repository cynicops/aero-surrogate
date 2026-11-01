"""Feature extraction, case-wise splits and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTPUTS = ["cl", "cd", "cm"]


def features(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Inputs for the surrogate. Re enters as log10(Re)."""
    cols = []
    for n in names:
        v = df[n].to_numpy(dtype=float)
        cols.append(np.log10(v) if n == "re" else v)
    return np.column_stack(cols)


def split_by_case(df: pd.DataFrame, fractions=(0.7, 0.15, 0.15), seed: int = 0, group: str = "case_id"):
    """Split by case (group), never by row: all rows of one case land in the same set.

    Low-fidelity rows share the case_id of the CFD case they describe, so they follow it.
    Returns (train_ids, val_ids, test_ids) as sets.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    ids = np.array(sorted(df[group].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(fractions[2] * n)))
    n_val = int(round(fractions[1] * n))
    test, val, train = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]
    return set(train), set(val), set(test)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray | None = None,
            names: list[str] = OUTPUTS) -> pd.DataFrame:
    y_true, y_pred = np.atleast_2d(y_true), np.atleast_2d(y_pred)
    rows = []
    for j, n in enumerate(names):
        t, p = y_true[:, j], y_pred[:, j]
        err = p - t
        ss_res, ss_tot = np.sum(err**2), np.sum((t - t.mean()) ** 2)
        r = {"output": n, "n": len(t), "rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))),
             "max_abs": float(np.max(np.abs(err))), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
             "mean_rel_err": float(np.mean(np.abs(err) / np.maximum(np.abs(t), 1e-9)))}
        if y_std is not None:
            s = np.atleast_2d(y_std)[:, j]
            r["coverage_2sigma"] = float(np.mean(np.abs(err) <= 2 * s))  # ~0.95 if calibrated & Gaussian
            r["mean_std"] = float(np.mean(s))
        rows.append(r)
    return pd.DataFrame(rows)
