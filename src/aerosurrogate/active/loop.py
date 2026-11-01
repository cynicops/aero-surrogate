"""Pool-based active learning with ensemble-variance acquisition, and the space-filling
baseline it must beat. Works with any oracle: the synthetic function, a lookup into an
existing CFD dataset (retrospective study) or a live CFD runner.

Fairness rules: both methods start from the same initial design, use the same surrogate
settings and seeds, and are scored on the same held-out test set. Failed oracle calls count
as spent runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..surrogate import EnsembleConfig, EnsembleSurrogate, features, metrics

Oracle = Callable[[pd.DataFrame], pd.DataFrame]  # returns cl, cd, cm (NaN = failed run)


@dataclass
class ALConfig:
    n_init: int = 8
    batch_size: int = 4
    budget: int = 48
    target_rmse_cl: float = 0.05  # stop when held-out CL RMSE <= this (benchmark mode)
    output_weights: dict = field(default_factory=lambda: {"cl": 1.0, "cd": 1.0, "cm": 0.5})
    diversity: float = 0.15  # min unit-cube distance between points picked in one batch
    seed: int = 0


def acquisition(sur: EnsembleSurrogate, Xpool: np.ndarray, y_scale: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted sum of ensemble standard deviations, each normalised by the output's spread."""
    _, sd = sur.predict(Xpool)
    return (sd / y_scale) @ weights


def _pick(scores: np.ndarray, U: np.ndarray, k: int, min_dist: float) -> list[int]:
    order = np.argsort(-scores)
    chosen: list[int] = []
    for i in order:
        if all(np.linalg.norm(U[i] - U[j]) >= min_dist for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    for i in order:  # top up if the diversity rule was too strict
        if len(chosen) == k:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return chosen


def _evaluate(sur_cfg, inputs, train_df, test_X, test_Y):
    ok = train_df[["cl", "cd", "cm"]].notna().all(axis=1)
    d = train_df[ok]
    sur = EnsembleSurrogate(inputs, ["cl", "cd", "cm"], sur_cfg).fit(features(d, inputs), d[["cl", "cd", "cm"]].to_numpy())
    mu, sd = sur.predict(test_X)
    m = metrics(test_Y, mu, sd)
    return sur, m


def run_comparison(
    pool: pd.DataFrame,
    oracle: Oracle,
    test: pd.DataFrame,
    inputs: list[str],
    unit: np.ndarray,
    al: ALConfig,
    sur_cfg: EnsembleConfig,
    progress=print,
) -> pd.DataFrame:
    """Run active learning and space filling (pool order = Sobol order) on the same pool.

    ``pool`` rows must be in space-filling (e.g. Sobol) order; ``unit`` is the pool in the
    unit cube (for the diversity rule). Returns a long history table.
    """
    test_X = features(test, inputs)
    test_Y = test[["cl", "cd", "cm"]].to_numpy()
    history = []
    for method in ("space_filling", "active"):
        chosen = list(range(al.n_init))
        labelled = pd.concat([pool.iloc[chosen].reset_index(drop=True), oracle(pool.iloc[chosen]).reset_index(drop=True)], axis=1)
        while True:
            sur, m = _evaluate(sur_cfg, inputs, labelled, test_X, test_Y)
            rec = {"method": method, "n_runs": len(chosen), "n_ok": int(labelled[["cl", "cd", "cm"]].notna().all(axis=1).sum())}
            for _, r in m.iterrows():
                rec[f"rmse_{r['output']}"] = r["rmse"]
                rec[f"r2_{r['output']}"] = r["r2"]
                rec[f"cov2s_{r['output']}"] = r["coverage_2sigma"]
            history.append(rec)
            progress(f"{method:13s} runs={rec['n_runs']:3d} rmse_cl={rec['rmse_cl']:.4f} rmse_cd={rec['rmse_cd']:.5f}")
            remaining = [i for i in range(len(pool)) if i not in chosen]
            if len(chosen) >= al.budget or not remaining:
                break
            k = min(al.batch_size, al.budget - len(chosen), len(remaining))
            if method == "space_filling":
                new = remaining[:k]
            else:
                ok = labelled[["cl", "cd", "cm"]].notna().all(axis=1)
                y_scale = labelled.loc[ok, ["cl", "cd", "cm"]].std().to_numpy() + 1e-12
                w = np.array([al.output_weights.get(o, 0.0) for o in ("cl", "cd", "cm")])
                scores = acquisition(sur, features(pool.iloc[remaining], inputs), y_scale, w)
                new = [remaining[i] for i in _pick(scores, unit[remaining], k, al.diversity)]
            res = oracle(pool.iloc[new]).reset_index(drop=True)
            labelled = pd.concat([labelled, pd.concat([pool.iloc[new].reset_index(drop=True), res], axis=1)],
                                 ignore_index=True)
            chosen += new
    return pd.DataFrame(history)


def runs_to_target(history: pd.DataFrame, target: float, col: str = "rmse_cl") -> dict:
    out = {}
    for method, h in history.groupby("method"):
        hit = h[h[col] <= target]
        out[method] = int(hit["n_runs"].iloc[0]) if len(hit) else None
    return out
