"""Optimisation on the surrogate, with an uncertainty penalty, and CFD verification.

Objective (maximised): pessimistic downforce  -mu_cl - kappa * sd_cl,
optionally subject to a drag cap  mu_cd + kappa * sd_cd <= cd_max  (penalty method).
Search: scipy differential evolution in the unit cube of the varied parameters (seeded).
Candidates are then checked for geometric validity; invalid ones are dropped and reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from ..geometry.wing import GeometryError, WingParams, build_wing, validate_wing
from ..sampling import ParameterSpace
from ..surrogate import EnsembleSurrogate, features


@dataclass
class OptimConfig:
    kappa: float = 1.0
    cd_max: float | None = None
    objective: str = "downforce"  # "downforce" or "efficiency" (-cl/cd)
    n_starts: int = 4
    maxiter: int = 150
    popsize: int = 20
    seed: int = 0
    min_gap: float = 0.003
    min_ride_height: float = 0.02


def _score(sur: EnsembleSurrogate, space: ParameterSpace, U: np.ndarray, cfg: OptimConfig) -> np.ndarray:
    df = space.from_unit(U)
    mu, sd = sur.predict(features(df, sur.input_names))
    cl, cd = mu[:, 0], mu[:, 1]
    if cfg.objective == "efficiency":
        val = (-cl - cfg.kappa * sd[:, 0]) / np.maximum(cd + cfg.kappa * sd[:, 1], 1e-6)
    else:
        val = -cl - cfg.kappa * sd[:, 0]
    if cfg.cd_max is not None:
        excess = np.maximum(0.0, cd + cfg.kappa * sd[:, 1] - cfg.cd_max)
        val = val - 1e3 * excess
    return val


def optimise(sur: EnsembleSurrogate, space: ParameterSpace, cfg: OptimConfig) -> pd.DataFrame:
    """Multi-start DE; returns candidates (best first) with predictions and validity."""
    dim = len(space.dims)
    found = []
    for k in range(cfg.n_starts):
        res = differential_evolution(
            lambda u: -float(_score(sur, space, u[None, :], cfg)[0]),
            bounds=[(0.0, 1.0)] * dim, seed=cfg.seed + k, maxiter=cfg.maxiter, popsize=cfg.popsize,
            tol=1e-8, polish=True, updating="immediate",
        )
        found.append(res.x)
    U = np.array(found)
    df = space.from_unit(U)
    mu, sd = sur.predict(features(df, sur.input_names))
    for j, n in enumerate(["cl", "cd", "cm"]):
        df[f"pred_{n}"], df[f"pred_{n}_std"] = mu[:, j], sd[:, j]
    df["score"] = _score(sur, space, U, cfg)
    df["start"] = range(cfg.n_starts)
    valid, why = [], []
    n_flaps = len(space.elements) - 1
    for _, r in df.iterrows():
        try:
            w = build_wing(space.elements, WingParams.from_flat(r.to_dict(), n_flaps))
            p = validate_wing(w, min_gap=cfg.min_gap, min_ride_height=cfg.min_ride_height)
        except GeometryError as e:
            p = [str(e)]
        valid.append(not p)
        why.append("; ".join(p))
    df["geometry_valid"], df["geometry_problems"] = valid, why
    # flag extrapolation: anything on the boundary of the box is suspicious
    df["on_bound"] = np.any((U < 1e-3) | (U > 1 - 1e-3), axis=1)
    return df.sort_values("score", ascending=False).reset_index(drop=True)
