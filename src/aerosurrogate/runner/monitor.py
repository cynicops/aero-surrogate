"""Convergence and divergence rules applied to the force-coefficient history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ConvergenceConfig


@dataclass
class ForceStats:
    n_iter: int
    cl: float
    cd: float
    cm: float
    cl_std: float
    cd_std: float
    cm_std: float
    cl_range_rel: float
    cd_range_rel: float
    cm_range_abs: float


def window_stats(hist: pd.DataFrame, window: int) -> ForceStats | None:
    """Mean/std/range of Cl, Cd, Cm over the last ``window`` iterations."""
    if hist is None or hist.empty or "Cl" not in hist:
        return None
    w = hist.tail(window)
    cl, cd, cm = w["Cl"].to_numpy(), w["Cd"].to_numpy(), w["CmPitch"].to_numpy()
    def rel(a, floor):
        return float((a.max() - a.min()) / max(abs(a.mean()), floor))
    return ForceStats(
        n_iter=int(hist.iloc[-1, 0]),
        cl=float(cl.mean()), cd=float(cd.mean()), cm=float(cm.mean()),
        cl_std=float(cl.std()), cd_std=float(cd.std()), cm_std=float(cm.std()),
        cl_range_rel=rel(cl, 0.05), cd_range_rel=rel(cd, 0.005),
        cm_range_abs=float(cm.max() - cm.min()),
    )


def forces_converged(hist: pd.DataFrame, cfg: ConvergenceConfig) -> tuple[bool, ForceStats | None]:
    """Stop rule: after min_iter, the window's relative ranges are below tolerance."""
    if hist is None or hist.empty or len(hist) < max(cfg.min_iter, cfg.window):
        return False, window_stats(hist, cfg.window)
    s = window_stats(hist, cfg.window)
    ok = (
        s.cl_range_rel <= cfg.cl_rel_tol
        and s.cd_range_rel <= cfg.cd_rel_tol
        and s.cm_range_abs <= cfg.cm_abs_tol
    )
    return ok, s


def diverged(hist: pd.DataFrame, limit: float = 100.0) -> str | None:
    """Return a reason if the coefficients blew up."""
    if hist is None or hist.empty:
        return None
    last = hist[["Cl", "Cd", "CmPitch"]].tail(5).to_numpy()
    if not np.all(np.isfinite(last)):
        return "non-finite force coefficients"
    if np.abs(last).max() > limit:
        return f"|coefficient| > {limit}"
    return None
