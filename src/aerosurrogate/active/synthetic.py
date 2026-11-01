"""SYNTHETIC test function for the active-learning loop. It is NOT aerodynamics data.

A smooth, cheap function of the wing parameters with a few features that resemble the real
problem (a downforce peak at small ride height, a soft "stall" when the total flap angle is
large, a slot-gap optimum), so the loop can be tested in seconds. Rows made from it carry
``source = "synthetic"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def synthetic_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    aoa = df["aoa_deg"].to_numpy(float)
    f1 = df["flap1_deg"].to_numpy(float)
    f2 = df["flap2_deg"].to_numpy(float)
    h = df["ride_height_over_chord"].to_numpy(float)
    re = df["re"].to_numpy(float)
    g1 = df.get("gap1", pd.Series(0.015, index=df.index)).to_numpy(float)
    o1 = df.get("overlap1", pd.Series(0.02, index=df.index)).to_numpy(float)

    ground = 1.0 + 0.9 * np.exp(-((np.log(h) - np.log(0.12)) / 0.7) ** 2) - 0.6 * np.exp(-h / 0.035)
    load = 1.2 + 0.11 * aoa + 0.035 * f1 + 0.022 * f2
    stall = 1.0 - 0.45 * _sig((aoa + f1 + f2 - 62.0) / 4.0)
    slot = 1.0 - 400.0 * (g1 - 0.016) ** 2 - 60.0 * (o1 - 0.015) ** 2
    reyn = 1.0 + 0.06 * np.log10(re / 5e5)
    down = load * ground * stall * slot * reyn
    cd = 0.025 + 0.006 * down**2 + 0.08 * _sig((aoa + f1 + f2 - 62.0) / 4.0) + 0.01 * np.exp(-h / 0.05)
    cm = 0.15 + 0.12 * down - 0.004 * f2
    return pd.DataFrame({"cl": -down, "cd": cd, "cm": cm}, index=df.index)
