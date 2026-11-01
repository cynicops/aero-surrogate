"""Collect per-case results after a run: Cp per element, y+ per wall patch, histories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .parsers import latest_time_dir, read_force_coeffs, read_patch_values, read_raw_surface, read_solver_info


def collect(case_dir: str | Path, wall_patches: list[str], u_inf: float = 1.0) -> dict:
    """Write results/*.csv and return summary numbers. Missing pieces are reported, not raised."""
    case_dir = Path(case_dir)
    out = case_dir / "results"
    out.mkdir(exist_ok=True)
    summary: dict = {"warnings": []}
    q = 0.5 * u_inf**2

    # force & residual histories
    hist = read_force_coeffs(case_dir)
    if not hist.empty:
        hist.to_csv(out / "force_history.csv", index=False)
    res = read_solver_info(case_dir)
    if not res.empty:
        res.to_csv(out / "residual_history.csv", index=False)

    # per-element contributions (last value)
    rows = []
    for p in wall_patches:
        h = read_force_coeffs(case_dir, f"forceCoeffs_{p}")
        if not h.empty:
            rows.append({"element": p, "cl": h["Cl"].iloc[-1], "cd": h["Cd"].iloc[-1], "cm": h["CmPitch"].iloc[-1]})
    if rows:
        pd.DataFrame(rows).to_csv(out / "element_coefficients.csv", index=False)

    # y+
    t = latest_time_dir(case_dir)
    yp_frames = []
    if t is not None and (t / "yPlus").exists():
        for p in wall_patches + ["ground"]:
            try:
                v = read_patch_values(t / "yPlus", p)
                yp_frames.append(pd.DataFrame({"patch": p, "y_plus": v}))
            except (KeyError, ValueError):
                continue
    if yp_frames:
        yp = pd.concat(yp_frames, ignore_index=True)
        yp.to_csv(out / "yplus.csv", index=False)
        wing = yp[yp["patch"] != "ground"]["y_plus"]
        summary["y_plus_max"] = float(wing.max())
        summary["y_plus_mean"] = float(wing.mean())
    else:
        summary["warnings"].append("no yPlus field found")

    # Cp (surface sample at the final write time)
    cp_frames = []
    base = case_dir / "postProcessing" / "wallPressure"
    if base.exists():
        times = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: float(d.name))
        if times:
            for p in wall_patches:
                files = list(times[-1].glob(f"*{p}*.raw"))
                if files:
                    df = read_raw_surface(files[0])
                    df = df[np.isclose(df["z"], df["z"].min())] if df["z"].nunique() > 1 else df
                    df["cp"] = df["p"] / q
                    df["element"] = p
                    cp_frames.append(df[["element", "x", "y", "cp"]])
    if cp_frames:
        pd.concat(cp_frames, ignore_index=True).to_csv(out / "cp.csv", index=False)
    else:
        summary["warnings"].append("no surface pressure sample found")
    return summary
