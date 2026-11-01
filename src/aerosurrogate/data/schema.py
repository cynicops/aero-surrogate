"""Shared data schema for OpenFOAM, STAR-CCM+, NeuralFoil/AeroSandbox and synthetic results.

One row per case (CSV), plus one folder per case, plus a metadata JSON sidecar that stores
units, reference length/area and sign conventions (never assumed).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

SCHEMA_VERSION = 1

# Required by SPEC.md, in this order.
REQUIRED_COLUMNS = [
    "case_id", "solver", "solver_version", "turbulence_model", "mesh_cells", "y_plus_max",
    "geometry_params", "aoa_deg", "re", "ride_height_over_chord", "cl", "cd", "cm",
    "converged", "residual_final", "wall_clock_s", "source",
]
# Optional, but written by this toolkit.
OPTIONAL_COLUMNS = [
    "status", "failure_reason", "stop_reason", "n_iter", "n_procs", "cl_std", "cd_std", "cm_std",
    "y_plus_mean", "mesh_refinement", "geometry_hash", "created_utc", "notes",
    "flap1_deg", "gap1", "overlap1", "flap2_deg", "gap2", "overlap2",
]
SOURCES = {"openfoam", "starccm", "neuralfoil", "synthetic"}


class SchemaError(ValueError):
    pass


def default_metadata(extra: dict | None = None) -> dict:
    meta = {
        "schema_version": SCHEMA_VERSION,
        "reference_length": "c_ref = sum of element chords; all lengths in the dataset are divided by c_ref",
        "reference_area": "c_ref x unit span (2D, coefficients are per unit span)",
        "dynamic_pressure": "q_inf = 0.5 rho U_inf^2",
        "reynolds_number": "Re = U_inf c_ref / nu",
        "moment_reference_point": "main-element quarter chord (on its chord line)",
        "axes": "x downstream (freestream direction), y up, ground plane at y = 0",
        "sign_conventions": {
            "cl": "positive up (lift); a downforce wing has cl < 0. Downforce coefficient = -cl",
            "cd": "positive downstream",
            "cm": "positive nose-up (standard aerodynamic convention), pitch axis -z",
            "aoa_deg": "main-element incidence; positive = leading edge down (more downforce)",
            "flap_deg": "deflection relative to the previous element's chord; positive = TE towards suction side",
        },
        "geometry_definitions": {
            "ride_height_over_chord": "ground to lowest point of the assembly / c_ref",
            "gap_i": "min surface distance between element i-1 and element i / c_ref",
            "overlap_i": "along element i-1 chord, from flap LE to element i-1 TE midpoint / c_ref (+ = flap LE upstream)",
        },
        "units": {
            "wall_clock_s": "seconds",
            "aoa_deg": "degrees",
            "flap_deg": "degrees",
            "other lengths": "non-dimensional (/ c_ref)",
            "coefficients": "non-dimensional",
        },
        "created_utc": now_utc(),
    }
    if extra:
        meta.update(extra)
    return meta


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_row(row: dict) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in row]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    if row["source"] not in SOURCES:
        raise SchemaError(f"source must be one of {sorted(SOURCES)}, got {row['source']!r}")
    try:
        json.loads(row["geometry_params"])
    except (TypeError, json.JSONDecodeError) as e:
        raise SchemaError(f"geometry_params must be a JSON string: {e}") from e
    if not isinstance(row["converged"], (bool,)) and row["converged"] not in (0, 1):
        raise SchemaError("converged must be bool")


def validate_frame(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    bad = set(df["source"].unique()) - SOURCES
    if bad:
        raise SchemaError(f"unknown sources: {sorted(bad)}")
    for g in df["geometry_params"]:
        json.loads(g)
