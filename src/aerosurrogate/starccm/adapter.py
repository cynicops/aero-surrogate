"""STAR-CCM+ adapter. UNTESTED: no STAR-CCM+ licence in the cloud workspace.

* ``write_macro``: render a Java macro that sets global parameters, runs, and exports reports.
* ``import_result``: turn the macro's CSV + a small JSON sidecar into a schema row
  (source = "starccm"), converting sign conventions explicitly.

Only the Python side (rendering and parsing) is unit-tested, against a synthetic fixture
written by us. The macro itself and the real CSV layout are UNTESTED until run on a licence.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import jinja2

from ..data.schema import now_utc, validate_row

UNTESTED = True
TEMPLATES = Path(__file__).parent / "templates"


def write_macro(case_id: str, params: dict, out_dir: str | Path, csv_path: str,
                reports: dict | None = None, iterations: int = 2000) -> Path:
    """UNTESTED on STAR-CCM+. Renders RunAeroCase_<id>.java into out_dir."""
    reports = reports or {"cl": "CL", "cd": "CD", "cm": "CM"}
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)), trim_blocks=True, lstrip_blocks=True,
                             undefined=jinja2.StrictUndefined)
    java_id = re.sub(r"\W", "_", case_id)
    text = env.get_template("RunAeroCase.java.j2").render(
        case_id=case_id, java_id=java_id, params={k: float(v) for k, v in params.items()},
        csv_path=csv_path, reports=reports, iterations=int(iterations))
    out = Path(out_dir) / f"RunAeroCase_{java_id}.java"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def read_macro_csv(path: str | Path) -> dict:
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["key"]] = row["value"]
    return data


def import_result(csv_path: str | Path, sidecar_path: str | Path) -> dict:
    """UNTESTED against real STAR-CCM+ output. Build a schema row.

    Sidecar JSON (written by you per case) must contain::

        {"case_id", "params" (flat WingParams fields), "geometry_params" (JSON-able),
         "reports": {"cl": name, "cd": name, "cm": name},
         "lift_sign": +1 or -1   # multiply STAR's lift report to get lift-positive-up
         "moment_sign": +1 or -1 # multiply STAR's moment report to get nose-up positive
         "reference": {"length": c_ref, "area": A_ref, "note": "..."},
         "turbulence_model", "mesh_cells", "converged", "residual_final", "wall_clock_s",
         "y_plus_max"}
    STAR reports must already be non-dimensionalised with the same c_ref and per-unit-span
    area as the OpenFOAM cases (docs/conventions.md); the sidecar records what was used.
    """
    raw = read_macro_csv(csv_path)
    side = json.loads(Path(sidecar_path).read_text())
    rep = side["reports"]

    def report(name):
        key = f"report:{name}"
        if key not in raw:
            raise KeyError(f"report '{name}' not found in {csv_path}")
        return float(raw[key])

    p = side["params"]
    row = {
        "case_id": side["case_id"],
        "solver": "STAR-CCM+",
        "solver_version": raw.get("star_version", "unknown"),
        "turbulence_model": side.get("turbulence_model", "unknown"),
        "mesh_cells": int(side.get("mesh_cells", 0)),
        "y_plus_max": float(side.get("y_plus_max", float("nan"))),
        "geometry_params": json.dumps(side.get("geometry_params", {"params": p}), sort_keys=True),
        "aoa_deg": float(p["aoa_deg"]),
        "re": float(p["re"]),
        "ride_height_over_chord": float(p["ride_height_over_chord"]),
        "cl": side.get("lift_sign", 1) * report(rep["cl"]),
        "cd": report(rep["cd"]),
        "cm": side.get("moment_sign", 1) * report(rep["cm"]),
        "converged": bool(side.get("converged", False)),
        "residual_final": float(side.get("residual_final", float("nan"))),
        "wall_clock_s": float(side.get("wall_clock_s", float("nan"))),
        "source": "starccm",
        "status": "ok" if side.get("converged") else "not_converged",
        "failure_reason": "",
        "notes": f"UNTESTED importer; reference: {side.get('reference')}",
        "created_utc": now_utc(),
    }
    row.update({k: v for k, v in p.items() if k.startswith(("flap", "gap", "overlap"))})
    validate_row(row)
    return row
