"""One case end to end: geometry -> mesh -> case files -> run -> post -> report -> schema row.

Failures at any stage produce a row with ``converged = False`` and a ``status`` /
``failure_reason``; they are logged in the dataset, never silently dropped.
"""

from __future__ import annotations

import json
import math
import traceback
from dataclasses import asdict
from pathlib import Path

import yaml

from ..config import CaseConfig
from ..data.schema import now_utc
from ..foam.case import render_context, write_case
from ..foam.post import collect
from ..geometry.wing import GeometryError, WingParams, build_wing, check_wing
from ..mesh.gmsh_mesher import mesh_wing
from .launchers import Launcher
from .local import RunResult, run_openfoam


def base_row(case_id: str, params: WingParams, cfg: CaseConfig, launcher: Launcher | None) -> dict:
    row = {
        "case_id": case_id,
        "solver": "simpleFoam",
        "solver_version": launcher.version_label if launcher else "",
        "turbulence_model": f"{cfg.flow.turbulence_model} (fully turbulent)",
        "mesh_cells": 0,
        "y_plus_max": float("nan"),
        "geometry_params": json.dumps({"elements": [asdict(e) for e in cfg.elements], "params": params.to_dict()}),
        "aoa_deg": params.aoa_deg,
        "re": params.re,
        "ride_height_over_chord": params.ride_height_over_chord,
        "cl": float("nan"),
        "cd": float("nan"),
        "cm": float("nan"),
        "converged": False,
        "residual_final": float("nan"),
        "wall_clock_s": 0.0,
        "source": "openfoam",
        "status": "pending",
        "failure_reason": "",
        "mesh_refinement": cfg.mesh.refinement,
        "created_utc": now_utc(),
    }
    row.update(params.to_flat())
    return row


def run_case(
    case_id: str,
    params: WingParams,
    cfg: CaseConfig,
    launcher: Launcher,
    out_root: str | Path,
    make_report: bool = True,
    mesh_only: bool = False,
) -> dict:
    case_dir = Path(out_root).resolve() / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_config.yaml").write_text(
        yaml.safe_dump({"case_id": case_id, "params": params.to_dict(), "config": cfg.to_dict(),
                        "launcher": launcher.version_label, "n_procs": launcher.n_procs}, sort_keys=False)
    )
    row = base_row(case_id, params, cfg, launcher)
    row["n_procs"] = launcher.n_procs

    # geometry
    try:
        wing = check_wing(
            build_wing(cfg.elements, params, n_per_side=cfg.limits.n_points_per_side),
            min_gap=cfg.limits.min_gap,
            min_ride_height=cfg.limits.min_ride_height,
            max_x_extent=cfg.limits.max_x_extent,
        )
    except GeometryError as e:
        row.update(status="geometry_invalid", failure_reason=str(e))
        _write_result(case_dir, row)
        return row
    row["geometry_params"] = wing.geometry_json()
    row["geometry_hash"] = wing.geometry_hash()

    # mesh
    try:
        mesh = mesh_wing(wing, cfg.mesh, case_dir, name="mesh")
    except Exception as e:  # gmsh raises plain Exceptions
        row.update(status="mesh_failed", failure_reason=f"{type(e).__name__}: {e}")
        _write_result(case_dir, row, traceback.format_exc())
        return row
    row["mesh_cells"] = mesh.n_cells

    # case files + run
    ctx = render_context(case_id, wing, mesh, cfg, launcher.n_procs)
    write_case(case_dir, ctx, mesh.msh_path)
    result: RunResult = run_openfoam(case_dir, launcher, cfg, wing.names, mesh_only=mesh_only)
    row.update(
        status=result.status,
        failure_reason=result.failure_reason,
        stop_reason=result.stop_reason,
        converged=bool(result.converged),
        n_iter=result.n_iter,
        wall_clock_s=round(result.wall_clock_s, 2),
        cl=result.cl, cd=result.cd, cm=result.cm,
        cl_std=result.cl_std, cd_std=result.cd_std, cm_std=result.cm_std,
        residual_final=result.residual_final,
        solver_version=result.foam_version or launcher.version_label,
    )
    if result.checkmesh.get("cells"):
        row["mesh_cells"] = result.checkmesh["cells"]
    if mesh_only:
        row["checkmesh"] = result.checkmesh
        _write_result(case_dir, row, extra={"mesh": asdict(mesh)})
        return row

    summary = {}
    if not math.isnan(result.cl):
        try:
            summary = collect(case_dir, wing.names, cfg.flow.u_inf)
            row["y_plus_max"] = summary.get("y_plus_max", float("nan"))
            row["y_plus_mean"] = summary.get("y_plus_mean", float("nan"))
        except Exception as e:
            row["notes"] = f"post-processing problem: {e}"
    _write_result(case_dir, row, extra={"run": result.to_dict(), "mesh": asdict(mesh), "post": summary})

    if make_report:
        try:
            from ..report.html import write_case_report

            write_case_report(case_dir, row, wing, mesh, cfg)
        except Exception as e:  # a report problem must not lose the CFD result
            row["notes"] = (row.get("notes", "") + f" report failed: {e}").strip()
            _write_result(case_dir, row, traceback.format_exc())
    return row


def _write_result(case_dir: Path, row: dict, tb: str | None = None, extra: dict | None = None) -> None:
    data = {"row": row}
    if extra:
        data.update(extra)
    if tb:
        data["traceback"] = tb
    (case_dir / "result.json").write_text(json.dumps(data, indent=2, default=str))
