"""Single-file HTML reports (figures embedded as base64 PNG)."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import jinja2
import pandas as pd

from .. import __version__
from . import plots

TEMPLATES = Path(__file__).parent / "templates"


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plots.plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _env():
    return jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)), autoescape=True)


def _clean(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and math.isnan(v):
            v = 0.0 if k in {"cl", "cd", "cm", "cl_std", "cd_std", "cm_std", "y_plus_max", "y_plus_mean",
                             "residual_final"} else v
        out[k] = v
    return out


def write_case_report(case_dir, row: dict, wing=None, mesh=None, cfg=None, gci: dict | None = None,
                      window: int | None = None) -> Path:
    case_dir = Path(case_dir)
    res_dir = case_dir / "results"
    window = window or (cfg.convergence.window if cfg else 200)
    figures = []
    if mesh is not None and wing is not None and Path(mesh.preview_path).exists():
        figures.append(("Mesh", fig_to_b64(plots.mesh_plot(mesh.preview_path, wing.polygons, row))))
    fh = res_dir / "force_history.csv"
    if fh.exists():
        rh = res_dir / "residual_history.csv"
        figures.append(("Convergence history", fig_to_b64(plots.history_plot(
            pd.read_csv(fh), pd.read_csv(rh) if rh.exists() else None, row, window))))
    if (res_dir / "cp.csv").exists():
        figures.append(("Surface pressure", fig_to_b64(plots.cp_plot(pd.read_csv(res_dir / "cp.csv"), row))))
    if (res_dir / "yplus.csv").exists():
        figures.append(("y+ histogram", fig_to_b64(plots.yplus_hist(pd.read_csv(res_dir / "yplus.csv"), row))))
    elements = []
    if (res_dir / "element_coefficients.csv").exists():
        elements = pd.read_csv(res_dir / "element_coefficients.csv").to_dict("records")

    setup = {}
    if cfg is not None:
        setup = {
            "elements": ", ".join(f"{e.airfoil} (c={e.chord:.3f}, TE {100*e.te_thickness:.2f}% c)" for e in cfg.elements),
            "AoA [deg]": row.get("aoa_deg"),
            "flaps [deg]": ", ".join(str(round(row.get(f"flap{i}_deg"), 3)) for i in (1, 2) if f"flap{i}_deg" in row),
            "gaps / overlaps [c_ref]": ", ".join(
                f"{row.get(f'gap{i}'):.4f} / {row.get(f'overlap{i}'):.4f}" for i in (1, 2) if f"gap{i}" in row),
            "ride height h/c": row.get("ride_height_over_chord"),
            "Re": f"{row.get('re'):.4g}",
            "ground": cfg.flow.ground,
            "inflow turbulence": f"I = {cfg.flow.turbulence_intensity:.3%}, nut/nu = {cfg.flow.viscosity_ratio}",
            "schemes": f"U: {cfg.solver.div_u}; k, omega: {cfg.solver.div_turb}",
        }
    if mesh is not None:
        setup.update({
            "cells": f"{mesh.n_cells:,} ({mesh.n_quads_2d:,} quads + {mesh.n_tris_2d:,} triangles in 2D)",
            "boundary layer": f"{mesh.bl_layers} layers, first cell {mesh.first_cell_height:.2e} c, total {mesh.bl_thickness:.2e} c",
            "mesh refinement factor": cfg.mesh.refinement if cfg else "",
        })
    html = _env().get_template("case_report.html.j2").render(
        row=_clean(row), provenance=plots.provenance(row), figures=figures, elements=elements, setup=setup,
        checkmesh=_checkmesh(case_dir), gci=gci, window=window, version=__version__, case_dir=str(case_dir),
    )
    out = case_dir / "report.html"
    out.write_text(html)
    return out


def _checkmesh(case_dir: Path) -> dict:
    log = case_dir / "logs" / "checkMesh.log"
    if not log.exists():
        return {}
    from ..foam.parsers import parse_checkmesh

    return parse_checkmesh(log.read_text(errors="replace"))
