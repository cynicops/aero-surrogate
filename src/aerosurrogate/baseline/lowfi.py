"""Cheap low-fidelity estimates from existing tools (we use them, we do not rebuild them):

* CL, CM: AeroSandbox ``AirfoilInviscid`` (linear-vortex panel method) on all elements at once,
  with its built-in mirror-image ground model (inviscid: cannot tell a moving from a fixed ground).
  CM is integrated from the panel surface speeds (Cp = 1 - gamma^2).
* CD: NeuralFoil profile drag per element, at the angle where NeuralFoil's CL matches the
  element's inviscid loading ("matched-loading" heuristic, ours). Crude by design.

Rows produced here are marked ``source = "neuralfoil"`` (SPEC.md) and the ``solver`` column
names both tools.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import asdict

import numpy as np

from ..data.schema import now_utc
from ..geometry.airfoil import load_section
from ..geometry.wing import Wing

SOLVER_NAME = "aerosandbox.AirfoilInviscid(ground images) + neuralfoil(matched-loading CD)"


@contextlib.contextmanager
def _quiet():
    """Silence IPOPT/CasADi output written straight to file descriptor 1."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def neuralfoil_section(name: str, te_thickness: float, alpha_deg, re, transition: str = "free",
                       model_size: str = "xlarge") -> dict:
    """NeuralFoil on one section in the lift-up frame. transition: 'free' (n_crit 9) or 'forced' (at LE)."""
    import aerosandbox as asb
    import neuralfoil as nf

    af = asb.Airfoil(name=name, coordinates=load_section(name, 120, te_thickness))
    kw = {"xtr_upper": 0.0, "xtr_lower": 0.0} if transition == "forced" else {}
    r = nf.get_aero_from_airfoil(af, alpha=np.atleast_1d(alpha_deg).astype(float), Re=re,
                                 model_size=model_size, **kw)
    return {k: np.asarray(r[k]) for k in ("CL", "CD", "CM", "analysis_confidence")}


def inviscid(wing: Wing, ground: bool = True) -> dict:
    import aerosandbox as asb

    afs = [asb.Airfoil(name=n, coordinates=p) for n, p in zip(wing.names, wing.polygons)]
    with _quiet():
        a = asb.AirfoilInviscid(airfoil=afs, op_point=asb.OperatingPoint(velocity=1.0, alpha=0.0),
                                ground_effect=ground)
    cl_circ = [float(af.Cl) for af in a.airfoils]
    # Pressure-integrated forces and moment (q = 1/2, c_ref = 1). Near a ground plane the
    # image system induces velocity at the wing, so Kutta-Joukowski with U_inf alone is not
    # the force; the pressure integral is used for CL, CD_pressure and CM.
    fx = fy = mz = 0.0
    cl_el = []
    x0, y0 = wing.moment_ref
    for af in a.airfoils:
        xy = np.column_stack([af.x(), af.y()])
        g = np.asarray(af.gamma, dtype=float)
        cp = 1.0 - g**2
        cp_mid = 0.5 * (cp[:-1] + cp[1:])
        d = np.diff(xy, axis=0)
        n_out = np.column_stack([d[:, 1], -d[:, 0]])  # outward for counter-clockwise ordering
        mid = 0.5 * (xy[:-1] + xy[1:])
        dfx, dfy = -cp_mid * n_out[:, 0], -cp_mid * n_out[:, 1]
        fx += dfx.sum()
        fy += dfy.sum()
        cl_el.append(float(dfy.sum()))
        mz += np.sum((mid[:, 0] - x0) * dfy - (mid[:, 1] - y0) * dfx)
    return {"cl": float(fy), "cl_elements": cl_el, "cl_circulation": float(sum(cl_circ)),
            "cd_pressure": float(fx), "cm": float(-mz)}  # nose-up positive


def lowfi_coefficients(wing: Wing, re: float, ground: bool = True) -> dict:
    inv = inviscid(wing, ground)
    alphas = np.arange(-10.0, 25.01, 0.25)
    cd_total = 0.0
    stalled = []
    for spec, chord, cl_ground in zip(wing.specs, wing.chords, inv["cl_elements"]):
        target = -cl_ground / chord  # lift-up frame, normalised by the element chord
        r = neuralfoil_section(spec.airfoil, spec.te_thickness, alphas, re * chord)
        cl_curve = r["CL"]
        i_max = int(np.argmax(cl_curve))
        rising = cl_curve[: i_max + 1]
        if target >= rising[-1]:
            i = i_max
            stalled.append(True)
        else:
            i = int(np.searchsorted(rising, target)) if np.all(np.diff(rising) > 0) else int(np.argmin(np.abs(rising - target)))
            i = min(max(i, 0), i_max)
            stalled.append(False)
        cd_total += float(r["CD"][i]) * chord
    return {"cl": inv["cl"], "cd": cd_total, "cm": inv["cm"], "stalled_elements": stalled,
            "cl_elements": inv["cl_elements"]}


LOWFI_PANELS_PER_SIDE = 40  # panel solution converged to <0.2 % in CL vs 80 (Phase 0 check)


def lowfi_wing(elements, params) -> Wing:
    from ..geometry.wing import build_wing

    return build_wing(elements, params, n_per_side=LOWFI_PANELS_PER_SIDE)


def lowfi_row(case_id: str, wing: Wing, solver_version: str | None = None) -> dict:
    """A schema row (source = neuralfoil) for an already-validated wing (build it with lowfi_wing)."""
    import aerosandbox as asb
    import neuralfoil as nf

    p = wing.params
    out = lowfi_coefficients(wing, p.re)
    row = {
        "case_id": case_id,
        "solver": SOLVER_NAME,
        "solver_version": solver_version or f"aerosandbox {asb.__version__}, neuralfoil {getattr(nf, '__version__', '0.3.3')}",
        "turbulence_model": "none (inviscid panel) + NeuralFoil (XFoil-trained, free transition)",
        "mesh_cells": 0,
        "y_plus_max": float("nan"),
        "geometry_params": wing.geometry_json(),
        "aoa_deg": p.aoa_deg,
        "re": p.re,
        "ride_height_over_chord": p.ride_height_over_chord,
        "cl": out["cl"],
        "cd": out["cd"],
        "cm": out["cm"],
        "converged": True,
        "residual_final": float("nan"),
        "wall_clock_s": float("nan"),
        "source": "neuralfoil",
        "status": "ok",
        "failure_reason": "",
        "notes": json.dumps({"stalled_elements": out["stalled_elements"]}),
        "geometry_hash": wing.geometry_hash(),
        "created_utc": now_utc(),
    }
    row.update(p.to_flat())
    return row


__all__ = ["inviscid", "lowfi_coefficients", "lowfi_row", "neuralfoil_section", "SOLVER_NAME", "asdict"]
