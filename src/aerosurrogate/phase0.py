"""Phase 0 feasibility spike: single element in ground effect.

* ride-height sweep (moving ground) + a near-freestream reference (slip ground, h/c = 5)
* 3-level mesh study with GCI at h/c = 0.2
* NeuralFoil and inviscid-panel references (sanity checks, NOT validation)
* comparison against published data if ``validation_csv`` exists (Gate 0)

The section is a public stand-in (AeroSandbox 'ls417' = LS(1)-0417 / GA(W)-1) until the
coordinates of the Zerihan & Zhang (2000) wing are obtained; see docs/phase0_report.md.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline.lowfi import inviscid, neuralfoil_section
from .config import CaseConfig, LauncherConfig
from .data.dataset import Dataset
from .geometry.wing import ElementSpec, WingParams, build_wing
from .report import plots
from .report.gci import cell_size_2d, gci_three_grids
from .runner.campaign import Job, run_campaign

SECTION = ElementSpec("ls417", 1.0, 0.001)
AOA = 4.0
RE = 5.0e5
RIDE_HEIGHTS = [0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
FREESTREAM_H = 5.0
GCI_H = 0.2
GCI_FACTORS = [1.0, 1.5, 2.25]  # mesh refinement factors (sizes / factor)


def build_jobs(base: CaseConfig) -> list[Job]:
    jobs = []
    for h in RIDE_HEIGHTS:
        jobs.append(Job(f"p0_h{h:.3f}", vars(WingParams(AOA, h, RE)), base.to_dict()))
    fs = CaseConfig.from_dict(base.to_dict())
    fs.flow = replace(fs.flow, ground="slip")
    jobs.append(Job(f"p0_freestream_h{FREESTREAM_H:.1f}", vars(WingParams(AOA, FREESTREAM_H, RE)), fs.to_dict()))
    for f in GCI_FACTORS[1:]:
        c = CaseConfig.from_dict(base.to_dict())
        c.mesh = c.mesh.refined(f)
        c.convergence = replace(c.convergence, wall_clock_limit_s=3600)
        jobs.append(Job(f"p0_h{GCI_H:.3f}_mesh{f:.2f}", vars(WingParams(AOA, GCI_H, RE)), c.to_dict()))
    return jobs


def run_phase0(launcher_cfg: LauncherConfig, out: str | Path, workers: int = 2,
               validation_csv: str | Path = "docs/validation/reference_lift.csv") -> dict:
    out = Path(out)
    base = CaseConfig(elements=[SECTION])
    ds = Dataset(out / "dataset", metadata={"campaign": "phase0", "section": "ls417 stand-in"})
    rows = run_campaign(build_jobs(base), launcher_cfg, ds.cases_dir, ds, workers=workers)
    df = pd.DataFrame(rows)
    df.to_csv(out / "phase0_runs.csv", index=False)
    return summarise(df, out, validation_csv)


def summarise(df: pd.DataFrame, out: Path, validation_csv) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"aoa_deg": AOA, "re": RE, "section": "ls417 (LS(1)-0417 / GA(W)-1), TE 0.1% c"}
    base_mask = ~df["case_id"].str.contains("mesh") & ~df["case_id"].str.contains("freestream")
    sweep = df[base_mask].sort_values("ride_height_over_chord")
    summary["sweep"] = sweep[["case_id", "ride_height_over_chord", "status", "cl", "cd", "cm", "cl_std",
                              "n_iter", "wall_clock_s", "mesh_cells", "y_plus_max", "residual_final"]].to_dict("records")

    # inviscid reference curve (images) for the same geometry
    inv = []
    for h in RIDE_HEIGHTS:
        w = build_wing([SECTION], WingParams(AOA, h, RE), n_per_side=40)
        inv.append({"h": h, "cl_inviscid": inviscid(w, ground=True)["cl"]})
    summary["inviscid"] = inv

    # freestream reference vs NeuralFoil
    fs = df[df["case_id"].str.contains("freestream")]
    nf_free = neuralfoil_section(SECTION.airfoil, SECTION.te_thickness, AOA, RE, "free")
    nf_forced = neuralfoil_section(SECTION.airfoil, SECTION.te_thickness, AOA, RE, "forced")
    summary["freestream"] = {
        "openfoam": fs[["cl", "cd", "cm", "status"]].to_dict("records"),
        "neuralfoil_free_transition_liftup_frame": {k: float(v[0]) for k, v in nf_free.items()},
        "neuralfoil_forced_transition_liftup_frame": {k: float(v[0]) for k, v in nf_forced.items()},
        "note": "OpenFOAM values are in the inverted (downforce) frame: compare -cl with NeuralFoil CL, "
                "+cm with -CM. h/c = 5 with a slip ground is close to, not exactly, freestream.",
    }

    # GCI
    g = df[(df["ride_height_over_chord"] == GCI_H) & ~df["case_id"].str.contains("freestream")].copy()
    g = g[g["cl"].notna()].sort_values("mesh_cells", ascending=False)
    if len(g) >= 3:
        cells = g["mesh_cells"].astype(int).tolist()[:3]
        hs = [cell_size_2d(n) for n in cells]
        summary["gci"] = {}
        for q in ("cl", "cd", "cm"):
            vals = g[q].tolist()[:3]
            try:
                summary["gci"][q] = gci_three_grids(*vals, *hs).to_dict()
            except ValueError as e:
                summary["gci"][q] = {"error": str(e), "values": vals}
        summary["gci_cells"] = cells
        summary["gci_wall_clock_s"] = g["wall_clock_s"].tolist()[:3]
        summary["gci_status"] = g["status"].tolist()[:3]

    # Gate 0 comparison against published data (only if provided)
    vpath = Path(validation_csv)
    if vpath.exists():
        ref = pd.read_csv(vpath, comment="#")
        comp = []
        for _, r in ref.iterrows():
            m = sweep[np.isclose(sweep["ride_height_over_chord"], r["h_over_c"])]
            if len(m):
                cfd = -float(m["cl"].iloc[0])
                err = (cfd - r["cl_downforce"]) / r["cl_downforce"]
                comp.append({"h_over_c": r["h_over_c"], "ref": r["cl_downforce"], "cfd": cfd, "rel_err": err,
                             "gated": bool(r["h_over_c"] >= 0.2), "pass": bool(abs(err) <= 0.10)})
        summary["validation"] = {"source": str(vpath), "comparison": comp}
    else:
        summary["validation"] = {"status": "PENDING: no published reference data available in the workspace"}

    _figures(sweep, inv, out, df)
    (out / "phase0_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def _figures(sweep: pd.DataFrame, inv: list[dict], out: Path, df: pd.DataFrame) -> None:
    plt = plots.plt
    row0 = sweep.iloc[0].to_dict() if len(sweep) else {}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ok = sweep["cl"].notna()
    axs[0].plot(sweep.loc[ok, "ride_height_over_chord"], -sweep.loc[ok, "cl"], "o-", label="OpenFOAM k-omega SST (coarse)")
    axs[0].plot([i["h"] for i in inv], [-i["cl_inviscid"] for i in inv], "s--", label="inviscid panel + images")
    axs[0].set_ylabel("downforce coefficient -CL")
    axs[1].plot(sweep.loc[ok, "ride_height_over_chord"], sweep.loc[ok, "cd"], "o-")
    axs[1].set_ylabel("CD")
    axs[2].plot(sweep.loc[ok, "ride_height_over_chord"], sweep.loc[ok, "cm"], "o-")
    axs[2].set_ylabel("CM (nose-up +)")
    for ax in axs:
        ax.set_xlabel("ride height h/c")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"Phase 0: ls417 stand-in, AoA {AOA} deg, Re {RE:.2g}, moving ground (NOT validated)", fontsize=10)
    fig.text(0.5, 0.005, f"simpleFoam [{row0.get('solver_version', '?')}] | coarse wall-function meshes "
             f"(~{int(sweep['mesh_cells'].median()) if len(sweep) else 0:,} cells) | kOmegaSST fully turbulent | "
             f"Re = {RE:.3g} | h/c swept", ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "phase0_ride_height_sweep.png", dpi=120)
    plt.close(fig)
