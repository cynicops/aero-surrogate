"""Command line entry point: ``python -m aerosurrogate <command> ...``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _launcher(path):
    from .config import LauncherConfig

    return LauncherConfig.from_yaml(path)


def cmd_phase0(a):
    from .phase0 import run_phase0

    s = run_phase0(_launcher(a.launcher), a.out, workers=a.workers)
    print(json.dumps({k: s[k] for k in ("sweep", "freestream", "validation")}, indent=1, default=float)[:4000])


def _jobs_from_space(ranges, case_yaml, n, seed, prefix, skip=0):
    from .config import CaseConfig
    from .geometry.wing import WingParams
    from .runner.campaign import Job
    from .sampling import ParameterSpace

    space = ParameterSpace.from_yaml(ranges)
    cfg = CaseConfig.from_yaml(case_yaml)
    if [vars(e) for e in space.elements] != [vars(e) for e in cfg.elements]:
        raise SystemExit("elements in the ranges file and the case config differ")
    df = space.sobol(n, seed=seed, skip=skip)
    jobs = []
    for i, r in df.iterrows():
        p = WingParams.from_flat(r.to_dict(), cfg.n_flaps)
        jobs.append(Job(f"{prefix}{i:04d}", vars(p), cfg.to_dict()))
    return jobs


def cmd_meshcheck(a):
    """Gate 1: N random valid cases must produce meshes that pass checkMesh."""
    import pandas as pd

    from .runner.campaign import run_campaign

    jobs = _jobs_from_space(a.ranges, a.case, a.n, a.seed, "mesh_")
    rows = run_campaign(jobs, _launcher(a.launcher), Path(a.out), None, workers=a.workers,
                        make_report=False, mesh_only=True)
    df = pd.DataFrame([{**{k: r.get(k) for k in ("case_id", "status", "failure_reason", "mesh_cells")},
                        **{f"cm_{k}": v for k, v in (r.get("checkmesh") or {}).items()}} for r in rows])
    df = df.sort_values("case_id")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "meshcheck_summary.csv", index=False)
    n_ok = int((df["status"] == "mesh_ok").sum())
    print(df.to_string(index=False))
    print(f"GATE 1 mesh check: {n_ok}/{len(df)} passed checkMesh")
    return 0 if n_ok == len(df) else 1


def cmd_campaign(a):
    from .data.dataset import Dataset
    from .runner.campaign import run_campaign

    jobs = _jobs_from_space(a.ranges, a.case, a.n, a.seed, a.prefix, skip=a.skip)
    ds = Dataset(a.out, metadata={"ranges_file": a.ranges, "case_config": a.case})
    run_campaign(jobs, _launcher(a.launcher), ds.cases_dir, ds, workers=a.workers, make_report=not a.no_report)


def cmd_case(a):
    import yaml

    from .config import CaseConfig
    from .data.dataset import Dataset
    from .geometry.wing import WingParams
    from .runner.launchers import make_launcher
    from .runner.pipeline import run_case

    cfg = CaseConfig.from_yaml(a.case)
    params = WingParams(**yaml.safe_load(Path(a.params).read_text()))
    ds = Dataset(a.out)
    row = run_case(a.case_id, params, cfg, make_launcher(_launcher(a.launcher)), ds.cases_dir)
    ds.append(row)
    print(json.dumps({k: row.get(k) for k in ("case_id", "status", "failure_reason", "cl", "cd", "cm",
                                              "converged", "wall_clock_s", "mesh_cells")}, default=str))


def cmd_lowfi(a):
    from .workflows import add_lowfi_rows

    print(f"added {add_lowfi_rows(a.dataset, a.workers)} low-fidelity rows")


def cmd_train(a):
    from .workflows import train_and_evaluate

    r = train_and_evaluate(a.dataset, a.ranges, a.out, n_splits=a.splits)
    print(r["summary"].to_string(index=False))


def cmd_active(a):
    from .workflows import al_retrospective_cfd, al_synthetic_benchmark

    if a.dataset:
        r = al_retrospective_cfd(a.dataset, a.ranges, a.out, target_rmse_cl=a.target)
    else:
        r = al_synthetic_benchmark(a.ranges, a.out)
    print(json.dumps(r, indent=1, default=str))


def cmd_optimise(a):
    from .workflows import optimise_and_export

    r = optimise_and_export(a.surrogate, a.ranges, a.out)
    print(json.dumps(r, indent=1, default=str))


def cmd_demo(a):
    """Synthetic end-to-end demo (no OpenFOAM): data -> surrogate -> AL -> optimise -> export."""
    import numpy as np
    import pandas as pd

    from .active.loop import ALConfig
    from .active.synthetic import synthetic_coefficients
    from .data.dataset import Dataset
    from .data.schema import now_utc
    from .sampling import ParameterSpace
    from .surrogate import EnsembleConfig
    from .workflows import al_synthetic_benchmark, optimise_and_export, train_and_evaluate

    out = Path(a.out)
    space = ParameterSpace.from_yaml(a.ranges)
    df = space.sobol(64, seed=0)
    y = synthetic_coefficients(df)
    ds = Dataset(out / "dataset", metadata={"note": "SYNTHETIC demo data, not aerodynamics"})
    for i, r in pd.concat([df, y], axis=1).iterrows():
        lf = {"cl": 1.3 * r.cl + 0.25 * np.sin(r.aoa_deg / 2.0), "cd": 0.6 * r.cd + 0.004 * r.flap2_deg / 35,
              "cm": 1.1 * r.cm - 0.05 * np.cos(r.flap1_deg / 10.0)}  # nonlinear distortion of the truth
        for solver, vals in (("synthetic truth", {"cl": r.cl, "cd": r.cd, "cm": r.cm}),
                             ("synthetic low-fi (biased copy)", lf)):
            row = {"case_id": f"syn_{i:03d}", "solver": solver, "solver_version": "n/a",
                   "turbulence_model": "n/a", "mesh_cells": 0, "y_plus_max": float("nan"),
                   "geometry_params": json.dumps({k: float(r[k]) for k in space.names}),
                   "aoa_deg": r.aoa_deg, "re": r.re, "ride_height_over_chord": r.ride_height_over_chord,
                   "cl": vals["cl"], "cd": vals["cd"], "cm": vals["cm"], "converged": True, "residual_final": 0.0,
                   "wall_clock_s": 0.0, "source": "synthetic", "status": "ok", "created_utc": now_utc(),
                   "notes": "SYNTHETIC"}
            row.update({k: float(r[k]) for k in space.names if k not in row})
            ds.append(row)
    fast = EnsembleConfig(n_members=3, hidden=(32, 32), epochs=500, patience=80)
    r3 = train_and_evaluate(out / "dataset", a.ranges, out / "phase3", n_splits=2, ens=fast,
                            truth=("solver", "synthetic truth"), lowfi=("solver", "synthetic low-fi (biased copy)"))
    print(r3["summary"].to_string(index=False))
    r4 = al_synthetic_benchmark(a.ranges, out / "phase4", seeds=(0,), pool_size=128,
                                al=ALConfig(n_init=8, batch_size=8, budget=32), ens=fast)
    print("AL:", r4["verdict"], r4["runs_to_target_mean_curve"])
    r5 = optimise_and_export(out / "phase3" / "surrogate_cfd_only", a.ranges, out / "phase5",
                             source_note="SYNTHETIC demo surrogate")
    print("export:", r5["aero_map_csv"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="aerosurrogate", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("phase0", help="Phase 0 single-element ground-effect sweep + GCI")
    s.add_argument("--launcher", default="configs/openfoam/docker_v2512.yaml")
    s.add_argument("--out", default="runs/phase0")
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(func=cmd_phase0)

    def common(s, n_default):
        s.add_argument("--launcher", default="configs/openfoam/docker_v2512.yaml")
        s.add_argument("--ranges", default="configs/ranges/front3.yaml")
        s.add_argument("--case", default="configs/case_coarse.yaml")
        s.add_argument("--n", type=int, default=n_default)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--workers", type=int, default=2)

    s = sub.add_parser("meshcheck", help="Gate 1: mesh N random cases and run checkMesh")
    common(s, 20)
    s.add_argument("--out", default="runs/gate1_meshcheck")
    s.set_defaults(func=cmd_meshcheck, seed=101)

    s = sub.add_parser("campaign", help="Sobol-sample N cases and run them (appends to a dataset)")
    common(s, 16)
    s.add_argument("--out", default="runs/campaign")
    s.add_argument("--prefix", default="cfd_")
    s.add_argument("--skip", type=int, default=0, help="skip the first SKIP Sobol points (to extend a campaign)")
    s.add_argument("--no-report", action="store_true")
    s.set_defaults(func=cmd_campaign)

    s = sub.add_parser("case", help="run one case: --params file.yaml with WingParams fields")
    s.add_argument("--launcher", default="configs/openfoam/docker_v2512.yaml")
    s.add_argument("--case", default="configs/case_coarse.yaml")
    s.add_argument("--params", default="configs/demo_case_params.yaml")
    s.add_argument("--case-id", dest="case_id", default="demo_case")
    s.add_argument("--out", default="runs/demo_cfd")
    s.set_defaults(func=cmd_case)

    s = sub.add_parser("lowfi", help="add NeuralFoil/AeroSandbox low-fidelity rows for every CFD case")
    s.add_argument("--dataset", required=True)
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(func=cmd_lowfi)

    s = sub.add_parser("train", help="Phase 3: train surrogates, compare with the low-fi baseline")
    s.add_argument("--dataset", required=True)
    s.add_argument("--ranges", required=True)
    s.add_argument("--out", default="runs/phase3")
    s.add_argument("--splits", type=int, default=5)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("active", help="Phase 4: AL vs space filling (synthetic, or --dataset for retrospective CFD)")
    s.add_argument("--ranges", default="configs/ranges/front3_reduced.yaml")
    s.add_argument("--dataset", default=None)
    s.add_argument("--target", type=float, default=0.05)
    s.add_argument("--out", default="runs/phase4")
    s.set_defaults(func=cmd_active)

    s = sub.add_parser("optimise", help="Phase 5: optimise on a saved surrogate and export an aero map")
    s.add_argument("--surrogate", required=True)
    s.add_argument("--ranges", required=True)
    s.add_argument("--out", default="runs/phase5")
    s.set_defaults(func=cmd_optimise)

    s = sub.add_parser("demo", help="synthetic end-to-end demo, no OpenFOAM needed")
    s.add_argument("--ranges", default="configs/ranges/front3_reduced.yaml")
    s.add_argument("--out", default="runs/demo")
    s.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
