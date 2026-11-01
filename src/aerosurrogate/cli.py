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

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
