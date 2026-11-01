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


def main(argv=None):
    p = argparse.ArgumentParser(prog="aerosurrogate", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("phase0", help="Phase 0 single-element ground-effect sweep + GCI")
    s.add_argument("--launcher", default="configs/openfoam/docker_v2512.yaml")
    s.add_argument("--out", default="runs/phase0")
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(func=cmd_phase0)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
