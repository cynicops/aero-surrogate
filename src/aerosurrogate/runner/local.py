"""Run one prepared case: gmshToFoam -> patch types -> checkMesh -> (decompose) -> solve
with force monitoring -> (reconstruct). Every failure is returned as a status, never raised.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import CaseConfig
from ..foam.case import patch_types_for, request_stop, set_patch_types
from ..foam.parsers import (
    foam_version_from_log,
    initial_residual_columns,
    log_failure_reason,
    parse_checkmesh,
    read_force_coeffs,
    read_solver_info,
)
from .launchers import Launcher
from .monitor import diverged, forces_converged, window_stats

# Status values written to the dataset (column `status`)
OK = "ok"  # forces converged (or residualControl reached)
NOT_CONVERGED = "not_converged"  # hit max_iter; values are window means, flagged
FAILED_STATUSES = (
    "geometry_invalid", "mesh_failed", "mesh_conversion_failed", "mesh_check_failed",
    "decompose_failed", "solver_failed", "diverged", "timeout", "reconstruct_failed",
    "postprocess_failed",
)


@dataclass
class RunResult:
    status: str
    failure_reason: str = ""
    converged: bool = False
    stop_reason: str = ""
    n_iter: int = 0
    wall_clock_s: float = 0.0  # solver only
    total_s: float = 0.0  # conversion + checks + solve + reconstruct
    cl: float = float("nan")
    cd: float = float("nan")
    cm: float = float("nan")
    cl_std: float = float("nan")
    cd_std: float = float("nan")
    cm_std: float = float("nan")
    residual_final: float = float("nan")
    checkmesh: dict = field(default_factory=dict)
    foam_version: str = ""
    n_procs: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def _tail(path: Path, n: int = 4000) -> str:
    try:
        return path.read_text(errors="replace")[-n:]
    except FileNotFoundError:
        return ""


def run_openfoam(
    case_dir: str | Path, launcher: Launcher, cfg: CaseConfig, wall_patches: list[str], mesh_only: bool = False
) -> RunResult:
    case_dir = Path(case_dir).resolve()
    logs = case_dir / "logs"
    logs.mkdir(exist_ok=True)
    t_start = time.time()
    res = RunResult(status="running", n_procs=launcher.n_procs)

    def fail(status: str, reason: str) -> RunResult:
        res.status, res.failure_reason = status, reason
        res.total_s = time.time() - t_start
        return res

    # 1) mesh conversion
    rc = launcher.run("gmshToFoam mesh.msh", case_dir, logs / "gmshToFoam.log", timeout=600)
    log_text = _tail(logs / "gmshToFoam.log", 20000)
    res.foam_version = foam_version_from_log(log_text) or launcher.version_label
    if rc != 0:
        return fail("mesh_conversion_failed", log_failure_reason(log_text) or f"gmshToFoam exit {rc}")
    try:
        set_patch_types(case_dir / "constant" / "polyMesh" / "boundary",
                        patch_types_for(wall_patches, cfg.flow.ground))
    except (KeyError, FileNotFoundError) as e:
        return fail("mesh_conversion_failed", f"patch typing: {e}")

    # 2) mesh check
    rc = launcher.run("checkMesh", case_dir, logs / "checkMesh.log", timeout=600)
    cm = parse_checkmesh(_tail(logs / "checkMesh.log", 200000))
    res.checkmesh = cm
    if rc != 0 or not cm["mesh_ok"]:
        return fail("mesh_check_failed", f"checkMesh failed {cm['failed_checks']} checks (exit {rc})")
    if mesh_only:
        res.status = "mesh_ok"
        res.total_s = time.time() - t_start
        return res

    # 2b) optional potential-flow initialisation
    if cfg.solver.potential_init:
        rc = launcher.run("potentialFoam -writep", case_dir, logs / "potentialFoam.log", timeout=900)
        if rc != 0:
            return fail("solver_failed", "potentialFoam: " + (log_failure_reason(_tail(logs / "potentialFoam.log")) or f"exit {rc}"))

    # 3) decompose
    if launcher.n_procs > 1:
        for d in case_dir.glob("processor*"):
            shutil.rmtree(d)
        rc = launcher.run("decomposePar -force", case_dir, logs / "decomposePar.log", timeout=600)
        if rc != 0:
            return fail("decompose_failed", log_failure_reason(_tail(logs / "decomposePar.log")) or f"exit {rc}")

    # 4) solve with monitoring
    conv = cfg.convergence
    t0 = time.time()
    proc = launcher.popen(launcher.parallel("simpleFoam"), case_dir, logs / "simpleFoam.log")
    stop_requested = False
    stop_reason = ""
    try:
        while True:
            try:
                rc = proc.wait(timeout=conv.poll_seconds)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.time() - t0 > conv.wall_clock_limit_s:
                launcher.kill(proc)
                res.wall_clock_s = time.time() - t0
                return fail("timeout", f"wall clock > {conv.wall_clock_limit_s:.0f} s")
            try:
                hist = read_force_coeffs(case_dir)
            except Exception:  # file being written; retry next poll
                continue
            reason = diverged(hist)
            if reason:
                launcher.kill(proc)
                res.wall_clock_s = time.time() - t0
                return fail("diverged", reason)
            if not stop_requested:
                ok, _ = forces_converged(hist, conv)
                if ok:
                    request_stop(case_dir)
                    stop_requested = True
                    stop_reason = "force coefficients converged"
    finally:
        proc._aero_log.close()
    res.wall_clock_s = time.time() - t0
    solver_log = _tail(logs / "simpleFoam.log", 20000)
    if rc != 0:
        return fail("solver_failed", log_failure_reason(solver_log) or f"simpleFoam exit {rc}")
    bad = log_failure_reason(solver_log)
    if bad and "NaN" in bad:
        return fail("diverged", bad)

    hist = read_force_coeffs(case_dir)
    if hist.empty:
        return fail("postprocess_failed", "no force coefficients written")
    s = window_stats(hist, conv.window)
    res.n_iter = s.n_iter
    res.cl, res.cd, res.cm = s.cl, s.cd, s.cm
    res.cl_std, res.cd_std, res.cm_std = s.cl_std, s.cd_std, s.cm_std
    residuals = read_solver_info(case_dir)
    if not residuals.empty:
        cols = initial_residual_columns(residuals)
        res.residual_final = float(np.nanmax(residuals[cols].tail(1).to_numpy())) if cols else float("nan")
    if "SIMPLE solution converged" in solver_log:
        stop_reason = stop_reason or "residualControl reached"
    res.stop_reason = stop_reason or "max_iter reached"
    res.converged = bool(stop_requested or "SIMPLE solution converged" in solver_log)

    # 5) reconstruct the final time for post-processing
    if launcher.n_procs > 1:
        rc = launcher.run("reconstructPar -latestTime", case_dir, logs / "reconstructPar.log", timeout=600)
        if rc != 0:
            return fail("reconstruct_failed", log_failure_reason(_tail(logs / "reconstructPar.log")) or f"exit {rc}")
        for d in case_dir.glob("processor*"):
            shutil.rmtree(d)

    res.status = OK if res.converged else NOT_CONVERGED
    res.total_s = time.time() - t_start
    return res
