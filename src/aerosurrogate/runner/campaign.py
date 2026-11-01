"""Run many cases concurrently (each case may itself use MPI) and log every outcome."""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from pathlib import Path

from ..config import CaseConfig, LauncherConfig
from ..data.dataset import Dataset
from ..geometry.wing import WingParams
from .launchers import make_launcher
from .pipeline import base_row, run_case


@dataclass
class Job:
    case_id: str
    params: dict  # WingParams fields
    config: dict  # CaseConfig.to_dict()


def _worker(job: Job, launcher_cfg: dict, out_root: str, make_report: bool, mesh_only: bool) -> dict:
    cfg = CaseConfig.from_dict(job.config)
    params = WingParams(**job.params)
    launcher = make_launcher(LauncherConfig(**launcher_cfg))
    try:
        return run_case(job.case_id, params, cfg, launcher, out_root, make_report=make_report, mesh_only=mesh_only)
    except Exception as e:  # never lose a case: log it as failed
        row = base_row(job.case_id, params, cfg, launcher)
        row.update(status="pipeline_error", failure_reason=f"{type(e).__name__}: {e}",
                   notes=traceback.format_exc()[-500:])
        return row


def run_campaign(
    jobs: list[Job],
    launcher_cfg: LauncherConfig,
    out_root: str | Path,
    dataset: Dataset | None = None,
    workers: int = 2,
    make_report: bool = True,
    mesh_only: bool = False,
    progress=print,
) -> list[dict]:
    out_root = str(Path(out_root).resolve())
    lc = vars(launcher_cfg).copy()
    rows: list[dict] = []
    ctx = mp.get_context("spawn")  # gmsh keeps global state: isolate it per process
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, j, lc, out_root, make_report, mesh_only): j for j in jobs}
        for f in cf.as_completed(futs):
            row = f.result()
            row.pop("checkmesh", None) if not mesh_only else None
            rows.append(row)
            if dataset is not None and not mesh_only:
                dataset.append(row)
            progress(f"[{len(rows)}/{len(jobs)}] {row['case_id']}: {row['status']} "
                     f"cl={row.get('cl', float('nan')):.4f} cd={row.get('cd', float('nan')):.5f} "
                     f"cells={row.get('mesh_cells')} t={row.get('wall_clock_s')}s {row.get('failure_reason', '')}")
    return rows
