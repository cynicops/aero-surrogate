"""Parsers for OpenFOAM output (function-object .dat files, fields, logs, checkMesh)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_dat(path: str | Path) -> pd.DataFrame:
    """Read a postProcessing .dat file. Column names come from the last '#' header line.

    Vector entries like ``(1 2 3)`` are flattened; tabs and 'N/A' are handled.
    """
    path = Path(path)
    header: list[str] | None = None
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            parts = line[1:].replace("\t", " ").split()
            if parts:
                header = parts
            continue
        vals = line.replace("(", " ").replace(")", " ").replace("N/A", "nan").split()
        try:
            rows.append([float(v) for v in vals])
        except ValueError:
            # solverInfo writes solver names (e.g. 'GAMG') and booleans ('true')
            rows.append([_maybe_float(v) for v in vals])
    if not rows:
        return pd.DataFrame(columns=header or [])
    width = max(len(r) for r in rows)
    if header is None or len(header) != width:
        header = (header or [])[:width] + [f"c{i}" for i in range(len(header or []), width)]
    df = pd.DataFrame([r + [np.nan] * (width - len(r)) for r in rows], columns=header)
    return df


def _maybe_float(v: str):
    try:
        return float(v)
    except ValueError:
        return {"true": 1.0, "false": 0.0}.get(v.lower(), np.nan)


def _concat_time_dirs(fo_dir: Path, filename: str) -> pd.DataFrame:
    if not fo_dir.exists():
        return pd.DataFrame()
    frames = []
    for d in sorted(fo_dir.iterdir(), key=lambda p: float(p.name) if _is_num(p.name) else -1):
        f = d / filename
        if f.exists():
            frames.append(read_dat(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    tcol = df.columns[0]
    return df.drop_duplicates(subset=tcol, keep="last").sort_values(tcol).reset_index(drop=True)


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_force_coeffs(case_dir: str | Path, name: str = "forceCoeffs1") -> pd.DataFrame:
    """Coefficient history with columns Time, Cd, Cl, CmPitch (+ others)."""
    return _concat_time_dirs(Path(case_dir) / "postProcessing" / name, "coefficient.dat")


def read_solver_info(case_dir: str | Path, name: str = "residuals") -> pd.DataFrame:
    return _concat_time_dirs(Path(case_dir) / "postProcessing" / name, "solverInfo.dat")


def initial_residual_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_initial")]


_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse_checkmesh(text: str) -> dict:
    """Extract headline numbers from checkMesh output."""
    def num(pattern, cast=float, default=None):
        pattern = pattern.replace("([\\d.eE+-]+)", _FLOAT)
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    failed = num(r"Failed (\d+) mesh checks", int, 0)
    return {
        "cells": num(r"\n\s*cells:\s+(\d+)", int),
        "faces": num(r"\n\s*faces:\s+(\d+)", int),
        "points": num(r"\n\s*points:\s+(\d+)", int),
        "max_non_orthogonality": num(r"Mesh non-orthogonality Max:\s*([\d.eE+-]+)"),
        "avg_non_orthogonality": num(r"Mesh non-orthogonality Max:\s*[\d.eE+-]+\s+average:\s*([\d.eE+-]+)"),
        "max_skewness": num(r"Max skewness =\s*([\d.eE+-]+)"),
        "max_aspect_ratio": num(r"Max aspect ratio =\s*([\d.eE+-]+)"),
        "min_volume": num(r"Min volume =\s*([\d.eE+-]+)"),
        "mesh_ok": ("Mesh OK." in text) and failed == 0,
        "failed_checks": failed,
    }


# The banner line "trapFpe: Floating point exception trapping enabled" is NOT a failure.
_FATAL = [
    (re.compile(r"--> FOAM FATAL (IO )?ERROR"), "FOAM FATAL ERROR"),
    (re.compile(r"sigFpe::sigHandler|Floating point exception \(core dumped\)"),
     "floating point exception (divergence)"),
    (re.compile(r"sigSegv::sigHandler|Segmentation fault"), "segmentation fault"),
    (re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])", re.IGNORECASE), "NaN in solution"),
]


def log_failure_reason(text: str) -> str | None:
    """Return a short reason if the solver log shows a fatal problem, else None."""
    for pat, reason in _FATAL:
        m = pat.search(text)
        if m:
            if reason == "FOAM FATAL ERROR":
                tail = text[m.end(): m.end() + 400].strip().splitlines()
                detail = " ".join(t.strip() for t in tail[:3] if t.strip())
                return f"{reason}: {detail[:200]}"
            return reason
    return None


def read_patch_values(field_file: str | Path, patch: str) -> np.ndarray:
    """Values of a volScalarField on one patch (handles uniform and nonuniform lists)."""
    text = Path(field_file).read_text()
    m = re.search(rf"\n\s*{re.escape(patch)}\s*\{{(.*?)\n\s*\}}", text, re.S)
    if not m:
        raise KeyError(f"patch {patch} not in {field_file}")
    body = m.group(1)
    mu = re.search(r"value\s+uniform\s+([^;]+);", body)
    if mu:
        return np.array([float(mu.group(1))])
    mn = re.search(r"value\s+nonuniform\s+List<scalar>\s*(\d+)\s*\((.*?)\)\s*;", body, re.S)
    if not mn:
        # very short lists are written on one line: 3(1 2 3)
        mn = re.search(r"value\s+nonuniform\s+List<scalar>\s*(\d+)\((.*?)\)\s*;", body, re.S)
    if not mn:
        raise ValueError(f"no value list for patch {patch} in {field_file}")
    vals = np.array(mn.group(2).split(), dtype=float)
    if len(vals) != int(mn.group(1)):
        raise ValueError("value count mismatch")
    return vals


def read_raw_surface(path: str | Path) -> pd.DataFrame:
    """Read a 'raw' surface sample (x y z value...)."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    cols = ["x", "y", "z"] + [f"v{i}" for i in range(data.shape[1] - 3)]
    df = pd.DataFrame(data, columns=cols)
    return df.rename(columns={"v0": "p"})


def latest_time_dir(case_dir: str | Path) -> Path | None:
    dirs = [d for d in Path(case_dir).iterdir() if d.is_dir() and _is_num(d.name) and float(d.name) > 0]
    return max(dirs, key=lambda d: float(d.name)) if dirs else None


def foam_version_from_log(text: str) -> str | None:
    """E.g. 'OpenFOAM v2512 (build _87ed40d2-20251219)' from the log banner."""
    m = re.search(r"Build\s*:\s*(\S+)\s+OPENFOAM=(\d+)", text)
    if m:
        return f"OpenFOAM v{m.group(2)} (build {m.group(1)})"
    m = re.search(r"Version:\s*v?(\d{4})", text)
    return f"OpenFOAM v{m.group(1)}" if m else None
