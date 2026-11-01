"""Export CL/CD/CM maps for a later lap-time simulator. Format: docs/coefficient_format.md.

Output: ``<name>.csv`` (one row per grid point) + ``<name>.json`` (metadata). The JSON holds the
fixed design, units, sign conventions, the surrogate's training range and an
``extrapolated`` flag per row, so a downstream tool can never assume what it means.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..data.schema import default_metadata, now_utc
from ..sampling import ParameterSpace
from ..surrogate import EnsembleSurrogate, features

FORMAT_VERSION = "aero-map/1"


def export_aero_map(
    sur: EnsembleSurrogate,
    space: ParameterSpace,
    design: dict,
    grid: dict[str, list[float]],
    out: str | Path,
    source_note: str,
) -> tuple[Path, Path]:
    """Evaluate the surrogate on a grid (e.g. ride height x AoA x Re) around a fixed design."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        r = dict(design)
        r.update(dict(zip(keys, combo)))
        rows.append(r)
    df = pd.DataFrame(rows)
    missing = [n for n in sur.input_names if n not in df.columns]
    if missing:
        raise ValueError(f"design/grid lacks surrogate inputs: {missing}")
    mu, sd = sur.predict(features(df, sur.input_names))
    for j, n in enumerate(["cl", "cd", "cm"]):
        df[n], df[f"{n}_std"] = mu[:, j], sd[:, j]
    df["downforce_coeff"] = -df["cl"]
    df["extrapolated"] = ~space.contains(df)
    cols = keys + ["cl", "cl_std", "cd", "cd_std", "cm", "cm_std", "downforce_coeff", "extrapolated"]
    csv = out.with_suffix(".csv")
    df[cols].to_csv(csv, index=False, float_format="%.6g")
    meta = {
        "format": FORMAT_VERSION,
        "created_utc": now_utc(),
        "generator": f"aerosurrogate {__version__}",
        "columns": {
            "cl, cd, cm": "ensemble-mean prediction (2D, per unit span)",
            "*_std": "ensemble standard deviation (epistemic only; excludes CFD modelling error)",
            "downforce_coeff": "-cl",
            "extrapolated": "true if the point lies outside the surrogate's training ranges",
        },
        "grid_axes": keys,
        "fixed_design": design,
        "training_ranges": {d.name: [d.low, d.high, d.scale] for d in space.dims},
        "surrogate_inputs": sur.input_names,
        "source": source_note,
        "conventions": default_metadata(),
        "use_in_3d": "These are 2D sectional coefficients. A lap-time model needs span/area, 3D and "
                     "endplate corrections, and validation against your STAR-CCM+ or track data.",
    }
    js = out.with_suffix(".json")
    js.write_text(json.dumps(meta, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    return csv, js
