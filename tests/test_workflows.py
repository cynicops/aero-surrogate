import json

import numpy as np
import pandas as pd

from aerosurrogate.active.synthetic import synthetic_coefficients
from aerosurrogate.data.dataset import Dataset
from aerosurrogate.data.schema import REQUIRED_COLUMNS
from aerosurrogate.sampling import ParameterSpace
from aerosurrogate.surrogate import EnsembleConfig
from aerosurrogate.workflows import train_and_evaluate

RANGES = "configs/ranges/front3_reduced.yaml"


def test_train_and_evaluate_on_synthetic(tmp_path):
    sp = ParameterSpace.from_yaml(RANGES)
    df = sp.sobol(32, seed=0)
    y = synthetic_coefficients(df)
    ds = Dataset(tmp_path / "d")
    for i, r in pd.concat([df, y], axis=1).iterrows():
        for solver, f in (("synthetic truth", 1.0), ("synthetic low-fi", 1.2)):
            row = {c: 0 for c in REQUIRED_COLUMNS}
            row.update(case_id=f"c{i}", solver=solver, solver_version="-", turbulence_model="-",
                       geometry_params=json.dumps({}), aoa_deg=r.aoa_deg, re=r.re,
                       ride_height_over_chord=r.ride_height_over_chord, cl=r.cl * f, cd=r.cd, cm=r.cm,
                       converged=True, source="synthetic", flap1_deg=r.flap1_deg, flap2_deg=r.flap2_deg)
            ds.append(row)
    res = train_and_evaluate(tmp_path / "d", RANGES, tmp_path / "o", n_splits=2,
                             ens=EnsembleConfig(n_members=2, hidden=(16,), epochs=200, patience=50),
                             truth=("solver", "synthetic truth"), lowfi=("solver", "synthetic low-fi"))
    s = res["summary"]
    assert set(s.model) == {"surrogate_cfd_only", "surrogate_multifidelity", "baseline_lowfi_only",
                            "baseline_lowfi_linear_corrected"}
    assert (tmp_path / "o" / "parity_split0.png").exists()
    assert (tmp_path / "o" / "surrogate_cfd_only" / "members.pt").exists()
    m = pd.read_csv(tmp_path / "o" / "metrics_all_splits.csv")
    assert (m.n_test > 0).all() and np.isfinite(m.rmse).all()
