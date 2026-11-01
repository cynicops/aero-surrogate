import json

import pandas as pd
import pytest

from aerosurrogate.data.dataset import Dataset, usable
from aerosurrogate.data.schema import REQUIRED_COLUMNS, SchemaError, validate_row


def row(**kw):
    r = {c: 0 for c in REQUIRED_COLUMNS}
    r.update(case_id="a", solver="simpleFoam", solver_version="v", turbulence_model="kOmegaSST",
             geometry_params=json.dumps({"x": 1}), converged=True, source="openfoam", cl=-1.0, cd=0.1, cm=0.1)
    r.update(kw)
    return r


def test_validate_row():
    validate_row(row())
    with pytest.raises(SchemaError, match="source"):
        validate_row(row(source="xfoil"))
    with pytest.raises(SchemaError, match="JSON"):
        validate_row(row(geometry_params="{bad"))
    r = row()
    del r["cl"]
    with pytest.raises(SchemaError, match="missing"):
        validate_row(r)


def test_dataset_roundtrip_and_metadata(tmp_path):
    ds = Dataset(tmp_path, metadata={"campaign": "t"})
    ds.append(row(case_id="a"))
    ds.append(row(case_id="b", converged=False, status="diverged", failure_reason="nan", cl=float("nan")))
    ds.append(row(case_id="c", source="synthetic"))
    df = ds.load()
    assert list(df["case_id"]) == ["a", "b", "c"]
    assert len(ds.load(source="synthetic")) == 1
    assert list(usable(df)["case_id"]) == ["a", "c"]  # failed case kept in data, excluded from training
    meta = ds.metadata()
    assert meta["campaign"] == "t" and "sign_conventions" in meta and "reference_length" in meta


def test_load_rejects_corrupt(tmp_path):
    ds = Dataset(tmp_path)
    pd.DataFrame([{"case_id": "x"}]).to_csv(ds.csv, index=False)
    with pytest.raises(SchemaError):
        ds.load()
