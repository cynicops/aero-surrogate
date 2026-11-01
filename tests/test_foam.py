import shutil
from pathlib import Path

import numpy as np
import pytest

from aerosurrogate.config import CaseConfig
from aerosurrogate.foam.case import (
    inflow_turbulence, patch_types_for, render_context, request_stop, set_patch_types, write_case,
)
from aerosurrogate.foam.parsers import (
    foam_version_from_log, log_failure_reason, parse_checkmesh, read_dat, read_patch_values, read_raw_surface,
)
from aerosurrogate.geometry import build_wing
from aerosurrogate.mesh.gmsh_mesher import MeshInfo

FIX = Path(__file__).parent / "fixtures"


def _fake_mesh(tmp_path):
    msh = tmp_path / "x.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    return MeshInfo(str(msh), 100, 10, 90, 1e-3, 3, 4e-3, 1e-3, 0.05, (-5, 0, 10, 5), "")


def test_render_case_files(tmp_path, three_elements, nominal_params):
    cfg = CaseConfig(elements=three_elements)
    w = build_wing(three_elements, nominal_params)
    mesh = _fake_mesh(tmp_path)
    ctx = render_context("c1", w, mesh, cfg, n_procs=2)
    case = write_case(tmp_path / "case", ctx, mesh.msh_path)
    cd = (case / "system/controlDict").read_text()
    assert "patches         (element0 element1 element2);" in cd
    assert "pitchAxis       (0 0 -1);" in cd
    assert f"Aref            {0.05}" in cd
    assert "forceCoeffs_element2" in cd
    tp = (case / "constant/transportProperties").read_text()
    assert "nu              2e-06;" in tp
    assert "kOmegaSST" in (case / "constant/turbulenceProperties").read_text()
    u = (case / "0/U").read_text()
    assert "ground  { type fixedValue; value uniform (1.0 0 0); }" in u  # moving ground
    assert (case / "mesh.msh").exists()
    assert "numberOfSubdomains 2;" in (case / "system/decomposeParDict").read_text()


def test_slip_ground_option(tmp_path, three_elements, nominal_params):
    cfg = CaseConfig(elements=three_elements)
    cfg.flow.ground = "slip"
    w = build_wing(three_elements, nominal_params)
    mesh = _fake_mesh(tmp_path)
    case = write_case(tmp_path / "c", render_context("c", w, mesh, cfg, 1), mesh.msh_path)
    assert "ground  { type slip; }" in (case / "0/U").read_text()
    assert "ground  { type zeroGradient; }" in (case / "0/k").read_text()
    assert patch_types_for(w.names, "slip")["ground"] == "patch"


def test_inflow_turbulence():
    k, om = inflow_turbulence(1.0, 0.01, 10.0, 1e-5)
    assert np.isclose(k, 1.5e-4) and np.isclose(om, 1.5e-4 / 1e-4)


def test_set_patch_types(tmp_path):
    b = tmp_path / "boundary"
    shutil.copy(FIX / "boundary_gmshToFoam", b)
    set_patch_types(b, patch_types_for(["element0"], "moving"))
    t = b.read_text()
    assert "physicalType" not in t
    assert t.count("type            empty;") == 2
    assert t.count("inGroups        1(wall);") == 2  # element0 + ground
    with pytest.raises(KeyError):
        shutil.copy(FIX / "boundary_gmshToFoam", b)
        set_patch_types(b, {"inlet": "patch"})  # unexpected patches present


def test_request_stop(tmp_path):
    (tmp_path / "system").mkdir()
    (tmp_path / "system/controlDict").write_text("stopAt          endTime;\nendTime 10;\n")
    request_stop(tmp_path)
    assert "stopAt          writeNow;" in (tmp_path / "system/controlDict").read_text()


def test_read_force_coefficients_fixture():
    df = read_dat(FIX / "coefficient.dat")
    assert {"Time", "Cd", "Cl", "CmPitch"} <= set(df.columns)
    assert df["Time"].iloc[0] == 1


def test_read_solver_info_fixture():
    df = read_dat(FIX / "solverInfo.dat")
    assert "p_initial" in df.columns and "Ux_initial" in df.columns
    assert df["p_initial"].iloc[0] == 1.0


def test_parse_checkmesh_fixture():
    r = parse_checkmesh((FIX / "checkMesh.log").read_text())
    assert r["mesh_ok"] and r["failed_checks"] == 0
    assert r["cells"] > 1000 and r["max_non_orthogonality"] > 0 and r["min_volume"] > 0


def test_parse_checkmesh_failure_text():
    r = parse_checkmesh("    cells:            10\n ***Zero or negative cell volume detected.\nFailed 1 mesh checks.\n")
    assert not r["mesh_ok"] and r["failed_checks"] == 1


def test_patch_values_fixture():
    assert read_patch_values(FIX / "yPlus", "element0").tolist() == [49.1, 12.5, 3.25]
    assert read_patch_values(FIX / "yPlus", "ground").tolist() == [7.5]
    assert read_patch_values(FIX / "yPlus", "inlet").tolist() == [1.5, 2.5]
    with pytest.raises(KeyError):
        read_patch_values(FIX / "yPlus", "nope")


def test_raw_surface_fixture():
    df = read_raw_surface(FIX / "p_element0.raw")
    assert list(df.columns) == ["x", "y", "z", "p"] and len(df) >= 3


@pytest.mark.parametrize(
    "text, expect",
    [
        ("trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\nEnd\n", None),
        ("#0  Foam::error::printStack\n#1  Foam::sigFpe::sigHandler(int)\n", "floating point"),
        ("--> FOAM FATAL ERROR: (openfoam-2512)\ncannot find file\n", "FOAM FATAL ERROR"),
        ("Cl = nan\n", "NaN"),
        ("bounding omega, min: -1 max: 2\n", None),
    ],
)
def test_log_failure_reason(text, expect):
    r = log_failure_reason(text)
    assert (r is None) if expect is None else (expect in r)


def test_version_from_log():
    assert foam_version_from_log("Build  : _87ed40d2-20251219 OPENFOAM=2512 version=2512\n") == \
        "OpenFOAM v2512 (build _87ed40d2-20251219)"
