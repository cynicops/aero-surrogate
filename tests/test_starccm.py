"""Tests the Python side of the STAR-CCM+ adapter against a SYNTHETIC fixture.
The macro and the real STAR-CCM+ CSV layout are UNTESTED."""
from pathlib import Path

import pytest

from aerosurrogate.starccm.adapter import UNTESTED, import_result, write_macro

FIX = Path(__file__).parent / "fixtures" / "starccm"


def test_flagged_untested():
    assert UNTESTED is True


def test_macro_renders(tmp_path):
    p = write_macro("case-1", {"aoa_deg": 4, "ride_height_over_chord": 0.2}, tmp_path, "/tmp/out.csv")
    t = p.read_text()
    assert "UNTESTED" in t and 'setParam(gpm, "aoa_deg", 4.0);' in t and "class RunAeroCase_case_1" in t


def test_import_converts_signs_into_schema():
    row = import_result(FIX / "case_s001.csv", FIX / "case_s001.json")
    assert row["source"] == "starccm" and row["solver"] == "STAR-CCM+"
    assert row["cl"] == pytest.approx(-1.85)  # lift-up convention: downforce is negative
    assert row["cm"] == pytest.approx(0.42)
    assert row["gap1"] == 0.015 and row["converged"] is True


def test_import_missing_report(tmp_path):
    (tmp_path / "x.csv").write_text("key,value\nreport:CL,1\n")
    with pytest.raises(KeyError):
        import_result(tmp_path / "x.csv", FIX / "case_s001.json")
