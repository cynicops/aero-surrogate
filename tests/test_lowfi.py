import numpy as np
import pytest

from aerosurrogate.baseline.lowfi import inviscid, lowfi_coefficients, lowfi_row, lowfi_wing, neuralfoil_section
from aerosurrogate.data.schema import validate_row
from aerosurrogate.geometry import ElementSpec, WingParams

SINGLE = [ElementSpec("ls417", 1.0, 0.001)]


@pytest.mark.slow
def test_inverted_wing_gives_downforce_and_ground_effect_increases_it():
    far = inviscid(lowfi_wing(SINGLE, WingParams(4, 5.0, 5e5)), ground=False)
    near = inviscid(lowfi_wing(SINGLE, WingParams(4, 0.2, 5e5)), ground=True)
    assert far["cl"] < 0 and near["cl"] < far["cl"]  # more downforce near the ground
    assert far["cm"] > 0  # inverted cambered section: nose-up moment about c/4
    assert abs(far["cd_pressure"]) < 0.01  # inviscid: ~no pressure drag (d'Alembert)
    assert np.isclose(far["cl"], far["cl_circulation"], rtol=0.02)  # no ground: KJ == pressure


@pytest.mark.slow
def test_freestream_matches_neuralfoil_sign_and_magnitude():
    inv = inviscid(lowfi_wing(SINGLE, WingParams(4, 5.0, 5e5)), ground=False)
    nf = neuralfoil_section("ls417", 0.001, 4.0, 5e5)
    assert nf["CL"][0] > 0  # NeuralFoil works in the lift-up frame
    assert 0.8 < -inv["cl"] / nf["CL"][0] < 1.3  # inviscid >= viscous lift, same order


@pytest.mark.slow
def test_lowfi_row_is_schema_valid():
    w = lowfi_wing(SINGLE, WingParams(4, 0.3, 5e5))
    r = lowfi_row("x1", w)
    validate_row(r)
    assert r["source"] == "neuralfoil" and r["cd"] > 0
    out = lowfi_coefficients(w, 5e5)
    assert len(out["stalled_elements"]) == 1
