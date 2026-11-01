import numpy as np
import pytest

from aerosurrogate.geometry.airfoil import AirfoilError, load_section, normalise, set_te_thickness, split_surfaces
from aerosurrogate.geometry.polygon import self_intersects, signed_area


@pytest.mark.parametrize("name", ["s1223", "ls417", "naca4412"])
def test_load_normalised_blunt(name):
    c = load_section(name, 80, 0.002)
    ile = int(np.argmin(c[:, 0]))
    assert np.allclose(c[ile], [0, 0], atol=2e-3)
    assert np.allclose(0.5 * (c[0] + c[-1]), [1, 0], atol=1e-6)
    assert np.isclose(c[0, 1] - c[-1, 1], 0.002, atol=1e-6)  # requested TE thickness
    assert signed_area(c) > 0  # counter-clockwise
    assert not self_intersects(c)


def test_normalise_handles_scaled_rotated_reversed_input():
    c = load_section("naca4412", 60, 0.0)
    th = np.deg2rad(10)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    messy = (c * 3.0) @ rot.T + [5, 2]
    back = normalise(messy[::-1])
    assert np.allclose(back, c, atol=1e-6)


def test_te_thickness_ramp_keeps_le():
    c = load_section("naca4412", 60, 0.0)
    t = set_te_thickness(c, 0.01)
    up, lo = split_surfaces(t)
    assert np.isclose(up[0, 1] - lo[-1, 1], 0.01)
    ile = int(np.argmin(c[:, 0]))
    assert np.allclose(t[ile], c[ile])


def test_unknown_airfoil_and_bad_thickness():
    with pytest.raises(Exception):
        load_section("definitely_not_an_airfoil_xyz", 60, 0.001)
    with pytest.raises(AirfoilError):
        set_te_thickness(load_section("naca4412", 60, 0.0), -0.1)
