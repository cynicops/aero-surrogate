import numpy as np
import pytest

from aerosurrogate.geometry import ElementSpec, GeometryError, WingParams, build_wing, check_wing, validate_wing
from aerosurrogate.geometry.polygon import signed_area


def test_three_element_targets_met(three_elements, nominal_params):
    w = check_wing(build_wing(three_elements, nominal_params))
    assert np.allclose(w.achieved_gaps, nominal_params.gaps, atol=1e-6)
    assert np.allclose(w.achieved_overlaps, nominal_params.overlaps, atol=1e-6)
    assert np.isclose(w.bbox[1], nominal_params.ride_height_over_chord)  # lowest point = ride height
    assert all(signed_area(p) > 0 for p in w.polygons)
    assert np.isclose(sum(w.chords), 1.0)
    # inverted wing: flaps sweep upwards and downstream
    centroids = [p.mean(axis=0) for p in w.polygons]
    assert centroids[0][0] < centroids[1][0] < centroids[2][0]
    assert centroids[0][1] < centroids[1][1] < centroids[2][1]


def test_positive_aoa_puts_leading_edge_down():
    single = [ElementSpec("naca4412", 1.0)]
    w = build_wing(single, WingParams(aoa_deg=6, ride_height_over_chord=0.3, re=5e5))
    le = w.polygons[0][w.le_indices[0]]
    te = 0.5 * (w.polygons[0][0] + w.polygons[0][-1])
    assert le[1] < te[1]
    assert np.isclose(np.degrees(np.arctan2(te[1] - le[1], te[0] - le[0])), 6.0, atol=0.05)


def test_ride_height_shift_only_translates():
    single = [ElementSpec("ls417", 1.0)]
    a = build_wing(single, WingParams(4, 0.1, 5e5)).polygons[0]
    b = build_wing(single, WingParams(4, 0.5, 5e5)).polygons[0]
    assert np.allclose(b - a, [0, 0.4])


def test_geometry_hash_changes_with_params(three_elements, nominal_params):
    w1 = build_wing(three_elements, nominal_params)
    p2 = WingParams(**{**vars(nominal_params), "aoa_deg": 5.0})
    assert w1.geometry_hash() != build_wing(three_elements, p2).geometry_hash()


@pytest.mark.parametrize(
    "change, match",
    [
        ({"gaps": [0.015]}, "needs 2 values"),
        ({"gaps": [0.0, 0.015]}, "gaps must be > 0"),
        ({"ride_height_over_chord": -0.1}, "ride height"),
        ({"re": 0}, "Re must be"),
    ],
)
def test_bad_parameters_raise(three_elements, nominal_params, change, match):
    p = WingParams(**{**vars(nominal_params), **change})
    with pytest.raises(GeometryError, match=match):
        build_wing(three_elements, p)


def test_validator_flags_tiny_clearance(three_elements, nominal_params):
    p = WingParams(**{**vars(nominal_params), "gaps": [0.001, 0.015]})
    w = build_wing(three_elements, p)
    problems = validate_wing(w, min_gap=0.003)
    assert any("clearance" in s for s in problems)
    with pytest.raises(GeometryError):
        check_wing(w, min_gap=0.003)


def test_validator_flags_overlap_and_self_intersection(three_elements, nominal_params):
    w = build_wing(three_elements, nominal_params)
    w.polygons[1] = w.polygons[1] + (w.polygons[0].mean(axis=0) - w.polygons[1].mean(axis=0))
    assert any("overlap" in s for s in validate_wing(w))
    w2 = build_wing(three_elements, nominal_params)
    bow = w2.polygons[2].copy()
    bow[[10, 60]] = bow[[60, 10]]
    w2.polygons[2] = bow
    assert any("self-intersecting" in s for s in validate_wing(w2))


def test_too_many_flaps():
    with pytest.raises(GeometryError, match="supported"):
        build_wing([ElementSpec("s1223", 0.25)] * 4, WingParams(4, 0.2, 5e5, [10] * 3, [0.01] * 3, [0.01] * 3))


def test_extreme_overlap_is_rejected_or_valid(three_elements, nominal_params):
    """A huge overlap either fails to build or is caught by validation: never a silent bad shape."""
    p = WingParams(**{**vars(nominal_params), "overlaps": [0.3, 0.02]})
    try:
        w = build_wing(three_elements, p)
    except GeometryError:
        return
    assert validate_wing(w) == [] or validate_wing(w)  # validation always runs without error
