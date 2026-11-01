import numpy as np

from aerosurrogate.geometry.polygon import (
    min_distance, points_in_polygon, polygons_intersect, rotation_matrix, self_intersects, signed_area,
)

SQ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)


def test_area_and_orientation():
    assert signed_area(SQ) == 1.0
    assert signed_area(SQ[::-1]) == -1.0


def test_rotation():
    v = rotation_matrix(np.pi / 2) @ np.array([1.0, 0.0])
    assert np.allclose(v, [0, 1])


def test_distance_and_intersection():
    other = SQ + [2.0, 0.0]
    assert np.isclose(min_distance(SQ, other), 1.0)
    assert not polygons_intersect(SQ, other)
    assert polygons_intersect(SQ, SQ + [0.5, 0.5])
    assert polygons_intersect(SQ, SQ * 0.2 + 0.4)  # containment, no edge crossing
    assert min_distance(SQ, SQ + [0.5, 0.5]) == 0.0


def test_point_in_polygon():
    assert points_in_polygon(np.array([[0.5, 0.5], [2, 2]]), SQ).tolist() == [True, False]


def test_self_intersection():
    bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
    assert self_intersects(bowtie)
    assert not self_intersects(SQ)
