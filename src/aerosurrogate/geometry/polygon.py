"""Small, dependency-free 2D polygon utilities (vectorised numpy).

Polygons are (N, 2) arrays of vertices; the closing edge (last -> first) is implied.
"""

from __future__ import annotations

import numpy as np


def rotation_matrix(angle_rad: float) -> np.ndarray:
    """Counter-clockwise rotation matrix."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def segments(poly: np.ndarray) -> np.ndarray:
    """Return the closed polygon's edges as an (N, 2, 2) array."""
    return np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)


def signed_area(poly: np.ndarray) -> float:
    """Shoelace area; positive for counter-clockwise ordering."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_segment_distance(points: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Distance from each point (M, 2) to each segment (N, 2, 2) -> (M, N)."""
    a = segs[:, 0, :][None, :, :]
    b = segs[:, 1, :][None, :, :]
    p = points[:, None, :]
    ab = b - a
    denom = np.sum(ab * ab, axis=-1)
    denom = np.where(denom == 0.0, 1.0, denom)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / denom, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (
        c[..., 0] - a[..., 0]
    )


def segments_intersect_matrix(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Boolean (N1, N2) matrix: proper or touching intersection between segments."""
    p1 = s1[:, None, 0, :]
    p2 = s1[:, None, 1, :]
    q1 = s2[None, :, 0, :]
    q2 = s2[None, :, 1, :]
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    eps = 1e-14
    z1, z2, z3, z4 = (np.abs(o) <= eps for o in (o1, o2, o3, o4))
    proper = (np.sign(o1) * np.sign(o2) < 0) & (np.sign(o3) * np.sign(o4) < 0)
    # touching / collinear cases: only if the zero-orientation point lies on the other segment
    touch = (
        (z1 & _on_segment(p1, p2, q1)) | (z2 & _on_segment(p1, p2, q2))
        | (z3 & _on_segment(q1, q2, p1)) | (z4 & _on_segment(q1, q2, p2))
    )
    return proper | touch


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Is c (already known collinear with a-b) within the bounding box of a-b?"""
    return (
        (np.minimum(a[..., 0], b[..., 0]) <= c[..., 0]) & (c[..., 0] <= np.maximum(a[..., 0], b[..., 0]))
        & (np.minimum(a[..., 1], b[..., 1]) <= c[..., 1]) & (c[..., 1] <= np.maximum(a[..., 1], b[..., 1]))
    )


def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray casting test for each point."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    xi, yi = poly[:, 0][None, :], poly[:, 1][None, :]
    xj, yj = np.roll(poly[:, 0], 1)[None, :], np.roll(poly[:, 1], 1)[None, :]
    cond = (yi > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
    hits = cond & (x < x_cross)
    return (np.sum(hits, axis=1) % 2) == 1


def polygons_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    """True if the two closed polygons overlap (edge crossing or containment)."""
    if segments_intersect_matrix(segments(a), segments(b)).any():
        return True
    return bool(points_in_polygon(a[:1], b)[0] or points_in_polygon(b[:1], a)[0])


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance between the boundaries of two polygons (0 if they intersect)."""
    if polygons_intersect(a, b):
        return 0.0
    d1 = point_segment_distance(a, segments(b)).min()
    d2 = point_segment_distance(b, segments(a)).min()
    return float(min(d1, d2))


def self_intersects(poly: np.ndarray) -> bool:
    """True if any two non-adjacent edges of the closed polygon intersect."""
    segs = segments(poly)
    n = len(segs)
    m = segments_intersect_matrix(segs, segs)
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    adjacent = (diff <= 1) | (diff == n - 1)
    return bool((m & ~adjacent).any())
