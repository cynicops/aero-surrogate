"""Parametric multi-element wing in ground effect (main plane + 0..2 flaps).

Conventions (also in docs/conventions.md):

* Ground frame: x downstream (freestream along +x), y up, ground plane at y = 0.
* Lengths are non-dimensional with the reference chord c_ref = sum of element chords = 1.
* The wing is assembled in the lift-up "aerofoil frame", then mirrored (y -> -y) so that it
  produces downforce, then translated so its lowest point sits at y = h (ride height).
* ``aoa_deg``: main-element incidence. Positive = leading edge down in the ground frame
  (more downforce). The assembly rotates about the main-element LE.
* ``flap_angles_deg[i]``: deflection of element i+1 relative to the chord of element i.
  Positive = trailing edge towards the suction side (up, in the ground frame).
* ``gaps[i]``: minimum surface-to-surface distance between element i and element i+1
  (fraction of c_ref). The flap position normal to the previous chord is solved so that
  this distance is met exactly.
* ``overlaps[i]``: distance, along the chord of element i, from the flap LE to the TE
  midpoint of element i (fraction of c_ref). Positive = flap LE upstream of that TE.
* Ride height ``h/c``: ground to the lowest point of the whole assembly, over c_ref.
* Polygons are counter-clockwise; each starts and ends at a TE corner (blunt TE).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .airfoil import load_section
from .polygon import min_distance, polygons_intersect, rotation_matrix, self_intersects


class GeometryError(ValueError):
    """Raised for an invalid or unbuildable wing configuration."""


@dataclass(frozen=True)
class ElementSpec:
    airfoil: str
    chord: float  # fraction of c_ref (chords are renormalised to sum to 1)
    te_thickness: float = 0.001  # fraction of this element's chord (0.4 % shed vortices in Phase 0)


@dataclass
class WingParams:
    aoa_deg: float
    ride_height_over_chord: float
    re: float
    flap_angles_deg: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    overlaps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_flat(cls, d: dict, n_flaps: int) -> "WingParams":
        """Build from flat sampling columns (flap1_deg, gap1, overlap1, ...)."""
        return cls(
            aoa_deg=float(d["aoa_deg"]),
            ride_height_over_chord=float(d["ride_height_over_chord"]),
            re=float(d["re"]),
            flap_angles_deg=[float(d[f"flap{i + 1}_deg"]) for i in range(n_flaps)],
            gaps=[float(d[f"gap{i + 1}"]) for i in range(n_flaps)],
            overlaps=[float(d[f"overlap{i + 1}"]) for i in range(n_flaps)],
        )

    def to_flat(self) -> dict:
        out = {
            "aoa_deg": self.aoa_deg,
            "ride_height_over_chord": self.ride_height_over_chord,
            "re": self.re,
        }
        for i, (a, g, o) in enumerate(zip(self.flap_angles_deg, self.gaps, self.overlaps)):
            out[f"flap{i + 1}_deg"] = a
            out[f"gap{i + 1}"] = g
            out[f"overlap{i + 1}"] = o
        return out


@dataclass
class Wing:
    """An assembled wing in the ground frame."""

    polygons: list[np.ndarray]  # one (N, 2) ccw polygon per element
    names: list[str]  # patch names: element0, element1, ...
    chords: list[float]  # fraction of c_ref
    ride_height: float  # = h/c_ref (c_ref = 1)
    moment_ref: np.ndarray  # main-element quarter-chord point, ground frame
    le_indices: list[int]  # LE vertex index per polygon
    te_corners: list[tuple[int, int]]  # (first, last) vertex index per polygon
    achieved_gaps: list[float]
    achieved_overlaps: list[float]
    element_angles_deg: list[float]  # chord angle of each element, aerofoil frame, + = nose up
    specs: list[ElementSpec]
    params: WingParams

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        allp = np.vstack(self.polygons)
        return float(allp[:, 0].min()), float(allp[:, 1].min()), float(
            allp[:, 0].max()
        ), float(allp[:, 1].max())

    def geometry_json(self) -> str:
        """Everything needed to rebuild the geometry (stored in the dataset)."""
        return json.dumps(
            {
                "elements": [asdict(s) for s in self.specs],
                "params": self.params.to_dict(),
                "achieved_gaps": [round(g, 8) for g in self.achieved_gaps],
                "achieved_overlaps": [round(o, 8) for o in self.achieved_overlaps],
            },
            sort_keys=True,
        )

    def geometry_hash(self) -> str:
        return hashlib.sha1(self.geometry_json().encode()).hexdigest()[:12]


def _place(section: np.ndarray, chord: float, angle_rad: float, le: np.ndarray) -> np.ndarray:
    return (section * chord) @ rotation_matrix(angle_rad).T + le


def _te_mid(poly: np.ndarray) -> np.ndarray:
    return 0.5 * (poly[0] + poly[-1])


def _solve_offset(
    prev: np.ndarray,
    flap_local: np.ndarray,
    chord: float,
    angle: float,
    te: np.ndarray,
    t_hat: np.ndarray,
    n_hat: np.ndarray,
    overlap: float,
    gap: float,
) -> tuple[np.ndarray, float]:
    """Find the offset s (along -n_hat, i.e. towards the pressure side) giving the gap."""

    base = (flap_local * chord) @ rotation_matrix(angle).T
    p_lo, p_hi = prev.min(axis=0), prev.max(axis=0)

    def place(s: float) -> np.ndarray:
        return base + (te - overlap * t_hat - s * n_hat)

    def too_close(s: float) -> bool:
        flap = place(s)
        # cheap bounding-box lower bound first
        f_lo, f_hi = flap.min(axis=0), flap.max(axis=0)
        sep = np.maximum(0.0, np.maximum(f_lo - p_hi, p_lo - f_hi))
        if float(np.hypot(*sep)) >= gap:
            return False
        return min_distance(prev, flap) < gap

    # March from far below upwards; the first crossing of `gap` is the physical slot.
    s_hi = 0.6
    if too_close(s_hi):
        raise GeometryError("cannot open the requested gap (flap too large or misplaced)")
    step = 0.005
    s = s_hi
    while s > -0.2:
        s_next = s - step
        if too_close(s_next):
            lo, hi = s_next, s  # too_close(lo) and not too_close(hi)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if too_close(mid):
                    lo = mid
                else:
                    hi = mid
            return place(hi), hi
        s = s_next
    raise GeometryError("gap target never reached (flap does not approach previous element)")


def build_wing(
    elements: list[ElementSpec],
    params: WingParams,
    n_per_side: int = 120,
) -> Wing:
    """Assemble the wing. Raises GeometryError if the configuration cannot be built."""
    n_flaps = len(elements) - 1
    if n_flaps < 0 or n_flaps > 2:
        raise GeometryError("supported: main plane + 0, 1 or 2 flaps")
    for name, seq in (
        ("flap_angles_deg", params.flap_angles_deg),
        ("gaps", params.gaps),
        ("overlaps", params.overlaps),
    ):
        if len(seq) != n_flaps:
            raise GeometryError(f"{name} needs {n_flaps} values, got {len(seq)}")
    if params.ride_height_over_chord <= 0:
        raise GeometryError("ride height must be > 0")
    if any(g <= 0 for g in params.gaps):
        raise GeometryError("gaps must be > 0")
    if params.re <= 0:
        raise GeometryError("Re must be > 0")

    total = sum(e.chord for e in elements)
    chords = [e.chord / total for e in elements]
    sections = [load_section(e.airfoil, n_per_side, e.te_thickness) for e in elements]

    # 1) aerofoil frame, main chord along +x, no incidence yet
    polys = [_place(sections[0], chords[0], 0.0, np.zeros(2))]
    angles = [0.0]  # chord angle, ccw positive (aerofoil frame)
    for i in range(1, len(elements)):
        prev = polys[-1]
        prev_angle = angles[-1]
        t_hat = np.array([np.cos(prev_angle), np.sin(prev_angle)])
        n_hat = np.array([-np.sin(prev_angle), np.cos(prev_angle)])
        angle = prev_angle - np.deg2rad(params.flap_angles_deg[i - 1])  # TE down
        poly, _ = _solve_offset(
            prev,
            sections[i],
            chords[i],
            angle,
            _te_mid(prev),
            t_hat,
            n_hat,
            params.overlaps[i - 1],
            params.gaps[i - 1],
        )
        polys.append(poly)
        angles.append(angle)

    # 2) incidence (nose-up in the aerofoil frame == clockwise rotation about main LE)
    rot = rotation_matrix(-np.deg2rad(params.aoa_deg))
    polys = [p @ rot.T for p in polys]
    angles = [a - np.deg2rad(params.aoa_deg) for a in angles]
    main_le = np.zeros(2)
    main_te = _te_mid(polys[0])

    # 3) mirror to the ground frame; reverse order to stay counter-clockwise
    mirror = np.array([1.0, -1.0])
    polys = [(p * mirror)[::-1].copy() for p in polys]
    main_te = main_te * mirror

    # 4) translate: lowest point at y = h, main LE at x = 0
    y_min = min(float(p[:, 1].min()) for p in polys)
    shift = np.array([0.0, params.ride_height_over_chord - y_min])
    polys = [p + shift for p in polys]
    main_le = main_le + shift
    main_te = main_te + shift
    moment_ref = main_le + 0.25 * (main_te - main_le)

    # achieved slot geometry, measured on the final polygons
    achieved_gaps, achieved_overlaps = [], []
    for i in range(1, len(polys)):
        achieved_gaps.append(min_distance(polys[i - 1], polys[i]))
        prev_te = _te_mid(polys[i - 1])
        le = _le_point(polys[i])
        chord_dir = prev_te - _le_point(polys[i - 1])
        chord_dir = chord_dir / np.linalg.norm(chord_dir)
        achieved_overlaps.append(float(np.dot(prev_te - le, chord_dir)))

    le_indices = [int(np.argmin(np.linalg.norm(p - _le_point(p), axis=1))) for p in polys]
    return Wing(
        polygons=polys,
        names=[f"element{i}" for i in range(len(polys))],
        chords=chords,
        ride_height=params.ride_height_over_chord,
        moment_ref=moment_ref,
        le_indices=le_indices,
        te_corners=[(0, len(p) - 1) for p in polys],
        achieved_gaps=achieved_gaps,
        achieved_overlaps=achieved_overlaps,
        element_angles_deg=[float(np.rad2deg(a)) for a in angles],
        specs=list(elements),
        params=params,
    )


def _le_point(poly: np.ndarray) -> np.ndarray:
    te_mid = _te_mid(poly)
    return poly[int(np.argmax(np.linalg.norm(poly - te_mid, axis=1)))]


def validate_wing(
    wing: Wing,
    min_gap: float = 0.003,
    min_ride_height: float = 0.02,
    max_x_extent: float = 1.5,
) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    problems: list[str] = []
    for name, poly in zip(wing.names, wing.polygons):
        if self_intersects(poly):
            problems.append(f"{name}: self-intersecting outline")
    for i in range(len(wing.polygons)):
        for j in range(i + 1, len(wing.polygons)):
            a, b = wing.polygons[i], wing.polygons[j]
            if polygons_intersect(a, b):
                problems.append(f"{wing.names[i]} and {wing.names[j]} overlap")
            else:
                d = min_distance(a, b)
                if d < min_gap:
                    problems.append(
                        f"{wing.names[i]}-{wing.names[j]} clearance {d:.4f} < min_gap {min_gap}"
                    )
    x0, y0, x1, _ = wing.bbox
    if y0 < min_ride_height - 1e-9:
        problems.append(f"lowest point y={y0:.4f} below min ride height {min_ride_height}")
    if x1 - x0 > max_x_extent:
        problems.append(f"streamwise extent {x1 - x0:.3f} > {max_x_extent}")
    return problems


def check_wing(wing: Wing, **kwargs) -> Wing:
    """Validate and raise GeometryError with all problems joined."""
    problems = validate_wing(wing, **kwargs)
    if problems:
        raise GeometryError("; ".join(problems))
    return wing
