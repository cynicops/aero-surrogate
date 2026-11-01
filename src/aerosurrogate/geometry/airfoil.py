"""Airfoil section loading and normalisation.

Coordinates come from AeroSandbox's bundled airfoil database (UIUC-derived .dat files and
NACA generators), so no network access is needed. We do not re-implement AeroSandbox.

Local section convention ("aerofoil frame", lift-up): unit chord, leading edge (LE) at
(0, 0), trailing-edge (TE) midpoint at (1, 0), Selig ordering
(upper-surface TE -> LE -> lower-surface TE), i.e. counter-clockwise.
The TE is opened to a finite (blunt) thickness so that it can be meshed robustly.
"""

from __future__ import annotations

import functools

import numpy as np

from .polygon import rotation_matrix, signed_area


class AirfoilError(ValueError):
    """Raised when a section cannot be loaded or is not usable."""


def _le_index(coords: np.ndarray) -> int:
    """Index of the leading edge: the point farthest from the TE midpoint."""
    te_mid = 0.5 * (coords[0] + coords[-1])
    return int(np.argmax(np.linalg.norm(coords - te_mid, axis=1)))


def normalise(coords: np.ndarray) -> np.ndarray:
    """Translate/rotate/scale so LE is at (0, 0) and the TE midpoint at (1, 0)."""
    coords = np.asarray(coords, dtype=float)
    ile = _le_index(coords)
    le = coords[ile]
    te_mid = 0.5 * (coords[0] + coords[-1])
    chord_vec = te_mid - le
    chord = float(np.linalg.norm(chord_vec))
    if chord <= 0:
        raise AirfoilError("degenerate airfoil: zero chord")
    angle = np.arctan2(chord_vec[1], chord_vec[0])
    out = (coords - le) @ rotation_matrix(-angle).T / chord
    # Selig order is counter-clockwise; flip if the source file is the other way round.
    if signed_area(out) < 0:
        out = out[::-1].copy()
    return out


def set_te_thickness(coords: np.ndarray, te_thickness: float) -> np.ndarray:
    """Set the TE thickness (fraction of chord) by adding a linear thickness ramp.

    Upper surface gets +ramp/2, lower surface -ramp/2, ramp = (t_target - t_current) * x.
    The LE is unchanged. Works on a normalised section in Selig order.
    """
    if te_thickness < 0:
        raise AirfoilError("te_thickness must be >= 0")
    coords = coords.copy()
    ile = _le_index(coords)
    current = coords[0, 1] - coords[-1, 1]
    delta = te_thickness - current
    x = np.clip(coords[:, 0], 0.0, None)
    upper = np.arange(len(coords)) < ile
    lower = np.arange(len(coords)) > ile
    coords[upper, 1] += 0.5 * delta * x[upper]
    coords[lower, 1] -= 0.5 * delta * x[lower]
    return coords


@functools.lru_cache(maxsize=64)
def _load_cached(name: str, n_per_side: int, te_thickness: float) -> np.ndarray:
    import aerosandbox as asb  # heavy import kept local

    af = asb.Airfoil(name)
    if af.coordinates is None or len(af.coordinates) < 10:
        raise AirfoilError(f"airfoil '{name}' not found in the AeroSandbox database")
    af = af.repanel(n_points_per_side=n_per_side)
    coords = normalise(af.coordinates)
    return set_te_thickness(coords, te_thickness)


def load_section(name: str, n_per_side: int = 120, te_thickness: float = 0.004) -> np.ndarray:
    """Return a normalised, blunt-TE section (copy) for an AeroSandbox airfoil name.

    Examples of names: "s1223", "naca4412", "ls417" (= GA(W)-1 / LS(1)-0417).
    """
    return _load_cached(name.lower(), int(n_per_side), float(te_thickness)).copy()


def split_surfaces(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a Selig-ordered section into (upper TE->LE, lower LE->TE), both including LE."""
    ile = _le_index(coords)
    return coords[: ile + 1], coords[ile:]
