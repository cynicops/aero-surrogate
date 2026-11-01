"""Grid Convergence Index (Roache), following the procedure of Celik et al. (2008),
"Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD
Applications", J. Fluids Eng. 130(7), 078001.

Index 1 = finest mesh, 3 = coarsest. Representative cell size h = (A_total / N)^(1/2) in 2D.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class GCIResult:
    phi1: float
    phi2: float
    phi3: float
    r21: float
    r32: float
    p: float  # observed order
    phi_ext: float  # Richardson extrapolation
    e_a21: float  # approximate relative error, fine pair
    e_ext21: float  # extrapolated relative error
    gci_fine: float  # fraction, multiply by 100 for %
    oscillatory: bool
    converged_iterations: int

    def to_dict(self) -> dict:
        return asdict(self)


def cell_size_2d(n_cells: int, area: float = 1.0) -> float:
    return math.sqrt(area / n_cells)


def gci_three_grids(
    phi1: float, phi2: float, phi3: float, h1: float, h2: float, h3: float, fs: float = 1.25,
    max_iter: int = 200,
) -> GCIResult:
    if not (h1 < h2 < h3):
        raise ValueError("need h1 < h2 < h3 (1 = finest)")
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = phi2 - phi1, phi3 - phi2
    if e21 == 0 or e32 == 0:
        raise ValueError("identical solutions on two grids: order undefined")
    s = math.copysign(1.0, e32 / e21)
    oscillatory = s < 0
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)  # initial guess (q = 0)
    n = 0
    for n in range(1, max_iter + 1):
        q = math.log((r21**p - s) / (r32**p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    phi_ext = (r21**p * phi1 - phi2) / (r21**p - 1)
    e_a21 = abs((phi1 - phi2) / phi1)
    e_ext21 = abs((phi_ext - phi1) / phi_ext)
    gci = fs * e_a21 / (r21**p - 1)
    return GCIResult(phi1, phi2, phi3, r21, r32, p, phi_ext, e_a21, e_ext21, gci, oscillatory, n)
