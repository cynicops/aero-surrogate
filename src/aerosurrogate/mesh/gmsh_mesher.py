"""2D hybrid mesh (quad boundary layers + triangles) around a multi-element wing in
ground effect, extruded one cell in z for OpenFOAM (front/back = empty patches).

Domain (non-dimensional, c_ref = 1): x in [-upstream, x_te + downstream], y in [0, top].
Patches: inlet, outlet, top, ground, element0..elementN, front, back.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from ..geometry.wing import Wing


class MeshError(RuntimeError):
    pass


@dataclass
class MeshConfig:
    upstream: float = 5.0
    downstream: float = 10.0
    top: float = 5.0
    span: float = 0.05
    surface_size: float = 0.008
    le_size: float = 0.003
    te_size: float = 0.0015
    far_size: float = 0.6
    growth_rate: float = 0.15  # size increase per unit wall distance
    wake_size: float = 0.025
    wake_length: float = 1.5
    y_plus_target: float = 50.0  # wall-function mesh by default (coarse smoke tests)
    bl_ratio: float = 1.2
    bl_layers: int = 8
    bl_max_fraction_of_gap: float = 0.35
    bl_max_fraction_of_ride_height: float = 0.25
    ground_bl: bool = True
    ground_y_plus_target: float = 50.0
    ground_bl_layers: int = 4
    refinement: float = 1.0  # >1 refines all sizes (used for mesh studies / GCI)
    algorithm: int = 6  # gmsh 2D algorithm (6 = Frontal-Delaunay)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MeshConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown mesh config keys: {sorted(unknown)}")
        return cls(**data)

    def refined(self, factor: float) -> "MeshConfig":
        d = asdict(self)
        d["refinement"] = self.refinement * factor
        return MeshConfig(**d)


@dataclass
class MeshInfo:
    msh_path: str
    n_cells: int
    n_quads_2d: int
    n_tris_2d: int
    first_cell_height: float
    bl_layers: int
    bl_thickness: float
    ground_first_cell_height: float | None
    span: float
    domain: tuple[float, float, float, float]
    preview_path: str


def first_cell_height(y_plus: float, re: float) -> float:
    """Flat-plate estimate (Cf = 0.026 Re^-1/7) of the first-cell height for a target y+,
    non-dimensional with c_ref and U_inf = 1. Local suction peaks give higher y+."""
    cf = 0.026 * re ** (-1.0 / 7.0)
    u_tau = math.sqrt(cf / 2.0)
    return y_plus / (re * u_tau)


def bl_plan(h1: float, ratio: float, layers: int, cap: float) -> tuple[float, int, float]:
    """Return (first height, layers, total thickness) respecting the thickness cap."""
    def thick(h, n):
        return h * (ratio**n - 1) / (ratio - 1) if ratio != 1 else h * n

    n = layers
    while n > 1 and thick(h1, n) > cap:
        n -= 1
    if thick(h1, n) > cap:  # even one layer is too thick: shrink it
        h1 = cap
        n = 1
    return h1, n, thick(h1, n)


def mesh_wing(wing: Wing, cfg: MeshConfig, out_dir: str | Path, name: str = "mesh") -> MeshInfo:
    """Mesh the wing and write ``<out_dir>/<name>.msh`` (Gmsh 2.2 ASCII)."""
    import gmsh

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = cfg.refinement
    re = wing.params.re

    x0, y0, x1, y1 = wing.bbox
    xmin, xmax = x0 - cfg.upstream, x1 + cfg.downstream
    ymax = max(cfg.top, y1 + 2.0)

    min_gap = min(wing.achieved_gaps) if wing.achieved_gaps else math.inf
    cap = min(cfg.bl_max_fraction_of_gap * min_gap, cfg.bl_max_fraction_of_ride_height * y0)
    h1, n_layers, thickness = bl_plan(
        first_cell_height(cfg.y_plus_target, re) / r,
        cfg.bl_ratio,
        max(1, int(round(cfg.bl_layers * r))),
        cap,
    )

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.model.add(name)
        geo = gmsh.model.geo

        # outer box
        p = [geo.addPoint(xmin, 0, 0), geo.addPoint(xmax, 0, 0), geo.addPoint(xmax, ymax, 0),
             geo.addPoint(xmin, ymax, 0)]
        c_ground = geo.addLine(p[0], p[1])
        c_outlet = geo.addLine(p[1], p[2])
        c_top = geo.addLine(p[2], p[3])
        c_inlet = geo.addLine(p[3], p[0])
        outer = geo.addCurveLoop([c_ground, c_outlet, c_top, c_inlet])
        loop_curves = [c_ground, c_outlet, c_top, c_inlet]

        element_curves: list[list[int]] = []
        le_points, te_points, holes = [], [], []
        for poly, ile in zip(wing.polygons, wing.le_indices):
            tags = [geo.addPoint(float(x), float(y), 0) for x, y in poly]
            a = geo.addSpline(tags[: ile + 1])
            b = geo.addSpline(tags[ile:])
            c = geo.addLine(tags[-1], tags[0])  # blunt trailing-edge base
            holes.append(geo.addCurveLoop([a, b, c]))
            element_curves.append([a, b, c])
            loop_curves += [a, b, c]
            le_points.append(tags[ile])
            te_points += [tags[0], tags[-1]]
        surf = geo.addPlaneSurface([outer] + holes)
        ext = geo.extrude([(2, surf)], 0, 0, cfg.span, numElements=[1], recombine=True)
        geo.synchronize()

        back = ext[0][1]
        vol = ext[1][1]
        lateral = [e[1] for e in ext[2:]]
        if len(lateral) != len(loop_curves):
            raise MeshError("unexpected extrusion output; cannot map patches")
        lat = dict(zip(loop_curves, lateral))

        gm = gmsh.model
        for tag, nm in ((c_inlet, "inlet"), (c_outlet, "outlet"), (c_top, "top"), (c_ground, "ground")):
            gm.setPhysicalName(2, gm.addPhysicalGroup(2, [lat[tag]]), nm)
        for i, curves in enumerate(element_curves):
            gm.setPhysicalName(2, gm.addPhysicalGroup(2, [lat[c] for c in curves]), wing.names[i])
        gm.setPhysicalName(2, gm.addPhysicalGroup(2, [surf]), "front")
        gm.setPhysicalName(2, gm.addPhysicalGroup(2, [back]), "back")
        gm.setPhysicalName(3, gm.addPhysicalGroup(3, [vol]), "fluid")

        # ---- size fields
        f = gm.mesh.field
        all_wall = [c for cs in element_curves for c in cs]
        fdist = f.add("Distance")
        f.setNumbers(fdist, "CurvesList", all_wall)
        f.setNumber(fdist, "Sampling", 400)
        s_surf, s_far = cfg.surface_size / r, cfg.far_size / r
        rate = cfg.growth_rate / r  # refine the growth too, so refined meshes stay geometrically similar

        def threshold(dist_field: int, size_min: float) -> int:
            t = f.add("Threshold")
            f.setNumber(t, "InField", dist_field)
            f.setNumber(t, "SizeMin", size_min)
            f.setNumber(t, "SizeMax", s_far)
            f.setNumber(t, "DistMin", 0.0)
            f.setNumber(t, "DistMax", (s_far - size_min) / rate)
            return t

        t_wall = threshold(fdist, s_surf)
        fle = f.add("Distance")
        f.setNumbers(fle, "PointsList", le_points)
        t_le = threshold(fle, cfg.le_size / r)
        fte = f.add("Distance")
        f.setNumbers(fte, "PointsList", te_points)
        t_te = threshold(fte, cfg.te_size / r)

        def box(xa, xb, ya, yb, size):
            b_ = f.add("Box")
            f.setNumber(b_, "VIn", size)
            f.setNumber(b_, "VOut", s_far)
            f.setNumber(b_, "XMin", xa)
            f.setNumber(b_, "XMax", xb)
            f.setNumber(b_, "YMin", ya)
            f.setNumber(b_, "YMax", yb)
            f.setNumber(b_, "Thickness", 0.3)
            return b_

        s_wake = cfg.wake_size / r
        b_wake = box(x0 - 0.1, x1 + cfg.wake_length, 0.0, y1 + 0.15, s_wake)
        b_far_wake = box(x1, x1 + 3 * cfg.wake_length, 0.0, y1 + 0.5, 3 * s_wake)
        b_channel = box(x0 - 0.3, x1 + 0.3, 0.0, max(y0, 0.02), min(s_wake, max(y0 / 6, s_surf)))
        fmin = f.add("Min")
        f.setNumbers(fmin, "FieldsList", [t_wall, t_le, t_te, b_wake, b_far_wake, b_channel])
        f.setAsBackgroundMesh(fmin)

        # ---- boundary layers
        bl = f.add("BoundaryLayer")
        f.setNumbers(bl, "CurvesList", all_wall)
        f.setNumber(bl, "Size", h1)
        f.setNumber(bl, "Ratio", cfg.bl_ratio)
        f.setNumber(bl, "Thickness", thickness)
        f.setNumber(bl, "Quads", 1)
        f.setNumbers(bl, "FanPointsList", [])
        f.setAsBoundaryLayer(bl)

        g_h1 = None
        if cfg.ground_bl:
            g_cap = 0.25 * y0
            g_h1, g_n, g_thick = bl_plan(
                first_cell_height(cfg.ground_y_plus_target, re) / r,
                cfg.bl_ratio,
                max(1, int(round(cfg.ground_bl_layers * r))),
                g_cap,
            )
            gbl = f.add("BoundaryLayer")
            f.setNumbers(gbl, "CurvesList", [c_ground])
            f.setNumbers(gbl, "PointsList", [p[0], p[1]])
            f.setNumber(gbl, "Size", g_h1)
            f.setNumber(gbl, "Ratio", cfg.bl_ratio)
            f.setNumber(gbl, "Thickness", g_thick)
            f.setNumber(gbl, "Quads", 1)
            f.setAsBoundaryLayer(gbl)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", cfg.algorithm)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)

        gm.mesh.generate(3)

        types, tags_by_type, _ = gm.mesh.getElements(3)
        n_cells = int(sum(len(t) for t in tags_by_type))
        if n_cells == 0:
            raise MeshError("gmsh produced no volume cells")
        t2, _, nodes2 = gm.mesh.getElements(2, surf)
        n_tri = n_quad = 0
        tri_conn, quad_conn = np.empty((0, 3), int), np.empty((0, 4), int)
        for et, nd in zip(t2, nodes2):
            if et == 2:
                tri_conn = np.asarray(nd, dtype=np.int64).reshape(-1, 3)
                n_tri = len(tri_conn)
            elif et == 3:
                quad_conn = np.asarray(nd, dtype=np.int64).reshape(-1, 4)
                n_quad = len(quad_conn)
        node_tags, coords, _ = gm.mesh.getNodes()
        xy = np.asarray(coords).reshape(-1, 3)[:, :2]
        lookup = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
        lookup[node_tags.astype(np.int64)] = np.arange(len(node_tags))
        preview = out_dir / f"{name}_preview.npz"
        np.savez_compressed(
            preview, xy=xy, tri=lookup[tri_conn], quad=lookup[quad_conn],
        )
        msh = out_dir / f"{name}.msh"
        gmsh.write(str(msh))
    finally:
        gmsh.finalize()

    return MeshInfo(
        msh_path=str(msh),
        n_cells=n_cells,
        n_quads_2d=n_quad,
        n_tris_2d=n_tri,
        first_cell_height=h1,
        bl_layers=n_layers,
        bl_thickness=thickness,
        ground_first_cell_height=g_h1,
        span=cfg.span,
        domain=(xmin, 0.0, xmax, ymax),
        preview_path=str(preview),
    )
