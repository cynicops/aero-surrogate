import numpy as np
import pytest

from aerosurrogate.geometry import ElementSpec, WingParams, build_wing
from aerosurrogate.mesh.gmsh_mesher import MeshConfig, bl_plan, first_cell_height, mesh_wing


def test_first_cell_height_scales_with_re():
    assert first_cell_height(1, 1e6) < first_cell_height(1, 1e5)
    assert np.isclose(first_cell_height(30, 5e5) / first_cell_height(1, 5e5), 30)


def test_bl_plan_respects_cap():
    h1, n, t = bl_plan(1e-3, 1.2, 10, cap=5e-3)
    assert t <= 5e-3 and n >= 1
    h1, n, t = bl_plan(1e-2, 1.2, 10, cap=5e-3)  # first layer alone too thick
    assert n == 1 and np.isclose(h1, 5e-3)


def test_mesh_config_yaml_rejects_unknown(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("surface_size: 0.01\nbogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        MeshConfig.from_yaml(p)


@pytest.mark.slow
def test_mesh_three_element(tmp_path, three_elements, nominal_params):
    w = build_wing(three_elements, nominal_params)
    cfg = MeshConfig(far_size=1.0, surface_size=0.015, wake_size=0.05)
    info = mesh_wing(w, cfg, tmp_path, "m")
    assert info.n_cells == info.n_quads_2d + info.n_tris_2d  # one cell in z
    assert info.n_quads_2d > 0 and info.bl_layers >= 1
    assert info.bl_thickness <= 0.35 * min(w.achieved_gaps) + 1e-12
    text = (tmp_path / "m.msh").read_text()
    for name in ["inlet", "outlet", "top", "ground", "element0", "element1", "element2", "front", "back", "fluid"]:
        assert f'"{name}"' in text
    z = np.load(info.preview_path)
    assert len(z["tri"]) == info.n_tris_2d


def test_refinement_increases_cells(tmp_path):
    w = build_wing([ElementSpec("naca4412", 1.0)], WingParams(4, 0.3, 5e5))
    base = MeshConfig(far_size=1.0, surface_size=0.02, wake_size=0.06)
    n1 = mesh_wing(w, base, tmp_path, "a").n_cells
    n2 = mesh_wing(w, base.refined(1.5), tmp_path, "b").n_cells
    assert n2 > 1.8 * n1  # ideal 2D ratio is 1.5**2 = 2.25


def test_wall_function_refinement_keeps_first_cell(tmp_path):
    w = build_wing([ElementSpec("naca4412", 1.0)], WingParams(4, 0.3, 5e5))
    base = MeshConfig(far_size=1.0, surface_size=0.02, wake_size=0.06, y_plus_target=50)
    a = mesh_wing(w, base, tmp_path, "a")
    b = mesh_wing(w, base.refined(1.5), tmp_path, "b")
    assert np.isclose(a.first_cell_height, b.first_cell_height)  # same y+, wall model unchanged
    assert b.bl_layers > a.bl_layers
    low = MeshConfig(far_size=1.0, surface_size=0.02, wake_size=0.06, y_plus_target=1)
    c, d = mesh_wing(w, low, tmp_path, "c"), mesh_wing(w, low.refined(1.5), tmp_path, "d")
    assert d.first_cell_height < c.first_cell_height  # low-Re meshes refine the wall cell too
