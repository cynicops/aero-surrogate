"""Write an OpenFOAM case directory from templates (Jinja2)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import jinja2

from .. import __version__
from ..config import CaseConfig
from ..geometry.wing import Wing
from ..mesh.gmsh_mesher import MeshInfo

TEMPLATE_DIR = Path(__file__).parent / "templates"


def inflow_turbulence(u_inf: float, intensity: float, viscosity_ratio: float, nu: float):
    k = 1.5 * (u_inf * intensity) ** 2
    omega = k / (nu * viscosity_ratio)
    return k, omega


def render_context(case_id: str, wing: Wing, mesh: MeshInfo, cfg: CaseConfig, n_procs: int) -> dict:
    u = cfg.flow.u_inf
    nu = u * 1.0 / wing.params.re
    k, omega = inflow_turbulence(u, cfg.flow.turbulence_intensity, cfg.flow.viscosity_ratio, nu)
    return {
        "version": __version__,
        "case_id": case_id,
        "solver": cfg.solver,
        "wall_patches": wing.names,
        "u_inf": u,
        "nu": f"{nu:.10g}",
        "k_inf": f"{k:.6g}",
        "omega_inf": f"{omega:.6g}",
        "l_ref": 1.0,
        "a_ref": mesh.span * 1.0,
        "cofr": [f"{wing.moment_ref[0]:.8f}", f"{wing.moment_ref[1]:.8f}"],
        "turbulence_model": cfg.flow.turbulence_model,
        "ground_moving": cfg.flow.ground == "moving",
        "n_procs": n_procs,
    }


def write_case(case_dir: str | Path, context: dict, msh_path: str | Path) -> Path:
    """Render every template into ``case_dir`` and copy the Gmsh mesh next to it."""
    case_dir = Path(case_dir)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    for tpl in sorted(TEMPLATE_DIR.rglob("*.j2")):
        rel = tpl.relative_to(TEMPLATE_DIR)
        out = case_dir / str(rel)[: -len(".j2")]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(env.get_template(rel.as_posix()).render(**context))
    dst = case_dir / "mesh.msh"
    if Path(msh_path).resolve() != dst.resolve():
        shutil.copy(msh_path, dst)
    return case_dir


PATCH_TYPES_DEFAULT = {
    "inlet": "patch",
    "outlet": "patch",
    "top": "patch",
    "front": "empty",
    "back": "empty",
}


def patch_types_for(wing_patch_names: list[str], ground: str) -> dict[str, str]:
    types = dict(PATCH_TYPES_DEFAULT)
    types["ground"] = "wall" if ground == "moving" else "patch"
    for n in wing_patch_names:
        types[n] = "wall"
    return types


def set_patch_types(boundary_file: str | Path, types: dict[str, str]) -> None:
    """Edit constant/polyMesh/boundary written by gmshToFoam: set each patch's type.

    Raises KeyError if a patch in ``types`` is missing or an unexpected patch exists.
    """
    path = Path(boundary_file)
    text = path.read_text()
    block = re.compile(r"(\n\s*)([A-Za-z_][\w.]*)(\s*\{)([^{}]*)(\})")
    seen = set()

    def repl(m: re.Match) -> str:
        name, body = m.group(2), m.group(4)
        if name == "FoamFile":
            return m.group(0)
        if name not in types:
            raise KeyError(f"unexpected patch '{name}' in {path}")
        seen.add(name)
        body = re.sub(r"\n\s*physicalType\s+\w+;", "", body)
        body = re.sub(r"\n\s*inGroups\s+[^;]*;", "", body)
        body = re.sub(r"(\btype\s+)\w+;", rf"\g<1>{types[name]};", body, count=1)
        if types[name] == "wall":
            body = body.rstrip() + "\n        inGroups        1(wall);\n    "
        return m.group(1) + name + m.group(3) + body + m.group(5)

    new = block.sub(repl, text)
    missing = set(types) - seen
    if missing:
        raise KeyError(f"patches missing from mesh: {sorted(missing)}")
    path.write_text(new)


def request_stop(case_dir: str | Path) -> None:
    """Ask a running solver to write and stop (controlDict is re-read every iteration)."""
    cd = Path(case_dir) / "system" / "controlDict"
    text = cd.read_text()
    text = re.sub(r"\bstopAt\s+\w+;", "stopAt          writeNow;", text, count=1)
    cd.write_text(text)
