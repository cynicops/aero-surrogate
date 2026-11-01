"""Case configuration: everything except the wing parameters lives in YAML files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from .geometry.wing import ElementSpec
from .mesh.gmsh_mesher import MeshConfig


def _build(cls, data: dict | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**data)


@dataclass
class FlowConfig:
    turbulence_model: str = "kOmegaSST"  # fully turbulent RANS (SPEC open question 4)
    turbulence_intensity: float = 0.005
    viscosity_ratio: float = 10.0
    ground: str = "moving"  # "moving" (belt at U_inf) or "slip" (no ground boundary layer)
    u_inf: float = 1.0  # non-dimensional; Re is set through nu


@dataclass
class SolverConfig:
    max_iter: int = 3000
    residual_tol: float = 1.0e-5
    simplec: bool = False
    relax_p: float = 0.3
    relax_u: float = 0.7
    relax_turb: float = 0.7
    n_non_orth: int = 0
    div_u: str = "bounded Gauss linearUpwind grad(U)"
    div_turb: str = "bounded Gauss upwind"
    element_force_interval: int = 10


@dataclass
class ConvergenceConfig:
    min_iter: int = 300
    window: int = 200  # iterations used for the force-convergence test and averaging
    cl_rel_tol: float = 0.002  # (max-min)/|mean| over the window
    cd_rel_tol: float = 0.005
    cm_abs_tol: float = 0.002
    poll_seconds: float = 2.0
    wall_clock_limit_s: float = 1800.0


@dataclass
class GeometryLimits:
    min_gap: float = 0.003
    min_ride_height: float = 0.02
    max_x_extent: float = 1.5
    n_points_per_side: int = 120


@dataclass
class CaseConfig:
    elements: list[ElementSpec]
    mesh: MeshConfig = field(default_factory=MeshConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    limits: GeometryLimits = field(default_factory=GeometryLimits)

    @classmethod
    def from_dict(cls, d: dict) -> "CaseConfig":
        elements = [ElementSpec(**e) for e in d["elements"]]
        return cls(
            elements=elements,
            mesh=_build(MeshConfig, d.get("mesh")),
            flow=_build(FlowConfig, d.get("flow")),
            solver=_build(SolverConfig, d.get("solver")),
            convergence=_build(ConvergenceConfig, d.get("convergence")),
            limits=_build(GeometryLimits, d.get("limits")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CaseConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {
            "elements": [asdict(e) for e in self.elements],
            "mesh": asdict(self.mesh),
            "flow": asdict(self.flow),
            "solver": asdict(self.solver),
            "convergence": asdict(self.convergence),
            "limits": asdict(self.limits),
        }

    @property
    def n_flaps(self) -> int:
        return len(self.elements) - 1


@dataclass
class LauncherConfig:
    kind: str = "docker"  # docker | native
    image: str = "opencfd/openfoam-default:2512"
    bashrc: str = "/usr/lib/openfoam/openfoam2512/etc/bashrc"
    version_label: str = "OpenFOAM v2512"
    n_procs: int = 2
    docker_user: str = "auto"  # "auto": current uid:gid unless root

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LauncherConfig":
        return _build(cls, yaml.safe_load(Path(path).read_text()))
