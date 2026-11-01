"""Parameter space definition and scrambled-Sobol sampling (seeded)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

from ..geometry.wing import ElementSpec


@dataclass(frozen=True)
class Dim:
    name: str
    low: float
    high: float
    scale: str = "linear"  # linear | log

    def from_unit(self, u):
        u = np.asarray(u, float)
        if self.scale == "log":
            return np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low)))
        return self.low + u * (self.high - self.low)

    def to_unit(self, x):
        x = np.asarray(x, float)
        if self.scale == "log":
            return (np.log(x) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (x - self.low) / (self.high - self.low)


class ParameterSpace:
    def __init__(self, dims: list[Dim], elements: list[ElementSpec] | None = None):
        self.dims = dims
        self.elements = elements or []

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ParameterSpace":
        d = yaml.safe_load(Path(path).read_text())
        dims = [Dim(k, float(v["low"]), float(v["high"]), v.get("scale", "linear"))
                for k, v in d["parameters"].items()]
        for dim in dims:
            if not dim.low < dim.high:
                raise ValueError(f"{dim.name}: low must be < high")
            if dim.scale == "log" and dim.low <= 0:
                raise ValueError(f"{dim.name}: log scale needs low > 0")
        elements = [ElementSpec(**e) for e in d.get("elements", [])]
        return cls(dims, elements)

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dims]

    def from_unit(self, u: np.ndarray) -> pd.DataFrame:
        u = np.atleast_2d(u)
        return pd.DataFrame({d.name: d.from_unit(u[:, i]) for i, d in enumerate(self.dims)})

    def to_unit(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([d.to_unit(df[d.name].to_numpy()) for d in self.dims])

    def sobol(self, n: int, seed: int = 0, skip: int = 0) -> pd.DataFrame:
        """Scrambled Sobol points. Powers of two keep the balance properties."""
        s = qmc.Sobol(d=len(self.dims), scramble=True, seed=seed)
        if skip:
            s.fast_forward(skip)
        u = s.random(n)
        return self.from_unit(u)

    def uniform(self, n: int, seed: int = 0) -> pd.DataFrame:
        return self.from_unit(np.random.default_rng(seed).random((n, len(self.dims))))

    def contains(self, df: pd.DataFrame) -> np.ndarray:
        u = self.to_unit(df)
        return np.all((u >= -1e-12) & (u <= 1 + 1e-12), axis=1)
