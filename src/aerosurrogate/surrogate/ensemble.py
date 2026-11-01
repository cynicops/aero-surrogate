"""Deep-ensemble MLP surrogate (PyTorch) for CL, CD, CM with an uncertainty estimate.

* K members, each with its own seed and a bootstrap resample of the training cases.
* Inputs and outputs are standardised with training statistics only.
* Prediction = ensemble mean; uncertainty = ensemble standard deviation (epistemic only:
  it says nothing about CFD modelling error, which the surrogate inherits).
* Optional multi-fidelity mode: low-fidelity predictions are appended as input features.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class EnsembleConfig:
    n_members: int = 5
    hidden: tuple[int, ...] = (64, 64)
    activation: str = "silu"
    epochs: int = 1500
    lr: float = 3e-3
    weight_decay: float = 1e-4
    bootstrap: bool = True
    val_fraction: float = 0.15  # inside the training set, for early stopping
    patience: int = 200
    seed: int = 0
    threads: int = 2


def _mlp(n_in: int, n_out: int, hidden, activation: str) -> nn.Module:
    act = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


@dataclass
class Scaler:
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, a: np.ndarray) -> "Scaler":
        s = a.std(axis=0)
        return cls(a.mean(axis=0).tolist(), np.where(s > 1e-12, s, 1.0).tolist())

    def fwd(self, a):
        return (a - np.asarray(self.mean)) / np.asarray(self.std)

    def inv(self, a):
        return a * np.asarray(self.std) + np.asarray(self.mean)

    def inv_std(self, s):
        return s * np.asarray(self.std)


class EnsembleSurrogate:
    def __init__(self, input_names: list[str], output_names: list[str], cfg: EnsembleConfig | None = None):
        self.input_names = list(input_names)
        self.output_names = list(output_names)
        self.cfg = cfg or EnsembleConfig()
        self.members: list[nn.Module] = []
        self.xs: Scaler | None = None
        self.ys: Scaler | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "EnsembleSurrogate":
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if len(X) < 4:
            raise ValueError("need at least 4 training cases")
        torch.set_num_threads(self.cfg.threads)
        self.xs, self.ys = Scaler.fit(X), Scaler.fit(Y)
        Xn = torch.tensor(self.xs.fwd(X), dtype=torch.float32)
        Yn = torch.tensor(self.ys.fwd(Y), dtype=torch.float32)
        rng = np.random.default_rng(self.cfg.seed)
        self.members, self.history = [], []
        n = len(X)
        for k in range(self.cfg.n_members):
            torch.manual_seed(self.cfg.seed * 1000 + k)
            perm = rng.permutation(n)
            n_val = max(1, int(round(self.cfg.val_fraction * n))) if n >= 10 else 0
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            if self.cfg.bootstrap:
                tr_idx = rng.choice(tr_idx, size=len(tr_idx), replace=True)
            model = _mlp(X.shape[1], Y.shape[1], self.cfg.hidden, self.cfg.activation)
            opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            best, best_state, since = np.inf, None, 0
            xt, yt = Xn[tr_idx], Yn[tr_idx]
            for epoch in range(self.cfg.epochs):
                model.train()
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xt), yt)
                loss.backward()
                opt.step()
                if n_val:
                    model.eval()
                    with torch.no_grad():
                        v = float(nn.functional.mse_loss(model(Xn[val_idx]), Yn[val_idx]))
                else:
                    v = loss.item()
                if v < best - 1e-7:
                    best, since = v, 0
                    best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}
                else:
                    since += 1
                    if since > self.cfg.patience:
                        break
            model.load_state_dict(best_state)
            model.eval()
            self.members.append(model)
            self.history.append({"member": k, "best_val_mse": best, "epochs": epoch + 1})
        return self

    # ---------------------------------------------------------------- prediction
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) in physical units, shape (n, n_outputs)."""
        if not self.members:
            raise RuntimeError("surrogate is not trained")
        Xn = torch.tensor(self.xs.fwd(np.asarray(X, float)), dtype=torch.float32)
        with torch.no_grad():
            preds = np.stack([m(Xn).numpy() for m in self.members])  # (K, n, out)
        mean_n, std_n = preds.mean(axis=0), preds.std(axis=0, ddof=1 if len(self.members) > 1 else 0)
        return self.ys.inv(mean_n), self.ys.inv_std(std_n)

    def predict_members(self, X: np.ndarray) -> np.ndarray:
        Xn = torch.tensor(self.xs.fwd(np.asarray(X, float)), dtype=torch.float32)
        with torch.no_grad():
            return np.stack([self.ys.inv(m(Xn).numpy()) for m in self.members])

    # -------------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"inputs": self.input_names, "outputs": self.output_names, "cfg": asdict(self.cfg),
                "xs": asdict(self.xs), "ys": asdict(self.ys), "history": self.history}
        (path / "surrogate.json").write_text(json.dumps(meta, indent=2))
        torch.save([m.state_dict() for m in self.members], path / "members.pt")

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleSurrogate":
        path = Path(path)
        meta = json.loads((path / "surrogate.json").read_text())
        cfg = EnsembleConfig(**{**meta["cfg"], "hidden": tuple(meta["cfg"]["hidden"])})
        s = cls(meta["inputs"], meta["outputs"], cfg)
        s.xs, s.ys, s.history = Scaler(**meta["xs"]), Scaler(**meta["ys"]), meta["history"]
        states = torch.load(path / "members.pt", weights_only=True)
        for st in states:
            m = _mlp(len(s.input_names), len(s.output_names), cfg.hidden, cfg.activation)
            m.load_state_dict(st)
            m.eval()
            s.members.append(m)
        return s
