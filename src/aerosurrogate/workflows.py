"""Phase 3-5 workflows: low-fidelity rows, surrogate training/evaluation, active-learning
benchmark, optimisation + export. Each function writes its outputs (CSV/JSON/PNG) into
``out`` and returns a summary dict. Every figure carries provenance text.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from .active.loop import ALConfig, run_comparison, runs_to_target
from .active.synthetic import synthetic_coefficients
from .data.dataset import Dataset, usable
from .data.schema import now_utc
from .geometry.wing import ElementSpec, GeometryError, WingParams, validate_wing
from .optimise.export import export_aero_map
from .optimise.optimiser import OptimConfig, optimise
from .report import plots
from .sampling import ParameterSpace
from .surrogate import OUTPUTS, EnsembleConfig, EnsembleSurrogate, features, metrics, split_by_case

plt = plots.plt


# ----------------------------------------------------------------------------- low-fi
def _lowfi_one(args):
    case_id, elements, flat, n_flaps = args
    from .baseline.lowfi import lowfi_row, lowfi_wing

    try:
        w = lowfi_wing([ElementSpec(**e) for e in elements], WingParams.from_flat(flat, n_flaps))
        if validate_wing(w):
            return None
        return lowfi_row(case_id, w)
    except GeometryError:
        return None


def add_lowfi_rows(dataset_dir: str | Path, workers: int = 2) -> int:
    """Add a source=neuralfoil row (same case_id) for every OpenFOAM case that lacks one."""
    ds = Dataset(dataset_dir)
    df = ds.load()
    have = set(df.loc[df.source == "neuralfoil", "case_id"])
    todo = []
    for _, r in df[(df.source == "openfoam") & ~df.case_id.isin(have)].iterrows():
        g = json.loads(r["geometry_params"])
        n_flaps = len(g["elements"]) - 1
        todo.append((r["case_id"], g["elements"], r.to_dict(), n_flaps))
    ctx = mp.get_context("spawn")
    n = 0
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for row in ex.map(_lowfi_one, todo):
            if row is not None:
                ds.append(row)
                n += 1
    return n


# ----------------------------------------------------------------------------- Phase 3
def train_and_evaluate(dataset_dir: str | Path, ranges: str | Path, out: str | Path,
                       n_splits: int = 5, test_fraction: float = 0.25, ens: EnsembleConfig | None = None,
                       truth: tuple[str, str] = ("source", "openfoam"),
                       lowfi: tuple[str, str] = ("source", "neuralfoil")) -> dict:
    """CFD-only surrogate vs multi-fidelity surrogate vs low-fi-only baseline, on case-wise splits.

    ``truth`` / ``lowfi`` select rows by (column, value); the defaults are OpenFOAM vs
    NeuralFoil/AeroSandbox. The synthetic demo selects by the ``solver`` column instead.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ens = ens or EnsembleConfig()
    space = ParameterSpace.from_yaml(ranges)
    inputs = space.names
    df = Dataset(dataset_dir).load()
    cfd = usable(df[df[truth[0]] == truth[1]]).set_index("case_id")
    low = df[df[lowfi[0]] == lowfi[1]].drop_duplicates("case_id").set_index("case_id")
    ids = sorted(set(cfd.index) & set(low.index))
    cfd, low = cfd.loc[ids], low.loc[ids]
    n_all = int((df[truth[0]] == truth[1]).sum())
    lf_cols = [f"lf_{o}" for o in OUTPUTS]
    for o in OUTPUTS:
        cfd[f"lf_{o}"] = low[o]
    data = cfd.reset_index()

    rows, parity = [], []
    for split in range(n_splits):
        tr, _, te = split_by_case(data, (1 - test_fraction, 0.0, test_fraction), seed=split)
        dtr, dte = data[data.case_id.isin(tr)], data[data.case_id.isin(te)]
        Ytr, Yte = dtr[OUTPUTS].to_numpy(), dte[OUTPUTS].to_numpy()
        models = {}
        s_cfd = EnsembleSurrogate(inputs, OUTPUTS, ens).fit(features(dtr, inputs), Ytr)
        models["surrogate_cfd_only"] = s_cfd.predict(features(dte, inputs))
        s_mf = EnsembleSurrogate(inputs + lf_cols, OUTPUTS, ens).fit(features(dtr, inputs + lf_cols), Ytr)
        models["surrogate_multifidelity"] = s_mf.predict(features(dte, inputs + lf_cols))
        models["baseline_lowfi_only"] = (dte[lf_cols].to_numpy(), None)
        # a fairer baseline: low-fi with a per-output linear correction fitted on the training cases
        corr = np.column_stack([np.polyval(np.polyfit(dtr[f"lf_{o}"], dtr[o], 1), dte[f"lf_{o}"]) for o in OUTPUTS])
        models["baseline_lowfi_linear_corrected"] = (corr, None)
        for name, (mu, sd) in models.items():
            m = metrics(Yte, mu, sd)
            m["model"], m["split"], m["n_train"], m["n_test"] = name, split, len(dtr), len(dte)
            rows.append(m)
            if split == 0:
                for j, o in enumerate(OUTPUTS):
                    parity.append(pd.DataFrame({"model": name, "output": o, "true": Yte[:, j], "pred": mu[:, j],
                                                "std": sd[:, j] if sd is not None else np.nan}))
        if split == 0:
            s_cfd.save(out / "surrogate_cfd_only")
            s_mf.save(out / "surrogate_multifidelity")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "metrics_all_splits.csv", index=False)
    agg = res.groupby(["model", "output"]).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), r2_mean=("r2", "mean"),
        mae_mean=("mae", "mean"), coverage_2sigma=("coverage_2sigma", "mean")).reset_index()
    agg.to_csv(out / "metrics_summary.csv", index=False)
    par = pd.concat(parity, ignore_index=True)
    par.to_csv(out / "parity_split0.csv", index=False)
    meta = {"dataset": str(dataset_dir), "ranges": str(ranges), "inputs": inputs, "n_cfd_rows": n_all,
            "n_usable_with_lowfi": len(ids), "n_splits": n_splits, "test_fraction": test_fraction,
            "ensemble": vars(ens), "created_utc": now_utc(), "provenance": _provenance(df)}
    (out / "phase3_summary.json").write_text(json.dumps(meta, indent=2, default=str))
    _parity_plot(par, out / "parity_split0.png", meta["provenance"])
    return {"summary": agg, "meta": meta}


def _provenance(df: pd.DataFrame) -> str:
    c = df[df.source == "openfoam"]
    if c.empty:
        return "synthetic data (no CFD)"
    return (f"simpleFoam [{c.solver_version.iloc[0]}] | coarse meshes {int(c.mesh_cells.min()):,}-"
            f"{int(c.mesh_cells.max()):,} cells | {c.turbulence_model.iloc[0]} | Re {c.re.min():.2g}-{c.re.max():.2g} | "
            f"h/c {c.ride_height_over_chord.min():.2f}-{c.ride_height_over_chord.max():.2f}")


def _parity_plot(par: pd.DataFrame, path: Path, prov: str) -> None:
    models = list(par.model.unique())
    fig, axs = plt.subplots(len(OUTPUTS), len(models), figsize=(3.2 * len(models), 8.5), squeeze=False)
    for i, o in enumerate(OUTPUTS):
        for j, mname in enumerate(models):
            ax = axs[i, j]
            d = par[(par.model == mname) & (par.output == o)]
            lo, hi = min(d.true.min(), d.pred.min()), max(d.true.max(), d.pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            if d["std"].notna().any():
                ax.errorbar(d.true, d.pred, yerr=2 * d["std"], fmt="o", ms=3, lw=0.7, capsize=2)
            else:
                ax.plot(d.true, d.pred, "o", ms=3)
            ax.set_title(f"{mname}\n{o}", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("predicted")
            if i == len(OUTPUTS) - 1:
                ax.set_xlabel("CFD (test cases)")
    fig.suptitle("Held-out test cases, split 0 (error bars = 2 x ensemble std)", fontsize=10)
    fig.text(0.5, 0.005, prov, ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- Phase 4
def al_synthetic_benchmark(ranges: str | Path, out: str | Path, seeds=(0, 1, 2), pool_size: int = 256,
                           al: ALConfig | None = None, ens: EnsembleConfig | None = None) -> dict:
    """Active learning vs space filling on the SYNTHETIC function (not aerodynamics)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    space = ParameterSpace.from_yaml(ranges)
    al = al or ALConfig(n_init=8, batch_size=4, budget=48, target_rmse_cl=0.05)
    ens = ens or EnsembleConfig(n_members=4, hidden=(64, 64), epochs=800, patience=120)
    test = space.sobol(512, seed=999)
    test = pd.concat([test, synthetic_coefficients(test)], axis=1)
    hist = []
    for s in seeds:
        pool = space.sobol(pool_size, seed=s)
        h = run_comparison(pool, synthetic_coefficients, test, space.names, space.to_unit(pool),
                           ALConfig(**{**vars(al), "seed": s}), EnsembleConfig(**{**vars(ens), "seed": s}),
                           progress=lambda m: None)
        h["seed"] = s
        hist.append(h)
    hist = pd.concat(hist, ignore_index=True)
    hist.to_csv(out / "al_synthetic_history.csv", index=False)
    res = _al_summary(hist, al.target_rmse_cl)
    _al_plot(hist, al.target_rmse_cl, out / "al_synthetic.png",
             f"SYNTHETIC test function ({ranges}), not CFD | ensemble x{ens.n_members} | seeds {list(seeds)}")
    (out / "al_synthetic_summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def al_retrospective_cfd(dataset_dir: str | Path, ranges: str | Path, out: str | Path, seeds=(0, 1, 2),
                         test_fraction: float = 0.25, target_rmse_cl: float = 0.05,
                         ens: EnsembleConfig | None = None) -> dict:
    """Pool-based AL on already-computed CFD cases (oracle = lookup). Small-data, retrospective."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    space = ParameterSpace.from_yaml(ranges)
    df = usable(Dataset(dataset_dir).load(source="openfoam")).reset_index(drop=True)
    ens = ens or EnsembleConfig(n_members=4, hidden=(32, 32), epochs=800, patience=120)
    hist = []
    for s in seeds:
        tr, _, te = split_by_case(df, (1 - test_fraction, 0.0, test_fraction), seed=s)
        pool = df[df.case_id.isin(tr)].sort_values("case_id").reset_index(drop=True)  # Sobol order
        test = df[df.case_id.isin(te)]
        lookup = pool.set_index("case_id")[OUTPUTS]

        def oracle(rows, lookup=lookup, pool=pool):
            return lookup.loc[pool.loc[rows.index, "case_id"]].reset_index(drop=True)

        al = ALConfig(n_init=6, batch_size=2, budget=len(pool), target_rmse_cl=target_rmse_cl, seed=s)
        h = run_comparison(pool[space.names + ["case_id"]].copy(), oracle, test, space.names,
                           space.to_unit(pool), al, EnsembleConfig(**{**vars(ens), "seed": s}),
                           progress=lambda m: None)
        h["seed"] = s
        hist.append(h)
    hist = pd.concat(hist, ignore_index=True)
    hist.to_csv(out / "al_cfd_retrospective_history.csv", index=False)
    res = _al_summary(hist, target_rmse_cl)
    _al_plot(hist, target_rmse_cl, out / "al_cfd_retrospective.png",
             _provenance(Dataset(dataset_dir).load()) + " | retrospective pool-based AL")
    (out / "al_cfd_retrospective_summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def _al_summary(hist: pd.DataFrame, target: float) -> dict:
    per_seed = {int(s): runs_to_target(h, target) for s, h in hist.groupby("seed")}
    mean_curve = hist.groupby(["method", "n_runs"])[["rmse_cl", "rmse_cd", "rmse_cm"]].mean().reset_index()
    final = mean_curve.groupby("method").tail(1).set_index("method")[["n_runs", "rmse_cl", "rmse_cd", "rmse_cm"]]
    curve_t = runs_to_target(mean_curve.assign(seed=0), target)
    if curve_t.get("active") and curve_t.get("space_filling"):
        verdict = ("active learning reached the target with fewer runs" if curve_t["active"] < curve_t["space_filling"]
                   else "active learning did NOT reach the target with fewer runs")
    elif curve_t.get("active") and not curve_t.get("space_filling"):
        verdict = "only active learning reached the target within the budget"
    elif curve_t.get("space_filling") and not curve_t.get("active"):
        verdict = "only space filling reached the target within the budget (active learning did not help)"
    else:
        verdict = "neither method reached the target within the budget"
    return {"target_rmse_cl": target, "runs_to_target_mean_curve": curve_t, "runs_to_target_per_seed": per_seed,
            "final_mean_rmse": final.to_dict(orient="index"), "verdict": verdict}


def _al_plot(hist: pd.DataFrame, target: float, path: Path, prov: str) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axs, ["rmse_cl", "rmse_cd", "rmse_cm"]):
        for method, h in hist.groupby("method"):
            g = h.groupby("n_runs")[col]
            m, s = g.mean(), g.std().fillna(0)
            ax.plot(m.index, m.values, "o-", ms=3, label=method)
            ax.fill_between(m.index, (m - s).values, (m + s).values, alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlabel("runs used")
        ax.set_ylabel(f"held-out {col}")
        ax.grid(alpha=0.3)
    axs[0].axhline(target, color="C3", ls="--", lw=0.8, label="target")
    axs[0].legend(fontsize=8)
    fig.suptitle("Runs vs accuracy (mean +/- sd over seeds)", fontsize=10)
    fig.text(0.5, 0.005, prov, ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- Phase 5
def optimise_and_export(surrogate_dir: str | Path, ranges: str | Path, out: str | Path,
                        cfg: OptimConfig | None = None, map_grid: dict | None = None,
                        source_note: str = "") -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sur = EnsembleSurrogate.load(surrogate_dir)
    space = ParameterSpace.from_yaml(ranges)
    cands = optimise(sur, space, cfg or OptimConfig())
    cands.to_csv(out / "candidates.csv", index=False)
    best = cands[cands.geometry_valid].iloc[0] if cands.geometry_valid.any() else cands.iloc[0]
    design = {k: float(best[k]) for k in space.names + list(space.fixed)}
    grid = map_grid or {"ride_height_over_chord": list(np.round(np.linspace(0.05, 0.6, 12), 4))}
    csv, js = export_aero_map(sur, space, design, grid, out / "aero_map", source_note or str(surrogate_dir))
    return {"best": best.to_dict(), "aero_map_csv": str(csv), "aero_map_json": str(js)}
