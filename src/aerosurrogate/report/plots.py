"""Report figures. Every figure carries a provenance line (solver, mesh, turbulence model,
Reynolds number, ride height), as required by CLAUDE.md."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402


def provenance(row: dict) -> str:
    return (
        f"{row.get('solver', '?')} [{row.get('solver_version', '?')}] | "
        f"mesh {int(row.get('mesh_cells', 0) or 0):,} cells | {row.get('turbulence_model', '?')} | "
        f"Re = {float(row.get('re', float('nan'))):.3g} | h/c = {float(row.get('ride_height_over_chord', float('nan'))):.3f}"
    )


def _stamp(fig, row: dict) -> None:
    fig.text(0.5, 0.005, provenance(row), ha="center", va="bottom", fontsize=7, color="0.35")


def cp_plot(cp: pd.DataFrame, row: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for el, g in cp.groupby("element"):
        ax.plot(g["x"], g["cp"], ".", ms=2.5, label=el)
    ax.invert_yaxis()
    ax.set_xlabel("x / c_ref")
    ax.set_ylabel("Cp")
    ax.set_title("Surface pressure coefficient (Cp axis inverted)")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.3)
    _stamp(fig, row)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def history_plot(hist: pd.DataFrame, res: pd.DataFrame | None, row: dict, window: int):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    it = hist.iloc[:, 0]
    for ax, col, lab in zip(axs[:2], ["Cl", "Cd"], ["Cl", "Cd"]):
        ax.plot(it, hist[col], lw=0.8)
        tail = hist[col].tail(window)
        lo, hi = tail.min(), tail.max()
        pad = max(abs(tail.mean()) * 0.05, (hi - lo) * 3, 1e-3)
        ax.set_ylim(tail.mean() - pad, tail.mean() + pad)
        ax.axvspan(it.iloc[-min(window, len(it))], it.iloc[-1], color="C1", alpha=0.15, label="averaging window")
        ax.set_title(f"{lab} history")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    if res is not None and not res.empty:
        for c in [c for c in res.columns if c.endswith("_initial")]:
            axs[2].semilogy(res.iloc[:, 0], res[c], lw=0.8, label=c.replace("_initial", ""))
        axs[2].legend(fontsize=7)
    axs[2].set_title("initial residuals")
    axs[2].set_xlabel("iteration")
    axs[2].grid(alpha=0.3)
    _stamp(fig, row)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def yplus_hist(yp: pd.DataFrame, row: dict):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    groups = [(p, g["y_plus"].to_numpy()) for p, g in yp.groupby("patch")]
    allv = np.concatenate([v for _, v in groups])
    bins = np.linspace(0, max(1.0, np.percentile(allv, 99.5)), 40)
    for p, v in groups:
        ax.hist(np.clip(v, 0, bins[-1]), bins=bins, histtype="step", label=f"{p} (max {v.max():.1f})")
    ax.set_xlabel("y+ (first cell)")
    ax.set_ylabel("faces")
    ax.set_title("y+ distribution per wall patch")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _stamp(fig, row)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def mesh_plot(preview_npz: str, polygons: list[np.ndarray], row: dict):
    z = np.load(preview_npz)
    xy = z["xy"]
    cells = [xy[t] for t in z["tri"]] + [xy[q] for q in z["quad"]]
    allp = np.vstack(polygons)
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    views = [((x0 - 0.15, x1 + 0.3), (0, y1 + 0.15)),
             ((x1 - 0.12, x1 + 0.03), (y1 - 0.08, y1 + 0.02))]
    for ax, (xl, yl) in zip(axs, views):
        ax.add_collection(PolyCollection(cells, facecolor="none", edgecolor="k", lw=0.15))
        for poly in polygons:
            ax.fill(poly[:, 0], poly[:, 1], color="C0", alpha=0.6)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
    axs[0].set_title("mesh near the wing (ground at y = 0)")
    axs[1].set_title("trailing-edge region of the last element")
    _stamp(fig, row)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def gci_plot(cells: list[int], values: list[float], phi_ext: float, name: str, row: dict):
    h = 1 / np.sqrt(np.asarray(cells, float))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(h, values, "o-", label=name)
    ax.axhline(phi_ext, ls="--", color="C3", label="Richardson extrapolation")
    ax.set_xlabel("h = 1/sqrt(N cells)")
    ax.set_ylabel(name)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"mesh convergence: {name}")
    _stamp(fig, row)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
