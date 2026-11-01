import numpy as np
import pandas as pd

from aerosurrogate.active.loop import ALConfig, _pick, run_comparison, runs_to_target
from aerosurrogate.active.synthetic import synthetic_coefficients
from aerosurrogate.optimise.export import export_aero_map
from aerosurrogate.optimise.optimiser import OptimConfig, optimise
from aerosurrogate.sampling import ParameterSpace
from aerosurrogate.surrogate import EnsembleConfig, EnsembleSurrogate, features

SP = "configs/ranges/front3_reduced.yaml"
FAST = EnsembleConfig(n_members=3, hidden=(32, 32), epochs=400, patience=80, seed=0)


def test_synthetic_has_ground_effect_peak():
    df = pd.DataFrame({"aoa_deg": 4.0, "flap1_deg": 20.0, "flap2_deg": 20.0, "re": 5e5,
                       "ride_height_over_chord": [0.03, 0.12, 0.5]})
    down = -synthetic_coefficients(df)["cl"].to_numpy()
    assert down[1] > down[0] and down[1] > down[2]


def test_pick_respects_diversity():
    U = np.array([[0, 0], [0.01, 0], [1, 1]], float)
    assert _pick(np.array([3.0, 2.0, 1.0]), U, 2, 0.5) == [0, 2]


def test_loop_runs_and_compares():
    sp = ParameterSpace.from_yaml(SP)
    pool = sp.sobol(64, seed=0)
    test = sp.sobol(64, seed=99)
    test = pd.concat([test, synthetic_coefficients(test)], axis=1)
    oracle = lambda df: synthetic_coefficients(df)  # noqa: E731
    h = run_comparison(pool, oracle, test, sp.names, sp.to_unit(pool),
                       ALConfig(n_init=8, batch_size=8, budget=24), FAST, progress=lambda s: None)
    assert set(h["method"]) == {"space_filling", "active"}
    assert h.groupby("method")["n_runs"].max().tolist() == [24, 24]
    assert set(runs_to_target(h, 10.0)) == {"active", "space_filling"}


def test_failed_runs_are_counted_not_dropped():
    sp = ParameterSpace.from_yaml(SP)
    pool = sp.sobol(32, seed=0)
    test = pd.concat([sp.sobol(16, seed=5), synthetic_coefficients(sp.sobol(16, seed=5))], axis=1)

    def flaky(df):
        out = synthetic_coefficients(df)
        out.iloc[::3] = np.nan  # every third run "fails"
        return out

    h = run_comparison(pool, flaky, test, sp.names, sp.to_unit(pool),
                       ALConfig(n_init=8, batch_size=4, budget=16), FAST, progress=lambda s: None)
    last = h[h.method == "active"].iloc[-1]
    assert last["n_runs"] == 16 and last["n_ok"] < 16


def test_optimiser_and_export(tmp_path):
    sp = ParameterSpace.from_yaml(SP)
    df = sp.sobol(128, seed=0)
    y = synthetic_coefficients(df)
    sur = EnsembleSurrogate(sp.names, ["cl", "cd", "cm"], FAST).fit(features(df, sp.names), y.to_numpy())
    cands = optimise(sur, sp, OptimConfig(n_starts=2, maxiter=20, popsize=8))
    assert len(cands) == 2 and {"pred_cl", "pred_cl_std", "geometry_valid"} <= set(cands.columns)
    assert cands["score"].iloc[0] >= cands["score"].iloc[1]
    design = cands.iloc[0][sp.names + list(sp.fixed)].to_dict()
    csv, js = export_aero_map(sur, sp, design, {"ride_height_over_chord": [0.05, 0.1, 0.6]}, tmp_path / "map", "test")
    m = pd.read_csv(csv)
    assert len(m) == 3 and m["extrapolated"].tolist() == [False, False, True]
    assert np.allclose(m["downforce_coeff"], -m["cl"])
