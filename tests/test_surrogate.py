import numpy as np
import pandas as pd
import pytest

from aerosurrogate.surrogate import EnsembleConfig, EnsembleSurrogate, features, metrics, split_by_case

FAST = EnsembleConfig(n_members=3, hidden=(32, 32), epochs=600, patience=100, seed=0)


def toy(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    Y = np.column_stack([np.sin(2 * X[:, 0]) + X[:, 1] ** 2, 0.1 * X[:, 0] * X[:, 1]])
    return X, Y


def test_fits_smooth_function_and_reports_uncertainty():
    X, Y = toy(200)
    s = EnsembleSurrogate(["a", "b"], ["y1", "y2"], FAST).fit(X, Y)
    Xt, Yt = toy(100, seed=1)
    mu, sd = s.predict(Xt)
    m = metrics(Yt, mu, sd, names=["y1", "y2"])
    assert (m["r2"] > 0.95).all()
    assert (sd > 0).all()


def test_uncertainty_grows_off_data():
    X, Y = toy(150)
    s = EnsembleSurrogate(["a", "b"], ["y1", "y2"], FAST).fit(X, Y)
    _, sd_in = s.predict(np.zeros((1, 2)))
    _, sd_out = s.predict(np.full((1, 2), 4.0))  # far outside the training box
    assert sd_out[0, 0] > sd_in[0, 0]


def test_reproducible_and_save_load(tmp_path):
    X, Y = toy(60)
    a = EnsembleSurrogate(["a", "b"], ["y1", "y2"], FAST).fit(X, Y)
    b = EnsembleSurrogate(["a", "b"], ["y1", "y2"], FAST).fit(X, Y)
    assert np.allclose(a.predict(X)[0], b.predict(X)[0], atol=1e-5)
    a.save(tmp_path / "m")
    c = EnsembleSurrogate.load(tmp_path / "m")
    assert np.allclose(a.predict(X)[0], c.predict(X)[0], atol=1e-6)


def test_untrained_and_tiny():
    s = EnsembleSurrogate(["a"], ["y"], FAST)
    with pytest.raises(RuntimeError):
        s.predict(np.zeros((1, 1)))
    with pytest.raises(ValueError):
        s.fit(np.zeros((2, 1)), np.zeros((2, 1)))


def test_split_by_case_no_leakage():
    df = pd.DataFrame({"case_id": [f"c{i // 2}" for i in range(40)], "source": ["openfoam", "neuralfoil"] * 20})
    tr, va, te = split_by_case(df, seed=3)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(df["case_id"])
    # both rows of a case end up in the same set
    for ids in (tr, va, te):
        assert len(df[df.case_id.isin(ids)]) == 2 * len(ids)


def test_features_log_re():
    df = pd.DataFrame({"re": [1e5, 1e6], "aoa_deg": [1.0, 2.0]})
    assert np.allclose(features(df, ["re", "aoa_deg"]), [[5, 1], [6, 2]])
