import numpy as np

from aerosurrogate.sampling import ParameterSpace

RANGES = "configs/ranges/front3.yaml"


def test_sobol_in_bounds_and_reproducible():
    sp = ParameterSpace.from_yaml(RANGES)
    a, b = sp.sobol(64, seed=3), sp.sobol(64, seed=3)
    assert a.equals(b)
    assert sp.contains(a).all()
    assert not a.equals(sp.sobol(64, seed=4))
    assert len(sp.elements) == 3


def test_log_scale_roundtrip():
    sp = ParameterSpace.from_yaml(RANGES)
    df = sp.sobol(32, seed=1)
    assert np.allclose(sp.from_unit(sp.to_unit(df)).to_numpy(), df.to_numpy())
    re = df["re"].to_numpy()
    assert re.min() >= 2e5 and re.max() <= 1e6
    # log scale: about half the samples below the geometric mean
    assert 0.3 < np.mean(re < np.sqrt(2e5 * 1e6)) < 0.7


def test_space_filling_better_than_random():
    from scipy.stats import qmc

    sp = ParameterSpace.from_yaml(RANGES)
    d_sobol = qmc.discrepancy(sp.to_unit(sp.sobol(128, seed=0)))
    d_rand = qmc.discrepancy(sp.to_unit(sp.uniform(128, seed=0)))
    assert d_sobol < d_rand


def test_fixed_parameters_added():
    sp = ParameterSpace.from_yaml("configs/ranges/front3_reduced.yaml")
    df = sp.sobol(8, seed=0)
    assert len(sp.names) == 5 and (df["gap1"] == 0.015).all()
    assert sp.contains(df).all()
