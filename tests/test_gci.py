import numpy as np
import pytest

from aerosurrogate.report.gci import cell_size_2d, gci_three_grids


def test_recovers_order_and_limit_on_synthetic_power_law():
    # synthetic: phi(h) = 1 + 0.5 h^2  (exact limit 1, order 2)
    hs = [0.01, 0.015, 0.0225]
    phis = [1 + 0.5 * h**2 for h in hs]
    r = gci_three_grids(*phis, *hs)
    assert np.isclose(r.p, 2.0, atol=1e-6)
    assert np.isclose(r.phi_ext, 1.0, atol=1e-9)
    assert not r.oscillatory and r.gci_fine > 0


def test_unequal_ratios():
    hs = [0.01, 0.013, 0.02]
    phis = [2 - 0.3 * h**1.5 for h in hs]
    r = gci_three_grids(*phis, *hs)
    assert np.isclose(r.p, 1.5, atol=1e-5)
    assert np.isclose(r.phi_ext, 2.0, atol=1e-8)


def test_oscillatory_flag_and_errors():
    r = gci_three_grids(1.0, 1.02, 0.99, 0.01, 0.015, 0.0225)
    assert r.oscillatory
    with pytest.raises(ValueError):
        gci_three_grids(1, 1, 2, 0.01, 0.015, 0.02)
    with pytest.raises(ValueError):
        gci_three_grids(1, 2, 3, 0.02, 0.015, 0.01)


def test_cell_size():
    assert np.isclose(cell_size_2d(100, 4.0), 0.2)
