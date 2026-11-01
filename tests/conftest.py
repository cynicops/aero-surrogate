import numpy as np
import pytest

from aerosurrogate.geometry import ElementSpec, WingParams

THREE = [ElementSpec("s1223", 0.55), ElementSpec("s1223", 0.27), ElementSpec("s1223", 0.18)]


@pytest.fixture
def three_elements():
    return list(THREE)


@pytest.fixture
def nominal_params():
    return WingParams(aoa_deg=4, ride_height_over_chord=0.2, re=5e5,
                      flap_angles_deg=[20, 20], gaps=[0.015, 0.015], overlaps=[0.02, 0.02])


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
