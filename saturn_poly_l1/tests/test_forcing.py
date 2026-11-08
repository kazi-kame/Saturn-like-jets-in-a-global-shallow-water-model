#============
#Imports and module setup
#============
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from diagnostics import verify_config, planetographic_to_planetocentric
from run_l1 import Forcing, jet_profile
CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/exploratory_proxy.json').read_text())

#============
#Setup inputs
#============
def setup_inputs():
    lon = np.linspace(0, 2 * np.pi, 64, endpoint=False)[:, None] + np.zeros((1, 40))
    lat = np.linspace(-1.53, 1.53, 40)[None, :] + np.zeros((64, 1))
    return (lon, lat)

#============
#Test unforced control has zero forcing
#============
def test_unforced_control_has_zero_forcing():
    lo, la = setup_inputs()
    F = Forcing(CONFIG, lo, la, 'A')
    for step in range(4):
        fu, fv = F.calculate(step, 60.0)
        assert np.all(fu == 0) and np.all(fv == 0)

#============
#Test initial state is axisymmetric
#============
def test_initial_state_is_axisymmetric():
    lo, la = setup_inputs()
    u = jet_profile(la, CONFIG) + np.zeros_like(lo)
    assert np.max(np.abs(u - u.mean(axis=0)[None, :])) < 1e-12

#============
#Test vortex is post initialization and not polygon seed
#============
def test_vortex_is_post_initialization_and_not_polygon_seed():
    lo, la = setup_inputs()
    B = Forcing(CONFIG, lo, la, 'B')
    C = Forcing(CONFIG, lo, la, 'C')
    for step in range(3):
        bf = B.calculate(step, 60.0)
        cf = C.calculate(step, 60.0)
        for x, y in zip(bf, cf):
            assert np.array_equal(x, y)
    for step in range(3, 145):
        bf = B.calculate(step, 60.0)
        cf = C.calculate(step, 60.0)
    assert any((np.max(np.abs(x - y)) > 0 for x, y in zip(bf, cf)))

#============
#Test seed reproducibility
#============
def test_seed_reproducibility():
    lo, la = setup_inputs()
    a = Forcing(CONFIG, lo, la, 'B')
    b = Forcing(CONFIG, lo, la, 'B')
    for step in range(5):
        x = a.calculate(step, 60.0)
        y = b.calculate(step, 60.0)
        assert all((np.array_equal(xx, yy) for xx, yy in zip(x, y)))
