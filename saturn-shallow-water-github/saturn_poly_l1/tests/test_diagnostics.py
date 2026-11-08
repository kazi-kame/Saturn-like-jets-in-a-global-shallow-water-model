#============
#Imports and module setup
#============
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from diagnostics import planetographic_to_planetocentric, ring_modes, global_mass_energy, verify_config

#============
#Test latitude conversion
#============
def test_latitude_conversion():
    assert abs(planetographic_to_planetocentric(0, 60268000, 54364000)) < 1e-12
    assert abs(planetographic_to_planetocentric(90, 60268000, 54364000) - 90) < 1e-12
    assert np.isclose(planetographic_to_planetocentric(-60.5, 60268000, 54364000), -planetographic_to_planetocentric(60.5, 60268000, 54364000))

#============
#Test modes synthetic
#============
def test_modes_synthetic():
    l = np.arange(128) * 2 * np.pi / 128
    for m in [6, 10]:
        result = ring_modes(2 * np.cos(m * l + 0.3))
        assert result['m'] == m and result['classification'] == 'dominant_mode'
    assert ring_modes(np.zeros(128))['classification'] == 'none'
    assert ring_modes(np.cos(6 * l) + np.cos(10 * l))['classification'] == 'mixed'

#============
#Test axisymmetric initial ring
#============
def test_axisymmetric_initial_ring():
    assert ring_modes(np.full(128, 40000.0))['m'] is None

#============
#Test spherical mass
#============
def test_spherical_mass():
    radius = 60000000.0
    lat = np.linspace(-np.pi / 2 + np.pi / 64, np.pi / 2 - np.pi / 64, 32)
    lon = np.arange(64) * 2 * np.pi / 64
    h = np.full((32, 64), 20000.0)
    x = np.zeros_like(h)
    results = global_mass_energy(x, x, h, lat, lon, radius, 11.19)
    assert np.isclose(results['layer_volume_m3'], 4 * np.pi * radius ** 2 * 20000, rtol=1e-12)

#============
#Test config
#============
def test_config():
    path = Path(__file__).resolve().parents[1] / 'configs/exploratory_proxy.json'
    assert verify_config(json.loads(path.read_text()))

#============
#Test independent gradient wind balance
#============
def test_independent_gradient_wind_balance():
    from diagnostics import gradient_wind_balanced_height
    lat = np.linspace(-np.pi / 2 + 0.05, np.pi / 2 - 0.05, 4001)
    wind = 100 * np.exp(-0.5 * ((lat - np.deg2rad(75)) / 0.045) ** 2)
    h = gradient_wind_balanced_height(lat, wind, 60268000.0, 2 * np.pi / 38361.6, 11.19, 20000)
    model_grad = np.gradient(h, lat)
    exact_grad = -(60268000.0 * 2 * (2 * np.pi / 38361.6) * np.sin(lat) * wind + wind * wind * np.tan(lat)) / 11.19
    assert np.max(np.abs(model_grad - exact_grad)) < 0.05 * np.max(abs(exact_grad))
    assert h.min() > 0
