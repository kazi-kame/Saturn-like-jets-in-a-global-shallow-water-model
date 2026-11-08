#============
#Imports and module setup
#============
from pathlib import Path
import sys
import json
import numpy as np
import pytest
from scipy.special import roots_legendre
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from galewsky import jet, balanced_height, balance_gradient, perturbation, initial_state, PHI0, PHI1, MEAN_DEPTH
from dynamic_benchmark import benchmark_config, prepare, weighted_difference
from diagnostics import verify_config

#============
#Test compact jet and perturbation conventions
#============
def test_compact_jet_and_perturbation_conventions():
    np.testing.assert_allclose(jet(np.array([0, PHI0, np.pi / 4, PHI1, np.pi / 2])), [0, 0, 80, 0, 0], atol=1e-12)
    assert perturbation(0, np.pi / 4) == pytest.approx(120 / np.sqrt(2))
    np.testing.assert_allclose(perturbation(0, 0.7), perturbation(2 * np.pi, 0.7), rtol=1e-14)
    assert perturbation(np.pi, np.pi / 4) < 1e-30

#============
#Test independent quadrature mean gradient and refinement
#============
def test_independent_quadrature_mean_gradient_and_refinement():
    x, w = roots_legendre(512)
    lat = np.arcsin(x)
    h = balanced_height(lat)
    assert np.dot(w, h) / 2 == pytest.approx(MEAN_DEPTH, abs=1e-08)
    np.testing.assert_allclose(h, balanced_height(lat, 256), rtol=0, atol=1e-08)
    phi = np.linspace(PHI0 + 0.01, PHI1 - 0.01, 100)
    eps = 1e-05
    derivative = (balanced_height(phi + eps) - balanced_height(phi - eps)) / (2 * eps)
    np.testing.assert_allclose(derivative, balance_gradient(phi), rtol=2e-05, atol=0.0001)

#============
#Test source constants and inviscid window are enforced
#============
def test_source_constants_and_inviscid_window_are_enforced():
    cfg = benchmark_config(30)
    verify_config(cfg)
    cfg['numerics']['hyperviscosity_m4_s'] = 1
    with pytest.raises(ValueError):
        verify_config(cfg)
    cfg = benchmark_config(30)
    cfg['physical_constants']['equatorial_radius_m'] += 1
    with pytest.raises(ValueError):
        verify_config(cfg)
    cfg = benchmark_config(30)
    cfg['numerics']['run_days'] = 1
    with pytest.raises(ValueError):
        verify_config(cfg)

#============
#Test preparation never evolves and keeps saturn separate
#============
def test_preparation_never_evolves_and_keeps_saturn_separate(tmp_path):
    jobs = prepare(tmp_path / 'plan')
    assert len(jobs) == 2
    for job in jobs:
        cfg = json.loads(Path(job['config']).read_text())
        assert cfg['numerics']['nlon'] == 1024 and cfg['numerics']['ncolat'] == 512
        assert not Path(job['run']).exists()
    assert 'saturn_A_6h' not in (tmp_path / 'plan/benchmark.sh').read_text()
    assert 'saturn_A_6h' in (tmp_path / 'plan/saturn_6h.sh').read_text()
    with pytest.raises(FileExistsError):
        prepare(tmp_path / 'plan')

#============
#Test weighted difference has physical absolute norm and null zero norm
#============
def test_weighted_difference_has_physical_absolute_norm_and_null_zero_norm():
    x, w = roots_legendre(16)
    lat = np.rad2deg(np.arcsin(x))
    ones = np.ones((16, 32))
    error = weighted_difference(ones, 2 * ones, lat)
    assert error['absolute_rms'] == pytest.approx(1)
    assert error['relative_l2'] == pytest.approx(0.5)
    assert weighted_difference(ones, 0 * ones, lat)['relative_l2'] is None
    with pytest.raises(ValueError):
        weighted_difference(ones, ones, lat[::-1])

#============
#Test real benchmark build only outputs declared fields
#============
def test_real_benchmark_build_only_outputs_declared_fields(tmp_path, monkeypatch):
    pytest.importorskip('dedalus.public')
    from dedalus.core.solvers import InitialValueSolver
    from run_l1 import main
    from scipy.io import netcdf_file

    #============
    #Forbidden
    #============
    def forbidden(*args, **kwargs):
        raise AssertionError('Fluid evolution prohibited in test')
    monkeypatch.setattr(InitialValueSolver, 'step', forbidden)
    cfg = benchmark_config(30)
    cfg['numerics'].update(nlon=128, ncolat=64)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    out = tmp_path / 'build'
    assert main(['--config', str(path), '--build-only', '--out', str(out)]) == 0
    with netcdf_file(out / 'states/step00000000.nc', 'r', mmap=False) as ds:
        assert ds.planet == b'Earth benchmark sphere'
        assert {'divergence', 'relative_vorticity', 'height_minus_initial_balance'} <= ds.variables.keys()
        lat = np.deg2rad(ds.variables['latitude_planetocentric'].data)[:, None]
        lon = np.deg2rad(ds.variables['longitude_east'].data)[None, :]
        expected = initial_state(lon, lat)[2]
        actual = ds.variables['layer_thickness'].data
        assert np.isfinite(actual).all() and actual.min() > 0
        assert abs(actual.mean() - expected.mean()) < 1

#============
#Test saved pair assessment reports sensitivity without claiming validation
#============
def test_saved_pair_assessment_reports_sensitivity_without_claiming_validation(tmp_path):
    from scipy.io import netcdf_file
    from dynamic_benchmark import assess
    from assess_runs import digest
    import hashlib
    out = tmp_path / 'plan'
    jobs = prepare(out)
    x, w = roots_legendre(8)
    lat = np.rad2deg(np.arcsin(x))
    lon = np.arange(16) * 360 / 16
    for i, job in enumerate(jobs):
        run = Path(job['run'])
        (run / 'states').mkdir(parents=True)
        cfg = json.loads(Path(job['config']).read_text())
        cfg['numerics'].update(nlon=16, ncolat=8, max_mode=3)
        cfg['forcing']['mode_max'] = 3
        raw = json.dumps(cfg).encode()
        Path(job['config']).write_bytes(raw)
        job['config_sha256'] = digest(job['config'])
        (run / 'config.lock.json').write_bytes(raw)
        dt = cfg['numerics']['time_step_s']
        compat = {'code_sha256': {'fixture': 'same'}, 'packages': {}, 'config_sha256': digest(job['config'])}
        m = {'status': 'completed', 'arm': 'A', 'compatibility': compat, 'compatibility_sha256': hashlib.sha256(json.dumps(compat, sort_keys=True).encode()).hexdigest(), 'config_sha256': digest(job['config']), 'final_step': int(21600 / dt)}
        (run / 'manifest.json').write_text(json.dumps(m))
        for hours in (0, 2, 4, 6):
            step = int(hours * 3600 / dt)
            with netcdf_file(run / 'states' / f'step{step:08d}.nc', 'w') as ds:
                ds.createDimension('lat', 8)
                ds.createDimension('lon', 16)
                ds.createVariable('latitude_planetocentric', 'd', ('lat',))[:] = lat
                ds.createVariable('longitude_east', 'd', ('lon',))[:] = lon
                for name in ('u_east', 'v_north', 'layer_thickness', 'divergence', 'relative_vorticity', 'height_minus_initial_balance'):
                    ds.createVariable(name, 'd', ('lat', 'lon'))[:] = np.full((8, 16), 1 + i)
                ds.step = step
                ds.sim_time_s = hours * 3600.0
    (out / 'plan.json').write_text(json.dumps(jobs))
    result = assess(out)
    assert result['independent_reference_match'] is None
    assert result['timestep_convergence_passed'] is None
    assert not result['scientifically_validated']
    assert result['rows'][1]['dt30_minus_dt15']['u_east']['absolute_rms'] == pytest.approx(1)
    with pytest.raises(FileExistsError):
        assess(out)
