#============
#Imports and module setup
#============
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.io import netcdf_file
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from assess_runs import assess_restart, digest, write_report
from verification_pipeline import prepare
from verification import stationary_state
from diagnostics import verify_config

#============
#Test stationary height satisfies independent gradient wind equation
#============
def test_stationary_height_satisfies_independent_gradient_wind_equation():
    cfg = json.loads((ROOT / 'configs/exploratory_proxy.json').read_text())
    cfg['verification_case'] = {'kind': 'stationary_solid_rotation', 'equatorial_speed_m_s': 20.0}
    cfg['numerics']['hyperviscosity_m4_s'] = 0
    phi = np.linspace(-1.4, 1.4, 10001)
    u, v, h = stationary_state(phi, cfg)
    p = cfg['physical_constants']
    gradient = np.gradient(h, phi, edge_order=2)
    tendency = -(p['equatorial_radius_m'] * 2 * (2 * np.pi / p['rotation_period_s']) * np.sin(phi) * u + u * u * np.tan(phi)) / p['gravitational_acceleration_m_s2']
    np.testing.assert_allclose(gradient[1:-1], tendency[1:-1], rtol=1e-06, atol=1e-07)
    assert h.min() > 0 and (not np.any(v))
    cfg['numerics']['hyperviscosity_m4_s'] = 1
    with pytest.raises(ValueError):
        verify_config(cfg)

#============
#Test prepare does not run and keeps interventions separate
#============
def test_prepare_does_not_run_and_keeps_interventions_separate(tmp_path):
    plan = prepare(tmp_path / 'plan')
    assert len(plan) == 13
    assert sum((job['build_only'] for job in plan)) == 3
    for job in plan:
        assert not Path(job['run']).exists()
        cfg = json.loads(Path(job['config']).read_text())
        verify_config(cfg)
        if not job['build_only']:
            assert float(job['command'][-1]) * cfg['numerics']['time_step_s'] == 7200
            assert cfg['numerics']['hyperviscosity_m4_s'] == 0
    with pytest.raises(FileExistsError):
        prepare(tmp_path / 'plan')

#============
#Fixture suite
#============
def fixture_suite(root):
    raw = b'{}'
    compat = {'arm': 'A'}
    fingerprint = hashlib.sha256(json.dumps(compat, sort_keys=True).encode()).hexdigest()

    #============
    #Create
    #============
    def create(name):
        run = root / name
        run.mkdir(parents=True)
        (run / 'config.lock.json').write_bytes(raw)
        m = {'status': 'completed', 'run_id': name, 'arm': 'A', 'config_sha256': digest(run / 'config.lock.json'), 'compatibility': compat, 'compatibility_sha256': fingerprint}
        return (run, m)

    #============
    #State
    #============
    def state(run, change=0):
        folder = run / 'states'
        folder.mkdir(exist_ok=True)
        with netcdf_file(folder / 'step00000024.nc', 'w') as f:
            f.createDimension('lat', 3)
            f.createDimension('lon', 4)
            f.createVariable('latitude_planetocentric', 'd', ('lat',))[:] = [-60, 0, 60]
            f.createVariable('longitude_east', 'd', ('lon',))[:] = [0, 90, 180, 270]
            for key in ('u_east', 'v_north', 'layer_thickness'):
                f.createVariable(key, 'd', ('lat', 'lon'))[:] = np.ones((3, 4)) + change
            f.step = 24
            f.sim_time_s = 1440.0
    full, m = create('full')
    state(full)
    (full / 'manifest.json').write_text(json.dumps(m))
    for split in (4, 12, 20):
        first, pm = create(f'first_{split}')
        cp = first / 'checkpoints' / f'step{split:08d}.npz'
        cp.parent.mkdir()
        cp.write_bytes(b'checkpoint fixture')
        pm['checkpoints'] = {f'checkpoints/{cp.name}': digest(cp)}
        (first / 'manifest.json').write_text(json.dumps(pm))
        child, cm = create(f'continued_{split}')
        state(child)
        cm.update(start_step=split, restart_parent={'run_id': pm['run_id'], 'checkpoint_sha256': digest(cp)})
        (child / 'manifest.json').write_text(json.dumps(cm))
    return state

#============
#Test restart assessor detects changed field and parent
#============
def test_restart_assessor_detects_changed_field_and_parent(tmp_path):
    state = fixture_suite(tmp_path)
    assert assess_restart(tmp_path)['passed']
    state(tmp_path / 'continued_12', change=0.001)
    assert not assess_restart(tmp_path)['passed']
    cp = tmp_path / 'first_4/checkpoints/step00000004.npz'
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        assess_restart(tmp_path)

#============
#Test report preserves existing file
#============
def test_report_preserves_existing_file(tmp_path):
    path = tmp_path / 'result.json'
    write_report(path, {'passed': None})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_report(path, {'passed': True})
    assert path.read_bytes() == original
