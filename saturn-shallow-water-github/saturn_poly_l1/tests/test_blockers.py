#============
#Imports and module setup
#============
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import verify_config, planetographic_to_planetocentric
from grid_io import grid_copy, set_grid, equation_namespace
from provenance import collect_provenance, load_checkpoint, RunRecord, sha256
from run_l1 import Forcing, main
CONFIG = json.loads((ROOT / 'configs/exploratory_proxy.json').read_text())

#============
#Test invalid configuration
#============
@pytest.mark.parametrize('section,key,value', [('physical_constants', 'rotation_period_s', float('nan')), ('numerics', 'hyperviscosity_m4_s', -1), ('numerics', 'save_every_steps', 0), ('numerics', 'checkpoint_every_steps', True), ('numerics', 'ncolat', 63), ('numerics', 'dealias', 1), ('numerics', 'max_mode', 64), ('forcing', 'mode_max', 64), ('forcing', 'correlation_time_s', 0), ('forcing', 'acceleration_scale_m_s2', float('inf')), ('forcing', 'vortex_center_planetographic_deg', 0), ('forcing', 'forcing_fwhm_planetographic_deg', 50), ('diagnostics', 'min_peak_to_second_ratio', 1)])
def test_invalid_configuration(section, key, value):
    cfg = copy.deepcopy(CONFIG)
    cfg[section][key] = value
    with pytest.raises(ValueError):
        verify_config(cfg)

#============
#Test vortex center curl opposes coriolis
#============
@pytest.mark.parametrize('hemisphere', [-1, 1])
def test_vortex_center_curl_opposes_coriolis(hemisphere):
    cfg = copy.deepcopy(CONFIG)
    cfg['forcing']['vortex_center_planetographic_deg'] = 55 * hemisphere
    cfg['forcing']['acceleration_scale_m_s2'] = 0
    p = cfg['physical_constants']
    f = cfg['forcing']
    phi = np.deg2rad(planetographic_to_planetocentric(f['vortex_center_planetographic_deg'], p['equatorial_radius_m'], p['polar_radius_m']))
    lam = np.deg2rad(f['vortex_center_lon_deg'])
    eps = 1e-05
    lo, la = np.meshgrid(lam + np.array([-eps, 0, eps]), phi + np.array([-eps, 0, eps]), indexing='ij')
    force = Forcing(cfg, lo, la, 'C')
    dt = cfg['numerics']['time_step_s']
    step = int(f['vortex_begin_day'] * 86400 / dt) + 1
    east, north = force.calculate(step, dt)
    curl = ((north[2, 1] - north[0, 1]) / (2 * eps) - (east[1, 2] * np.cos(la[1, 2]) - east[1, 0] * np.cos(la[1, 0])) / (2 * eps)) / (p['equatorial_radius_m'] * np.cos(phi))
    exact = -2 * hemisphere * f['vortex_acceleration_scale_m_s2'] / (p['equatorial_radius_m'] * np.deg2rad(f['vortex_sigma_deg']) ** 2)
    assert np.isclose(curl, exact, rtol=1e-06, atol=0)
    assert curl * np.sin(phi) < 0

#============
#Test verification trigger crosses onset and shutoff
#============
def test_verification_trigger_crosses_onset_and_shutoff():
    spec = importlib.util.spec_from_file_location('restart_script', ROOT / 'scripts/integration_test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.verification_config()
    verify_config(cfg)
    lo, la = np.meshgrid(np.linspace(0, 6, 12), np.linspace(-1.4, 1.4, 10), indexing='ij')
    b = Forcing(cfg, lo, la, 'B')
    c = Forcing(cfg, lo, la, 'C')
    for step in range(24):
        bf = b.calculate(step, 60)
        cf = c.calculate(step, 60)
        equal = all((np.array_equal(x, y) for x, y in zip(bf, cf)))
        assert equal == (step < 6 or step >= 18)

#============
#Field
#============
class Field:

    #============
    #Initialize object state
    #============
    def __init__(self):
        self.scale = 1.5
        self.data = np.zeros((2, 4, 6))

    #============
    #Change scales
    #============
    def change_scales(self, value):
        self.scale = value

    #============
    #Getitem
    #============
    def __getitem__(self, key):
        assert key == 'g' and self.scale == 1
        return self.data

#============
#Test explicit grid exchange after dealias and copy isolation
#============
def test_explicit_grid_exchange_after_dealias_and_copy_isolation():
    field = Field()
    set_grid(field, np.ones((2, 4, 6)))
    field.scale = 1.5
    result = grid_copy(field)
    result[0] = 7
    assert np.all(field.data == 1)
    with pytest.raises(ValueError):
        set_grid(field, np.ones((2, 6, 9)))
    with pytest.raises(ValueError):
        set_grid(field, np.full((2, 4, 6), np.nan))

#============
#Test operator namespace cannot mask dt
#============
def test_operator_namespace_cannot_mask_dt():
    ns = equation_namespace(1, 2, 3, 4, 5, 6, 7, lambda x: x)
    assert not {'dt', 'lap', 'grad', 'div', 'ave'} & ns.keys()
    assert eval('dt(u)', dict(dt=lambda x: 2 * x, **ns)) == 2

#============
#Fixture record
#============
def fixture_record(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg['numerics'].update(nlon=8, ncolat=8, max_mode=2)
    cfg['forcing'].update(mode_min=1, mode_max=2)
    raw = json.dumps(cfg).encode()
    manifest, inputs = collect_provenance(cfg, raw, 'C')
    record = RunRecord(tmp_path / 'parent', manifest, inputs, raw)
    lo, la = np.meshgrid(np.arange(8) * 2 * np.pi / 8, np.linspace(-1.4, 1.4, 8), indexing='ij')
    force = Forcing(cfg, lo, la, 'C')
    return (cfg, raw, record, force)

#============
#Test checkpoint rng roundtrip and parent preservation
#============
def test_checkpoint_rng_roundtrip_and_parent_preservation(tmp_path):
    cfg, raw, record, force = fixture_record(tmp_path)
    for step in range(4):
        force.calculate(step, 60)
    with record:
        record.checkpoint(np.zeros((2, 8, 8)), np.ones((8, 8)) * 20000, force, 4, 240.0)
        record.update(status='completed')
    checkpoint = record.output / 'checkpoints/step00000004.npz'
    before = {str(p): sha256(p.read_bytes()) for p in record.output.rglob('*') if p.is_file()}
    state = load_checkpoint(checkpoint, record.manifest['compatibility_sha256'], cfg)
    restored = Forcing(cfg, force.lon, force.lat, 'C')
    restored.coeff = state['forcing_coeff']
    restored.rng.bit_generator.state = state['rng_state']
    assert all((np.array_equal(x, y) for x, y in zip(force.calculate(4, 60), restored.calculate(4, 60))))
    manifest, inputs = collect_provenance(cfg, raw, 'C')
    with RunRecord(tmp_path / 'continued', manifest, inputs, raw, state) as child:
        child.update(status='built_not_evolved')
    assert child.manifest['restart_parent']['run_id'] == record.manifest['run_id']
    assert before == {str(p): sha256(p.read_bytes()) for p in record.output.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='mismatch'):
        load_checkpoint(checkpoint, 'changed', cfg)
    checkpoint.write_bytes(checkpoint.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(checkpoint, record.manifest['compatibility_sha256'], cfg)

#============
#Test existing directory preserved and failures recorded
#============
def test_existing_directory_preserved_and_failures_recorded(tmp_path):
    cfg, raw, record, force = fixture_record(tmp_path)
    with pytest.raises(RuntimeError, match='test failure'):
        with record:
            raise RuntimeError('test failure')
    before = (record.output / 'manifest.json').read_bytes()
    assert json.loads(before)['status'] == 'failed'
    with pytest.raises(FileExistsError):
        with record:
            pass
    assert (record.output / 'manifest.json').read_bytes() == before

#============
#Test restart fingerprint tracks inputs
#============
@pytest.mark.parametrize('change', ['config', 'arm', 'code'])
def test_restart_fingerprint_tracks_inputs(tmp_path, monkeypatch, change):
    cfg, raw, record, force = fixture_record(tmp_path)
    if change == 'code':
        original = Path.read_bytes
        target = ROOT / 'src/run_l1.py'
        monkeypatch.setattr(Path, 'read_bytes', lambda path: original(path) + (b'\n' if path == target else b''))
    new, _ = collect_provenance(cfg, raw + b' ' if change == 'config' else raw, 'A' if change == 'arm' else 'C')
    assert new['compatibility_sha256'] != record.manifest['compatibility_sha256']

#============
#Test config cli never imports solver or creates output
#============
def test_config_cli_never_imports_solver_or_creates_output(tmp_path, monkeypatch):
    import run_l1

    #============
    #Forbidden
    #============
    def forbidden():
        raise AssertionError('environment/solver access during config validation')
    monkeypatch.setattr(run_l1, 'check_environment', forbidden)
    assert main(['--validate-config', '--out', str(tmp_path / 'unused')]) == 0
    assert not (tmp_path / 'unused').exists()
