#============
#Imports and module setup
#============
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from wave_history import modal_history, post_trigger_summary
from trigger_experiment import prepare
from run_l1 import Forcing

#============
#Test modal translation growth and phase wrap
#============
def test_modal_translation_growth_and_phase_wrap():
    times = np.arange(4) * 600.0
    m = 6
    omega = 1e-05
    growth = 2e-05
    z = np.zeros((4, 8), complex)
    z[:, m - 1] = 2 * np.exp(growth * times) * np.exp(1j * (3.13 - m * omega * times))
    result = modal_history(times, z, 0.01)
    for row in result['intervals']:
        mode = row['modes'][m - 1]
        assert mode['principal_phase_drift_deg_per_hour'] == pytest.approx(omega * 180 / np.pi * 3600)
        assert mode['log_amplitude_change_per_hour'] == pytest.approx(growth * 3600)
    assert not result['phase_aliasing_resolved']

#============
#Test weak samples do not supply or bridge phase rates
#============
def test_weak_samples_do_not_supply_or_bridge_phase_rates():
    result = modal_history([0, 1, 2, 3], np.array([[1], [0], [1], [1j]]), 0.01)
    assert result['intervals'][0]['modes'][0]['principal_phase_drift_deg_per_hour'] is None
    assert result['intervals'][1]['modes'][0]['principal_phase_drift_deg_per_hour'] is None
    assert result['intervals'][2]['modes'][0]['principal_phase_drift_deg_per_hour'] is not None
    with pytest.raises(ValueError):
        modal_history([0, 0], np.ones((2, 1)), 0.01)

#============
#Test phase alias remains explicit
#============
def test_phase_alias_remains_explicit():
    t = np.array([0.0, 3600.0])
    z = np.exp(1j * 2 * np.pi * t / 3600)[:, None]
    result = modal_history(t, z, 0.01)
    assert result['intervals'][0]['modes'][0]['principal_phase_drift_deg_per_hour'] == pytest.approx(0, abs=1e-12)
    assert not result['coherent_pattern_speed_established']
    assert result['intervals'][0]['modes'][0]['phase_branch_spacing_deg_per_hour'] == 360

#============
#Test post trigger ratios require boundary and later samples
#============
def test_post_trigger_ratios_require_boundary_and_later_samples():
    rows = [{'time_s': t, 'max_height_difference_m': a, 'global_height_rms_m': a / 2, 'global_velocity_rms_m_s': 0} for t, a in [(0, 0), (3, 2), (6, 1)]]
    assert post_trigger_summary(rows, 7)['final_to_shutoff_ratios'] is None
    result = post_trigger_summary(rows, 3)
    assert result['final_to_shutoff_ratios']['max_height_difference_m'] == 0.5
    assert result['final_to_shutoff_ratios']['global_velocity_rms_m_s'] is None

#============
#Test short trigger plan and force shutoff
#============
def test_short_trigger_plan_and_force_shutoff(tmp_path):
    cfg, jobs = prepare(tmp_path / 'plan')
    assert len(jobs) == 2 and all((not Path(j['run']).exists() for j in jobs))
    assert cfg['numerics']['save_every_steps'] * cfg['numerics']['time_step_s'] == 360
    f = cfg['forcing']
    assert (f['vortex_begin_day'] + f['vortex_duration_days']) * 86400 == 10800
    lon, lat = np.meshgrid(np.linspace(0, 6, 8), np.linspace(-1.4, 1.4, 8), indexing='ij')
    b = Forcing(cfg, lon, lat, 'B')
    c = Forcing(cfg, lon, lat, 'C')
    for step, active in [(575, False), (576, True), (719, True), (720, False), (721, False)]:
        fb = b.calculate(step, 15)
        fc = c.calculate(step, 15)
        assert any((not np.array_equal(x, y) for x, y in zip(fb, fc))) == active
        assert np.array_equal(b.coeff, c.coeff)
    with pytest.raises(FileExistsError):
        prepare(tmp_path / 'plan')
