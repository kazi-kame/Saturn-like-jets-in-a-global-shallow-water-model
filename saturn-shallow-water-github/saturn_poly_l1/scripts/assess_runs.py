#============
#Imports and module setup
#============
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import netcdf_file
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import gradient_wind_balanced_height
from run_l1 import jet_profile

#============
#Digest
#============
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

#============
#Read state
#============
def read_state(path):
    with netcdf_file(path, 'r', mmap=False) as ds:
        result = {name: np.array(ds.variables[name].data, copy=True) for name in ('u_east', 'v_north', 'layer_thickness', 'latitude_planetocentric', 'longitude_east')}
        result['step'] = int(ds.step)
        result['time_s'] = float(ds.sim_time_s)
    if any((not np.isfinite(value).all() for value in result.values())):
        raise ValueError('Nonfinite state')
    return result

#============
#Run metadata
#============
def run_metadata(path, allow_build=False):
    path = Path(path)
    m = json.loads((path / 'manifest.json').read_text())
    if m['status'] not in (('completed', 'built_not_evolved') if allow_build else ('completed',)):
        raise ValueError(f'Incomplete run: {path}')
    if digest(path / 'config.lock.json') != m['config_sha256']:
        raise ValueError('Changed configuration')
    fingerprint = hashlib.sha256(json.dumps(m['compatibility'], sort_keys=True).encode()).hexdigest()
    if fingerprint != m['compatibility_sha256']:
        raise ValueError('Invalid compatibility fingerprint')
    return m

#============
#Assess restart
#============
def assess_restart(root, rtol=1e-09, atol=1e-07):
    root = Path(root).resolve()
    full = root / 'full'
    fm = run_metadata(full)
    statepath = full / 'states/step00000024.nc'
    reference = read_state(statepath)
    if reference['step'] != 24:
        raise ValueError('Expected step 24')
    evidence = {str(statepath): digest(statepath), str(full / 'manifest.json'): digest(full / 'manifest.json')}
    comparisons = []
    for split in (4, 12, 20):
        first = root / f'first_{split}'
        child = root / f'continued_{split}'
        pm = run_metadata(first)
        cm = run_metadata(child)
        if any((m['compatibility_sha256'] != fm['compatibility_sha256'] for m in (pm, cm))):
            raise ValueError('Comparison mixes code/config/arm/environment')
        cp = first / 'checkpoints' / f'step{split:08d}.npz'
        if pm['checkpoints'].get(f'checkpoints/{cp.name}') != digest(cp) or cm['restart_parent']['checkpoint_sha256'] != digest(cp) or cm['restart_parent']['run_id'] != pm['run_id'] or (cm['start_step'] != split):
            raise ValueError('Parent checkpoint link mismatch')
        path = child / 'states/step00000024.nc'
        state = read_state(path)
        for key in ('latitude_planetocentric', 'longitude_east', 'step', 'time_s'):
            if not np.array_equal(state[key], reference[key]):
                raise ValueError('Coordinates or time mismatch')
        errors = {}
        passed = True
        for key in ('u_east', 'v_north', 'layer_thickness'):
            a = state[key]
            b = reference[key]
            if a.shape != b.shape:
                raise ValueError('State shape mismatch')
            delta = np.abs(a - b)
            limit = atol + rtol * np.abs(b)
            errors[key] = {'max_absolute': float(delta.max()), 'max_tolerance_ratio': float((delta / limit).max())}
            passed &= bool(np.all(delta <= limit))
        comparisons.append({'split_step': split, 'passed': passed, 'errors': errors})
        for item in (path, cp, first / 'manifest.json', child / 'manifest.json'):
            evidence[str(item)] = digest(item)
    return {'kind': 'restart', 'arm': fm['arm'], 'passed': all((x['passed'] for x in comparisons)), 'compatibility': fm['compatibility'], 'rtol': rtol, 'atol': atol, 'comparisons': comparisons, 'evidence_sha256': evidence, 'scope': '24-step equivalence for this exact configuration and implementation; not physical validation'}

#============
#Assess balance
#============
def assess_balance(run):
    run = Path(run).resolve()
    m = run_metadata(run, allow_build=True)
    cfg = json.loads((run / 'config.lock.json').read_text())
    if cfg.get('verification_case'):
        raise ValueError('Use proxy jet run for this comparison')
    path = run / 'states/step00000000.nc'
    state = read_state(path)
    if state['step'] != 0 or state['time_s'] != 0:
        raise ValueError('Initial state required')
    p = cfg['physical_constants']
    n = cfg['numerics']
    lat = np.deg2rad(state['latitude_planetocentric'])
    profiles = []
    for count in (16385, 32769):
        fine = np.linspace(-np.pi / 2 + 1e-08, np.pi / 2 - 1e-08, count)
        height = gradient_wind_balanced_height(fine, jet_profile(fine, cfg), p['equatorial_radius_m'], 2 * np.pi / p['rotation_period_s'], p['gravitational_acceleration_m_s2'], n['mean_layer_depth_m'])
        profiles.append(np.interp(lat, fine, height))
    h = state['layer_thickness']
    expected = profiles[-1][:, None]
    return {'kind': 'initial_balance', 'run_id': m['run_id'], 'max_height_error_m': float(np.max(np.abs(h - expected))), 'rms_height_error_m': float(np.sqrt(np.mean((h - expected) ** 2))), 'reference_grid_refinement_difference_m': float(np.max(abs(profiles[1] - profiles[0]))), 'max_zonal_asymmetry_m': float(np.max(abs(h - h.mean(axis=1, keepdims=True)))), 'max_initial_wind_proxy_error_m_s': float(np.max(abs(state['u_east'] - jet_profile(lat, cfg)[:, None]))), 'passed': None, 'scope': 'Independent gradient-wind quadrature comparison; no acceptance threshold or convergence claim', 'evidence_sha256': {str(path): digest(path), str(run / 'manifest.json'): digest(run / 'manifest.json'), str(run / 'config.lock.json'): digest(run / 'config.lock.json')}}

#============
#Write report
#============
def write_report(path, result):
    result = dict(result, assessment_code_sha256=digest(__file__))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = {}
    for source in [Path(__file__), ROOT / 'src/diagnostics.py', ROOT / 'src/run_l1.py', ROOT / 'src/verification.py', ROOT / 'scripts/verification_pipeline.py', ROOT / 'scripts/validate_baseline.py', ROOT / 'src/galewsky.py', ROOT / 'scripts/dynamic_benchmark.py', ROOT / 'scripts/wave_history.py', ROOT / 'scripts/trigger_experiment.py']:
        data = source.read_bytes()
        hash_ = hashlib.sha256(data).hexdigest()
        archived = path.parent / 'assessment_code' / f'{hash_}_{source.name}'
        archived.parent.mkdir(exist_ok=True)
        if not archived.exists():
            archived.write_bytes(data)
        elif archived.read_bytes() != data:
            raise ValueError('Assessment source archive mismatch')
        sources[str(archived.resolve())] = hash_
    result['assessment_source_sha256'] = sources
    with path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')

#============
#Command-line interface
#============
def main():
    p = argparse.ArgumentParser(description='Assess runs')
    p.add_argument('kind', choices=['restart', 'balance'])
    p.add_argument('input')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = assess_restart(a.input) if a.kind == 'restart' else assess_balance(a.input)
    write_report(a.out, result)
    print(json.dumps(result, indent=2))
    return 1 if result.get('passed') is False else 0
if __name__ == '__main__':
    raise SystemExit(main())
