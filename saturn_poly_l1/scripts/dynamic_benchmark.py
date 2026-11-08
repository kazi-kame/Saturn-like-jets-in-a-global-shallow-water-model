#============
#Imports and module setup
#============
import argparse
import copy
import json
from pathlib import Path
import shlex
import sys
import numpy as np
from scipy.special import roots_legendre
from scipy.io import netcdf_file
from assess_runs import ROOT, digest, run_metadata, write_report
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import verify_config
from galewsky import RADIUS, OMEGA, GRAVITY, MEAN_DEPTH
THREADS = 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1'

#============
#Benchmark config
#============
def benchmark_config(dt):
    c = json.loads((ROOT / 'configs/exploratory_proxy.json').read_text())
    c['config_id'] = f'galewsky_adjustment_1024x512_dt{dt}'
    c['verification_case'] = {'kind': 'galewsky_adjustment'}
    c['physical_constants'] = dict(equatorial_radius_m=RADIUS, polar_radius_m=RADIUS, rotation_period_s=2 * np.pi / OMEGA, gravitational_acceleration_m_s2=GRAVITY)
    c['numerics'].update(nlon=1024, ncolat=512, time_step_s=float(dt), run_days=0.25, mean_layer_depth_m=MEAN_DEPTH, hyperviscosity_m4_s=0.0, save_every_steps=int(7200 / dt), checkpoint_every_steps=int(7200 / dt))
    verify_config(c)
    return c

#============
#Prepare experiment inputs
#============
def prepare(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for dt in (30, 15):
        c = benchmark_config(dt)
        name = f'galewsky_dt{dt}'
        path = out / (name + '.json')
        path.write_text(json.dumps(c, indent=2) + '\n')
        command = ['python', str(ROOT / 'src/run_l1.py'), '--config', str(path), '--out', str(out / name)]
        jobs.append({'name': name, 'config': str(path), 'config_sha256': digest(path), 'run': str(out / name), 'command': command})
    (out / 'benchmark.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + THREADS + '\n' + '\n'.join((shlex.join(j['command']) for j in jobs)) + '\n')
    (out / 'plan.json').write_text(json.dumps(jobs, indent=2) + '\n')
    working = out / 'saturn_working_1024x512.json'
    working.write_bytes((ROOT / 'configs/saturn_working_1024x512.json').read_bytes())
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail', THREADS]
    for arm in ('A', 'B', 'C'):
        command = ['python', str(ROOT / 'src/run_l1.py'), '--config', str(working), '--arm', arm, '--steps', '1440', '--out', str(out / f'saturn_{arm}_6h')]
        commands.append(shlex.join(command))
    (out / 'saturn_6h.sh').write_text('\n'.join(commands) + '\n')
    (out / 'saturn_plan.json').write_text(json.dumps({'config': str(working), 'config_sha256': digest(working), 'steps': 1440, 'duration_s': 21600, 'arms': ['A', 'B', 'C'], 'note': 'Exploratory six-hour runs; C trigger active after 8640 s, does not reach shutoff.'}, indent=2) + '\n')
    return jobs

#============
#Load snapshot
#============
def load_snapshot(run, hours):
    run = Path(run)
    cfg = json.loads((run / 'config.lock.json').read_text())
    step_float = hours * 3600 / cfg['numerics']['time_step_s']
    step = round(step_float)
    if not np.isclose(step, step_float):
        raise ValueError('Requested time not on timestep')
    path = run / 'states' / f'step{step:08d}.nc'
    with netcdf_file(path, 'r', mmap=False) as ds:
        if int(ds.step) != step or not np.isclose(float(ds.sim_time_s), hours * 3600):
            raise ValueError('Wrong snapshot time')
        names = ['u_east', 'v_north', 'layer_thickness', 'divergence', 'relative_vorticity', 'height_minus_initial_balance', 'latitude_planetocentric', 'longitude_east']
        arrays = {name: np.array(ds.variables[name].data, copy=True) for name in names}
    if any((not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError('Nonfinite snapshot')
    n = cfg['numerics']
    if arrays['latitude_planetocentric'].shape != (n['ncolat'],) or arrays['longitude_east'].shape != (n['nlon'],):
        raise ValueError('Snapshot coordinate counts do not match configuration')
    for name in names[:-2]:
        if arrays[name].shape != (n['ncolat'], n['nlon']):
            raise ValueError('Snapshot field shape mismatch')
    return (arrays, path)

#============
#Weighted difference
#============
def weighted_difference(a, b, latitude):
    nodes, weights = roots_legendre(len(latitude))
    if not np.allclose(np.sin(np.deg2rad(latitude)), nodes, rtol=0, atol=1e-12):
        raise ValueError('Gaussian latitude grid required for quadrature')
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(latitude):
        raise ValueError('Incompatible comparison arrays')
    rms = lambda x: float(np.sqrt(np.sum(weights * np.mean(x * x, axis=1)) / 2))
    error = rms(a - b)
    norm = rms(b)
    return {'absolute_rms': error, 'relative_l2': error / norm if norm > 0 else None, 'max_absolute': float(np.max(abs(a - b)))}

#============
#Assess
#============
def assess(out):
    out = Path(out).resolve()
    jobs = json.loads((out / 'plan.json').read_text())
    if [j['name'] for j in jobs] != ['galewsky_dt30', 'galewsky_dt15']:
        raise ValueError('Unexpected plan')
    evidence = {str(out / 'plan.json'): digest(out / 'plan.json')}
    configs = []
    manifests = []
    for job, expected_dt in zip(jobs, (30, 15)):
        run = Path(job['run'])
        m = run_metadata(run)
        manifests.append(m)
        if digest(job['config']) != job['config_sha256'] or digest(run / 'config.lock.json') != job['config_sha256']:
            raise ValueError('Changed planned or executed config')
        c = json.loads((run / 'config.lock.json').read_text())
        verify_config(c)
        configs.append(c)
        if c['numerics']['time_step_s'] != expected_dt:
            raise ValueError('Unexpected comparison timestep')
        if m['arm'] != 'A' or c.get('verification_case', {}).get('kind') != 'galewsky_adjustment':
            raise ValueError('Not benchmark arm A')
        if m['final_step'] * c['numerics']['time_step_s'] != 21600:
            raise ValueError('Incomplete benchmark window')
        for path in (run / 'manifest.json', run / 'config.lock.json'):
            evidence[str(path)] = digest(path)
    for key in ('code_sha256', 'packages'):
        if manifests[0]['compatibility'][key] != manifests[1]['compatibility'][key]:
            raise ValueError('Incompatible run implementations')
    for c in configs:
        c.pop('config_id', None)
        for key in ('time_step_s', 'save_every_steps', 'checkpoint_every_steps'):
            c['numerics'].pop(key)
    if configs[0] != configs[1]:
        raise ValueError('Paired runs differ beyond timestep/output cadence')
    rows = []
    for hours in (0, 2, 4, 6):
        a, pa = load_snapshot(jobs[0]['run'], hours)
        b, pb = load_snapshot(jobs[1]['run'], hours)
        for key in ('latitude_planetocentric', 'longitude_east'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError('Comparison grids differ')
        errors = {key: weighted_difference(a[key], b[key], a['latitude_planetocentric']) for key in ('u_east', 'v_north', 'layer_thickness', 'divergence', 'relative_vorticity', 'height_minus_initial_balance')}
        rows.append({'hours': hours, 'dt30_minus_dt15': errors})
        evidence[str(pa)] = digest(pa)
        evidence[str(pb)] = digest(pb)
    result = {'kind': 'galewsky_timestep_comparison', 'rows': rows, 'evidence_sha256': evidence, 'independent_reference_match': None, 'timestep_convergence_passed': None, 'scientifically_validated': False, 'note': 'Two timesteps measure sensitivity, not convergence order. No external numerical reference arrays compared. No external field comparison has been performed.'}
    write_report(out / 'comparison.json', result)
    return result

#============
#Command-line interface
#============
def main():
    p = argparse.ArgumentParser(description='Dynamic benchmark')
    p.add_argument('--out', required=True)
    p.add_argument('--assess', action='store_true')
    a = p.parse_args()
    if a.assess:
        print(json.dumps(assess(a.out), indent=2))
    else:
        prepare(a.out)
        print('Prepared benchmark.sh and separate saturn_6h.sh; no fluid solver executed.')
if __name__ == '__main__':
    main()
