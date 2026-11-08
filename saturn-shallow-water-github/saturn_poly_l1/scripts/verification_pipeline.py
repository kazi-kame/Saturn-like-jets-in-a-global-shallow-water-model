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
from assess_runs import assess_balance, read_state, run_metadata, write_report, digest, ROOT
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import verify_config
from verification import stationary_state

#============
#Prepare experiment inputs
#============
def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / 'configs/exploratory_proxy.json').read_text())
    jobs = []
    for size in (32, 64, 128):
        c = copy.deepcopy(cfg)
        c['numerics'].update(ncolat=size, nlon=2 * size)
        jobs.append((f'initial_balance_{2 * size}x{size}', c, True))
    for speed, label in ((0.0, 'rest'), (20.0, 'solid_rotation')):
        for size, dt in ((32, 60), (64, 60), (128, 60), (64, 30), (64, 15)):
            c = copy.deepcopy(cfg)
            c['verification_case'] = {'kind': 'stationary_solid_rotation', 'equatorial_speed_m_s': speed, 'derivation': 'Gradient-wind balance with a prescribed global mean depth'}
            c['numerics'].update(ncolat=size, nlon=2 * size, time_step_s=dt, hyperviscosity_m4_s=0.0, run_days=7200 / 86400, save_every_steps=int(7200 / dt), checkpoint_every_steps=int(7200 / dt))
            jobs.append((f'{label}_{2 * size}x{size}_dt{dt}', c, False))
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail', 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1']
    plan = []
    for name, c, build_only in jobs:
        verify_config(c)
        c['config_id'] = name
        config = output / (name + '.json')
        config.write_text(json.dumps(c, indent=2) + '\n')
        run = output / name
        command = ['python', str(ROOT / 'src/run_l1.py'), '--config', str(config), '--arm', 'A', '--out', str(run)]
        if build_only:
            command.append('--build-only')
        else:
            command += ['--steps', str(round(7200 / c['numerics']['time_step_s']))]
        commands.append(shlex.join(command))
        plan.append({'name': name, 'run': str(run), 'config': str(config), 'config_sha256': digest(config), 'build_only': build_only, 'command': command})
    (output / 'run.sh').write_text('\n'.join(commands) + '\n')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan

#============
#Assess stationary
#============
def assess_stationary(run):
    run = Path(run)
    m = run_metadata(run)
    cfg = json.loads((run / 'config.lock.json').read_text())
    if m['arm'] != 'A' or cfg['numerics']['hyperviscosity_m4_s'] != 0:
        raise ValueError('Not an inviscid control')
    final = m['final_step']
    path = run / 'states' / f'step{final:08d}.nc'
    s = read_state(path)
    if s['step'] != final or not np.isclose(s['time_s'], 7200, rtol=0, atol=1e-09):
        raise ValueError('Wrong verification duration')
    lat = np.deg2rad(s['latitude_planetocentric'])[:, None]
    expected = stationary_state(lat, cfg)
    errors = {}
    for key, value in zip(('u_east', 'v_north', 'layer_thickness'), expected):
        errors[key] = float(np.max(abs(s[key] - value)))
    tolerances = {'u_east': 1e-07, 'v_north': 1e-07, 'layer_thickness': 1e-05}
    return {'kind': 'stationary_verification', 'passed': all((errors[k] <= tolerances[k] for k in errors)), 'max_absolute_errors': errors, 'absolute_screening_tolerances': tolerances, 'evidence_sha256': {str(path): digest(path), str(run / 'manifest.json'): digest(run / 'manifest.json')}, 'scope': 'Analytical stationary solid-rotation case; not independent Williamson/Galewsky validation or nonlinear convergence proof'}

#============
#Command-line interface
#============
def main():
    parser = argparse.ArgumentParser(description='Verification pipeline')
    parser.add_argument('--out', required=True)
    parser.add_argument('--assess', action='store_true', help='Only assess existing outputs from this plan')
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if not args.assess:
        jobs = prepare(output)
        print(f'Prepared {len(jobs)} cases; no solver execution. Run: bash {output}/run.sh')
        return
    plan = json.loads((output / 'plan.json').read_text())
    reports = []
    for job in plan:
        if digest(job['config']) != job['config_sha256']:
            raise ValueError('Changed planned config')
        run = Path(job['run'])
        if digest(run / 'config.lock.json') != job['config_sha256']:
            raise ValueError('Run does not match plan')
        report = assess_balance(run) if job['build_only'] else assess_stationary(run)
        report['case'] = job['name']
        reports.append(report)
    write_report(output / 'assessment.json', {'cases': reports, 'scientifically_validated': False})
    print(json.dumps(reports, indent=2))
if __name__ == '__main__':
    main()
