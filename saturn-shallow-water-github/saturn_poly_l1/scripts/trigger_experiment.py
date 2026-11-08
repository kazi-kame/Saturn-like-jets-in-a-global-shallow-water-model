#============
#Imports and module setup
#============
import argparse
import json
from pathlib import Path
import shlex
import sys
from assess_runs import ROOT, digest
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import verify_config

#============
#Prepare experiment inputs
#============
def prepare(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / 'configs/saturn_working_1024x512.json').read_text())
    cfg['config_id'] = 'matched_trigger_off_3h'
    cfg['numerics'].update(run_days=0.25, save_every_steps=24, checkpoint_every_steps=720)
    cfg['forcing'].update(vortex_begin_day=0.1, vortex_duration_days=0.025)
    cfg['experiment_protocol'] = {'trigger_on_s': 8640, 'trigger_off_s': 10800, 'end_s': 21600, 'output_cadence_s': 360, 'same_stochastic_seed_and_history': True}
    if cfg['numerics']['time_step_s'] != 15 or (cfg['numerics']['nlon'], cfg['numerics']['ncolat']) != (1024, 512):
        raise ValueError('Protocol requires recorded 1024x512, dt15 working configuration')
    verify_config(cfg)
    config = out / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail', 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1']
    jobs = []
    for arm in ('B', 'C'):
        run = out / f'saturn_{arm}_6h'
        command = ['python', str(ROOT / 'src/run_l1.py'), '--config', str(config), '--arm', arm, '--steps', '1440', '--out', str(run)]
        commands.append(shlex.join(command))
        jobs.append({'arm': arm, 'run': str(run), 'command': command})
    (out / 'run.sh').write_text('\n'.join(commands) + '\n')
    (out / 'plan.json').write_text(json.dumps({'protocol': cfg['experiment_protocol'], 'config': str(config), 'config_sha256': digest(config), 'jobs': jobs, 'execution_status': 'prepared_not_run'}, indent=2) + '\n')
    return (cfg, jobs)

#============
#Command-line interface
#============
def main():
    p = argparse.ArgumentParser(description='Trigger experiment')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    prepare(a.out)
    print(f'Prepared {a.out}/run.sh. Trigger on 2.4 h, off 3 h; observe through 6 h. No simulation run.')
if __name__ == '__main__':
    main()
