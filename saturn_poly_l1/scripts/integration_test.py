#============
#Imports and module setup
#============
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assess_runs import assess_restart, write_report
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'src/run_l1.py'
CONFIG = ROOT / 'configs/exploratory_proxy.json'

#============
#Cmd
#============
def cmd(*args):
    subprocess.run([sys.executable, str(RUNNER), *map(str, args)], check=True)

#============
#Verification config
#============
def verification_config():
    config = json.loads(CONFIG.read_text())
    config['numerics']['checkpoint_every_steps'] = 4
    config['numerics']['save_every_steps'] = 12
    config['numerics']['run_days'] = 0.1
    dt = config['numerics']['time_step_s']
    config['forcing']['vortex_begin_day'] = 6 * dt / 86400
    config['forcing']['vortex_duration_days'] = 12 * dt / 86400
    config['config_id'] = 'restart_verification_only_short_trigger'
    return config

#============
#Check restart
#============
def check_restart(arm, root):
    config = verification_config()
    configfile = root / 'config.json'
    configfile.write_text(json.dumps(config))
    run_full = root / 'full'
    cmd('--arm', arm, '--config', configfile, '--steps', '24', '--out', run_full)
    from scipy.io import netcdf_file
    for split in (4, 12, 20):
        first = root / f'first_{split}'
        continued = root / f'continued_{split}'
        cmd('--arm', arm, '--config', configfile, '--steps', split, '--out', first)
        checkpoint = first / 'checkpoints' / f'step{split:08d}.npz'
        cmd('--arm', arm, '--config', configfile, '--steps', 24 - split, '--restart', checkpoint, '--out', continued)
        with netcdf_file(run_full / 'states/step00000024.nc', 'r', mmap=False) as ds1, netcdf_file(continued / 'states/step00000024.nc', 'r', mmap=False) as ds2:
            for name in ('u_east', 'v_north', 'layer_thickness'):
                a = np.array(ds1.variables[name].data)
                b = np.array(ds2.variables[name].data)
                if not np.allclose(a, b, rtol=1e-09, atol=1e-07):
                    raise AssertionError(f'{arm} split {split}: {name} max abs error {np.max(abs(a - b))}')
    write_report(root / 'restart_assessment.json', assess_restart(root))
    return f'{arm}: 24-step restart comparisons at 4, 12, 20 passed (numerical tolerance only)'

#============
#Command-line interface
#============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['A', 'B', 'C'], default='A')
    parser.add_argument('--output', default='runs/integration_tests')
    args = parser.parse_args()
    root = Path(args.output).resolve() / f'restart_{args.arm}'
    root.mkdir(parents=True, exist_ok=False)
    print(check_restart(args.arm, root))
if __name__ == '__main__':
    main()
