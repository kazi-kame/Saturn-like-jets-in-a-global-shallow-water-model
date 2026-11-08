#============
#Imports and module setup
#============
from pathlib import Path
import argparse, csv, json
import numpy as np
from assess_runs import assess_restart, run_metadata, write_report

#============
#Read
#============
def read(run):
    p = Path(run)
    data = json.loads((p / 'summary.json').read_text())
    with (p / 'diagnostics.csv').open() as f:
        rows = list(csv.DictReader(f))
    return (data, rows)

#============
#Command-line interface
#============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--max-closed-mass-drift', type=float, default=0.0001)
    parser.add_argument('--restart-suite', help='Existing restart_<arm> directory; assessed without rerunning')
    parser.add_argument('--out', help='New report path; existing reports are preserved')
    args = parser.parse_args()
    if not np.isfinite(args.max_closed_mass_drift) or args.max_closed_mass_drift <= 0:
        parser.error('Mass tolerance must be finite and positive')
    summary, rows = read(args.run)
    manifest = run_metadata(args.run)
    restart = assess_restart(args.restart_suite) if args.restart_suite else None
    if restart is not None:
        for key in ('arm', 'code_sha256', 'packages'):
            if restart['compatibility'][key] != manifest['compatibility'][key]:
                raise ValueError(f'Restart evidence does not match control: {key}')
    mass = np.array([float(r['volume_m3']) for r in rows])
    energy = np.array([float(r['energy_per_density']) for r in rows])
    rel_mass = (mass[-1] - mass[0]) / mass[0]
    rel_energy = (energy[-1] - energy[0]) / energy[0]
    initial = [rows[0]['north_class'], rows[0]['south_class']]
    flags = {'finite_mass': bool(np.all(np.isfinite(mass))), 'finite_energy': bool(np.all(np.isfinite(energy))), 'initial_axisymmetric_ring': initial == ['none', 'none'], 'mass_drift_within_tolerance': bool(abs(rel_mass) < args.max_closed_mass_drift), 'positive_depth': all((float(r['min_h_m']) > 0 for r in rows))}
    result = {'run': args.run, 'arm': summary['arm'], 'relative_mass_change': rel_mass, 'relative_mechanical_energy_change': rel_energy, 'checks': flags, 'independent_benchmarks_passed': False, 'spatial_convergence_passed': False, 'timestep_convergence_passed': False, 'restart_equivalence_passed': restart['passed'] if restart else None, 'restart_evidence': restart, 'scientifically_validated': False, 'note': 'No scientific validation claim; mandatory independent benchmark and convergence suites remain.'}
    write_report(args.out or Path(args.run, 'preliminary_validation.json'), result)
    print(json.dumps(result, indent=2))
    if not all(flags.values()):
        raise SystemExit(1)
if __name__ == '__main__':
    main()
