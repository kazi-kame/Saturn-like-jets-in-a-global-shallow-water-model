#============
#Imports and module setup
#============
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from .diagnostics import spectrum, summarize

#============
#Digest
#============
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

#============
#Command-line interface
#============
def main():
    parser = argparse.ArgumentParser(description='  main  ')
    sub = parser.add_subparsers(dest='command', required=True)
    analyze = sub.add_parser('analyze-contour', help='Analyze one full uniform contour CSV')
    analyze.add_argument('input', type=Path)
    analyze.add_argument('--max-mode', type=int, required=True)
    analyze.add_argument('--minimum-amplitude-m', type=float, required=True)
    analyze.add_argument('--minimum-share', type=float, required=True)
    reference = sub.add_parser('linear-reference', help='Static global rest-state eigenanalysis; not a Saturn simulation')
    reference.add_argument('--mode', type=int, required=True)
    reference.add_argument('--lmax', type=int, required=True)
    reference.add_argument('--radius-m', type=float, required=True)
    reference.add_argument('--gravity-m-s2', type=float, required=True)
    reference.add_argument('--depth-m', type=float, required=True)
    reference.add_argument('--rotation-rad-s', type=float, required=True)
    reference.add_argument('--quadrature-points', type=int)
    reference.add_argument('--gain-times-s', type=float, nargs='+', default=[])
    reference.add_argument('--forcing-frequencies-rad-s', type=float, nargs='+', default=[])
    study = sub.add_parser('reference-study', help='Explicit reproducible rest-state reference sweep; not Saturn calibration')
    study.add_argument('--config', type=Path, required=True)
    study.add_argument('--output', type=Path)
    study.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'reference-study':
            from .reference_study import validate_config, run_reference_study
            if args.check_only:
                result = validate_config(json.loads(args.config.read_text()))
            else:
                if args.output is None:
                    raise ValueError('--output is required unless --check-only is used')
                result = run_reference_study(args.config, args.output, Path(__file__).resolve().parent.parent)
        elif args.command == 'linear-reference':
            from .dynamics import Sphere
            from .global_linear import reference_report
            import numpy
            import scipy
            result = reference_report(Sphere(args.radius_m, args.rotation_rad_s, args.gravity_m_s2), args.mode, args.lmax, args.depth_m, args.quadrature_points, args.gain_times_s, args.forcing_frequencies_rad_s)
            result['dependencies'] = {'numpy': numpy.__version__, 'scipy': scipy.__version__}
        else:
            with args.input.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            coefficients = spectrum([float(r['longitude_rad']) for r in rows], [float(r['displacement_m']) for r in rows], args.max_mode)
            result = summarize(coefficients, args.minimum_amplitude_m, args.minimum_share)
            result.update({'input_sha256': digest(args.input), 'input': str(args.input), 'max_mode': args.max_mode, 'minimum_amplitude_m': args.minimum_amplitude_m, 'minimum_share': args.minimum_share, 'coefficients': {m: {'real_m': a.real, 'imag_m': a.imag} for m, a in coefficients.items()}})
        result.update({'python': platform.python_version(), 'code_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}})
        print(json.dumps(result, indent=2, allow_nan=False))
    except ImportError:
        parser.exit(2, 'Linear analysis needs NumPy/SciPy: install requirements.txt in a virtual environment.\n')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Error: {error}\n')
if __name__ == '__main__':
    main()
