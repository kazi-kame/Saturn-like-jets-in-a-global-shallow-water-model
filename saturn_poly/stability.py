#============
#Imports and module setup
#============
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from numpy.polynomial import Polynomial
from scipy.linalg import eig, cholesky, solve_triangular
from .dynamics import Sphere
from .global_linear import build_operator
from .response import EnergyResponse

#============
#Construct balanced polynomial jets
#============
def polynomial_base(sphere, mean_depth, coefficients):
    c = np.asarray(coefficients, dtype=float)
    if c.ndim != 1 or not 1 <= len(c) <= 13 or (not np.all(np.isfinite(c))) or (not np.isfinite(mean_depth)) or (mean_depth <= 0):
        raise ValueError('Require 1–13 finite coefficients and positive mean depth')
    W, mu = (Polynomial(c), Polynomial([0, 1]))
    H = (-(2 * sphere.radius * sphere.rotation * mu * W + mu * W * W) / sphere.gravity).integ()
    integral = H.integ()
    H += mean_depth - (integral(1) - integral(-1)) / 2
    lower_bound = H.coef[0] - np.sum(np.abs(H.coef[1:]))
    if not np.isfinite(lower_bound) or lower_bound <= 0:
        raise ValueError('Conservative global positive-depth bound failed')

    #============
    #Base
    #============
    def base(phi):
        x, cosine = (np.sin(phi), np.cos(phi))
        return (cosine * W(x), -x * W(x) + cosine ** 2 * W.deriv()(x), H(x), cosine * H.deriv()(x))
    return (base, float(lower_bound))

#============
#Assemble the maintained-background linear operator
#============
def maintained_operator(sphere, mode, lmax, base, nu, nlat=None):
    if not np.isfinite(nu) or nu < 0:
        raise ValueError('Hyperdiffusion must be finite and nonnegative')
    op = build_operator(sphere, mode, lmax, base=base, nlat=nlat)
    rates = np.array([nu * ((degree * (degree + 1) - (kind != 'height')) / sphere.radius ** 2) ** 2 for kind, degree in op.labels])
    op.matrix = op.matrix - np.diag(rates)
    return op

#============
#Calculate the modal spectrum
#============
def spectrum(op, mode, archive=None, sphere=None, base=None):
    A = EnergyResponse(op.matrix, op.energy_matrix).matrix
    values, vectors = eig(A)
    scale = max(np.linalg.norm(A, 2), np.finfo(float).tiny)
    rows = []
    order = np.argsort(values.real)[::-1]
    if archive is not None:
        coefficients = solve_triangular(cholesky(op.energy_matrix, lower=False), vectors[:, order])
        from .stability_modes import hemisphere_shares, spectral_tail_norm
        tail_start, tails = spectral_tail_norm(coefficients, op.energy_matrix, op.labels)
        shares = hemisphere_shares(sphere, base, mode, op.labels[-1][1], coefficients, op.quadrature_points)
        np.savez_compressed(archive, eigenvalues=values[order], coefficients=coefficients, energy_matrix=op.energy_matrix, labels=np.array(op.labels, dtype=str))
    for rank, i in enumerate(order):
        s, v = (values[i], vectors[:, i])
        rows.append(dict(growth_s_inverse=float(s.real), frequency_rad_s=float(s.imag), pattern_speed_rad_s=float(-s.imag / mode) if mode else None, relative_residual=float(np.linalg.norm(A @ v - s * v) / ((scale + abs(s)) * np.linalg.norm(v)))))
        if archive is not None:
            rows[-1]['north_energy_fraction'] = float(shares[rank])
            rows[-1]['tail_start_degree'] = tail_start
            rows[-1]['tail_energy_norm_ratio'] = float(tails[rank])
    return rows

#============
#Validate model inputs
#============
def validate(config):
    keys = {'radius_m', 'rotation_rad_s', 'gravity_m_s2', 'mean_depth_m', 'velocity_polynomial_m_s', 'nu_m4_s', 'modes', 'truncations', 'quadrature_factors', 'base_interpretation'}
    smooth = 'smooth_jets' in config
    if smooth:
        keys = keys - {'velocity_polynomial_m_s'} | {'smooth_jets'}
    if set(config) != keys:
        raise ValueError('Missing or unknown configuration keys')
    if config['base_interpretation'] != 'maintained_smooth_proxy_not_observation':
        raise ValueError('Explicit maintained proxy interpretation required')
    for key in ('radius_m', 'rotation_rad_s', 'gravity_m_s2', 'mean_depth_m', 'nu_m4_s'):
        x = config[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or (not np.isfinite(x)):
            raise ValueError(f'Invalid {key}')
    if config['nu_m4_s'] < 0:
        raise ValueError('Negative hyperdiffusion')
    for key in ('modes', 'truncations', 'quadrature_factors'):
        v = config[key]
        if not isinstance(v, list) or not v or v != sorted(set(v)) or any((type(x) is not int for x in v)):
            raise ValueError(f'Invalid {key}')
    if not 0 <= min(config['modes']) <= max(config['modes']) <= min(config['truncations']):
        raise ValueError('Modes must fit every truncation')
    if len(config['truncations']) < 2 or not 1 <= min(config['truncations']) or max(config['truncations']) > 256 or (len(config['quadrature_factors']) < 2) or (min(config['quadrature_factors']) < 2) or (max(config['quadrature_factors']) * max(config['truncations']) + 4 > 2048):
        raise ValueError('Need two truncations <=256 and two quadratures within 2048 points')
    sphere = Sphere(config['radius_m'], config['rotation_rad_s'], config['gravity_m_s2'])
    if smooth:
        from .jet_reference import smooth_base
        base, bound = smooth_base(sphere, config['mean_depth_m'], config['smooth_jets'])
    else:
        base, bound = polynomial_base(sphere, config['mean_depth_m'], config['velocity_polynomial_m_s'])
    return (sphere, base, bound)

#============
#Execute the configured analysis
#============
def run(config_path, output, save_modes=False):
    raw = Path(config_path).read_bytes()
    config = json.loads(raw)
    sphere, base, bound = validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.lock.json').write_bytes(raw)
    root = Path(__file__).resolve().parents[1]
    files = list((root / 'saturn_poly').glob('*.py')) + list((root / 'tests').glob('*.py'))
    files += list(root.glob('requirements*.txt'))
    hashes = {}
    for path in files:
        data = path.read_bytes()
        rel = path.relative_to(root)
        target = output / 'inputs' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    report = dict(mode_archives_saved=save_modes, status='running', time_integration_performed=False, scientifically_validated=False, config_sha256=hashlib.sha256(raw).hexdigest(), input_sha256=hashes, environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__), positive_depth_lower_bound_m=bound, cases=[], convergence_assessed=False, interpretation='Maintained smooth proxy; no observed polygon or instability claim')

    #============
    #Save
    #============
    def save():
        tmp = output / 'report.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        tmp.replace(output / 'report.json')
    save()
    try:
        for mode in config['modes']:
            for lmax in config['truncations']:
                for factor in config['quadrature_factors']:
                    nlat = factor * lmax + 4
                    op = maintained_operator(sphere, mode, lmax, base, config['nu_m4_s'], nlat)
                    archive = output / f'modes_m{mode:02d}_l{lmax:03d}_q{nlat:04d}.npz'
                    if save_modes:
                        rows = spectrum(op, mode, archive, sphere, base)
                    else:
                        rows = spectrum(op, mode)
                    case = dict(mode=mode, lmax=lmax, nlat=nlat, eigenvalues=rows)
                    if save_modes:
                        case.update(mode_archive=archive.name, mode_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
                    report['cases'].append(case)
                    save()
        for rel, expected in hashes.items():
            if hashlib.sha256((root / rel).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Input changed during calculation: {rel}')
        report['status'] = 'completed'
        save()
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
        save()
        raise

#============
#Command-line interface
#============
def main():
    parser = argparse.ArgumentParser(description='Stability')
    parser.add_argument('--config', required=True)
    parser.add_argument('--out')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--save-modes', action='store_true', help='Save eigenvectors and hemisphere energy fractions')
    args = parser.parse_args()
    if args.validate_only:
        _, _, bound = validate(json.loads(Path(args.config).read_text()))
        print(json.dumps({'configuration': 'valid', 'positive_depth_lower_bound_m': bound}))
    elif not args.out:
        parser.error('--out required unless --validate-only')
    else:
        run(args.config, args.out, args.save_modes)
if __name__ == '__main__':
    main()
