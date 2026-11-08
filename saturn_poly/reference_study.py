#============
#Imports and module setup
#============
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from datetime import datetime, timezone
import numpy as np
import scipy
from .dynamics import Sphere
from .global_linear import reference_report
KEYS = {'purpose', 'radius_m', 'effective_gravity_m_s2', 'depth_m', 'rotation_rad_s', 'modes', 'truncations', 'quadrature_factors', 'relative_tolerance'}

#============
#Validate config
#============
def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Reference configuration has missing or unknown keys')
    if config['purpose'] != 'nonrotating_rest_numerical_reference_not_saturn':
        raise ValueError('This pipeline supports only the declared nonrotating rest reference')
    for key in ('radius_m', 'effective_gravity_m_s2', 'depth_m', 'relative_tolerance', 'rotation_rad_s'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
            raise ValueError(f'{key} must be a finite number')
        if key != 'rotation_rad_s' and value <= 0:
            raise ValueError(f'{key} must be positive')
    if config['rotation_rad_s'] != 0 or config['relative_tolerance'] >= 1:
        raise ValueError('Require zero rotation and relative tolerance below one')
    for key in ('modes', 'truncations', 'quadrature_factors'):
        values = config[key]
        if not isinstance(values, list) or not values or any((isinstance(x, bool) or not isinstance(x, int) for x in values)) or (values != sorted(set(values))):
            raise ValueError(f'{key} must be a nonempty increasing list of distinct integers')
    if len(config['truncations']) < 2 or not 1 <= min(config['truncations']) <= max(config['truncations']) <= 48:
        raise ValueError('Need at least two truncations between 1 and 48')
    if not 0 <= min(config['modes']) <= max(config['modes']) <= min(config['truncations']):
        raise ValueError('Modes must be nonnegative and fit every truncation')
    if len(config['quadrature_factors']) < 2 or min(config['quadrature_factors']) < 2:
        raise ValueError('Need at least two quadrature factors >=2')
    if max(config['quadrature_factors']) * max(config['truncations']) + 4 > 512:
        raise ValueError('Quadrature exceeds supported limit')
    count = len(config['modes']) * len(config['truncations']) * len(config['quadrature_factors'])
    return {'configuration': 'valid', 'case_count': count, 'scope': 'static analytic reference, not a Saturn scan or fluid simulation'}

#============
#Assess reference
#============
def assess_reference(report, config):
    mode, lmax = (report['mode'], report['lmax'])
    expected = []
    for degree in range(mode, lmax + 1):
        if degree:
            omega = math.sqrt(config['effective_gravity_m_s2'] * config['depth_m'] * degree * (degree + 1)) / config['radius_m']
            expected.extend((-omega, 0.0, omega))
        else:
            expected.append(0.0)
    actual = [complex(s['real'], s['imag']) for s in report['eigenvalues_s_inverse']]
    if len(actual) != len(expected) or not all((math.isfinite(abs(x)) for x in actual)):
        raise ValueError('Spectrum has wrong size or nonfinite entries')
    scale = max((abs(x) for x in expected))
    metrics = {'frequency_error_relative_to_fastest_wave': max((abs(a - b) for a, b in zip(sorted((x.imag for x in actual)), sorted(expected)))) / scale, 'real_part_error_relative_to_fastest_wave': max((abs(x.real) for x in actual)) / scale, 'eigenpair_relative_residual': report['max_eigenpair_relative_residual'], 'energy_skew_relative_residual': report['energy_skew_relative_residual']}
    if not all((math.isfinite(x) and x >= 0 for x in metrics.values())):
        raise ValueError('Invalid reference residual')
    return {'metrics': metrics, 'tolerance': config['relative_tolerance'], 'passed': all((x <= config['relative_tolerance'] for x in metrics.values())), 'frequency_scale_s_inverse': scale}

#============
#Write json
#============
def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

#============
#Compare numerical and analytical reference spectra
#============
def run_reference_study(config_path, output, root):
    config_path, output, root = (Path(config_path), Path(output), Path(root))
    raw = config_path.read_bytes()
    config = json.loads(raw)
    plan = validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'initializing', 'started_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': hashlib.sha256(raw).hexdigest(), 'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__}, 'case_count': plan['case_count'], 'input_sha256': {}, 'scientific_readiness': 'not_ready', 'time_integration_performed': False}
    _write_json(output / 'manifest.json', manifest)
    try:
        (output / 'config.lock.json').write_bytes(raw)
        paths = sorted((root / 'saturn_poly').glob('*.py'))
        paths += sorted((root / 'tests').glob('*.py'))
        paths += sorted(root.glob('requirements*.txt'))
        for path in paths:
            relative = path.relative_to(root)
            target = output / 'inputs' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            target.write_bytes(data)
            manifest['input_sha256'][str(relative)] = hashlib.sha256(data).hexdigest()
        manifest['status'] = 'running'
        _write_json(output / 'manifest.json', manifest)
        sphere = Sphere(config['radius_m'], 0, config['effective_gravity_m_s2'])
        cases = []
        for mode in config['modes']:
            for lmax in config['truncations']:
                for factor in config['quadrature_factors']:
                    start = time.perf_counter()
                    report = reference_report(sphere, mode, lmax, config['depth_m'], factor * lmax + 4)
                    report['wall_seconds'] = time.perf_counter() - start
                    report['assessment'] = assess_reference(report, config)
                    cases.append(report)
                    _write_json(output / 'results.json', {'cases': cases, 'complete': False})
        passed = all((c['assessment']['passed'] for c in cases))
        for relative, expected_hash in manifest['input_sha256'].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f'Input changed during study: {relative}')
        result = {'complete': True, 'all_reference_checks_passed': passed, 'cases': cases, 'interpretation': 'Agreement with exact rest-state spectra at each truncation only. Does not establish observed-jet convergence or reproduce a nonlinear benchmark.'}
        _write_json(output / 'results.json', result)
        manifest.update(status='completed', reference_checks_passed=passed, completed_utc=datetime.now(timezone.utc).isoformat())
        _write_json(output / 'manifest.json', manifest)
        return {'output': str(output), 'case_count': len(cases), 'reference_checks_passed': passed, 'scientific_readiness': 'not_ready'}
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        _write_json(output / 'manifest.json', manifest)
        raise
