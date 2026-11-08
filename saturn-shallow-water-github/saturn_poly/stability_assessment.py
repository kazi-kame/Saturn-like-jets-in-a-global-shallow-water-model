#============
#Imports and module setup
#============
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

#============
#Assess
#============
def assess(folder):
    folder = Path(folder)
    raw = (folder / 'report.json').read_bytes()
    report = json.loads(raw)
    config_raw = (folder / 'config.lock.json').read_bytes()
    config = json.loads(config_raw)
    if report['status'] != 'completed':
        raise ValueError('Completed study required')

    #============
    #Verify
    #============
    def verify(path, expected):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {path}')
    verify(folder / 'config.lock.json', report['config_sha256'])
    for rel, expected in report['input_sha256'].items():
        verify(folder / 'inputs' / rel, expected)
    cases = {}
    for case in report['cases']:
        key = (case['mode'], case['lmax'], case['nlat'])
        if key in cases:
            raise ValueError('Duplicate spectrum')
        m, ell, _ = key
        expected_size = 3 * (ell - m + 1) - (2 if m == 0 else 0)
        eigenvalues = case['eigenvalues']
        if len(eigenvalues) != expected_size or not all((math.isfinite(e[field]) for e in eigenvalues for field in ('growth_s_inverse', 'frequency_rad_s', 'relative_residual'))):
            raise ValueError('Invalid spectrum')
        cases[key] = max(eigenvalues, key=lambda e: e['growth_s_inverse'])
    expected = {(m, l, f * l + 4) for m, l, f in itertools.product(config['modes'], config['truncations'], config['quadrature_factors'])}
    if set(cases) != expected:
        raise ValueError('Incomplete/unexpected case set')
    rows = []
    for ell in config['truncations']:
        low, high = (min(config['quadrature_factors']) * ell + 4, max(config['quadrature_factors']) * ell + 4)
        m = max(config['modes'], key=lambda m: cases[m, ell, high]['growth_s_inverse'])
        leading = cases[m, ell, high]
        rows.append(dict(lmax=ell, nlat=high, leading_mode=m, **leading, leading_growth_quadrature_difference_s_inverse=abs(leading['growth_s_inverse'] - cases[m, ell, low]['growth_s_inverse'])))
    changes = []
    high_factor = max(config['quadrature_factors'])
    lo, hi = config['truncations'][-2:]
    for m in config['modes']:
        a, b = (cases[m, lo, high_factor * lo + 4], cases[m, hi, high_factor * hi + 4])
        changes.append(dict(mode=m, lower_truncation_growth_s_inverse=a['growth_s_inverse'], higher_truncation_growth_s_inverse=b['growth_s_inverse'], growth_difference_s_inverse=b['growth_s_inverse'] - a['growth_s_inverse']))
    return dict(kind='saved_spectral_abscissa_comparison', source_report_sha256=hashlib.sha256(raw).hexdigest(), cases_verified=len(cases), leading_by_truncation=rows, final_truncation_changes=changes, maximum_eigenpair_residual=max((e['relative_residual'] for c in report['cases'] for e in c['eigenvalues'])), leading_mode_changes=len({r['leading_mode'] for r in rows}) > 1, convergence_established=False, eigenbranches_matched=False, scientifically_validated=False, interpretation='Compares largest real parts, not tracked eigenbranches. No preferred physical mode established.')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Stability assessment')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = assess(args.root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (output / 'assessment.py').write_bytes(source)
    result['assessment_code_sha256'] = hashlib.sha256(source).hexdigest()
    (output / 'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['leading_by_truncation'], indent=2))
