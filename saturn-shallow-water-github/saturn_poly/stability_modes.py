#============
#Imports and module setup
#============
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import linear_sum_assignment
from .global_linear import _basis

#============
#Spectral tail norm
#============
def spectral_tail_norm(coefficients, metric, labels):
    degrees = np.array([int(label[1]) for label in labels])
    low, high = (int(degrees.min()), int(degrees.max()))
    start = high - max(1, int(np.ceil(0.1 * (high - low + 1)))) + 1
    tail = np.array(coefficients, copy=True)
    tail[degrees < start] = 0
    total = np.real(np.sum(coefficients.conj() * (metric @ coefficients), axis=0))
    numerator = np.real(np.sum(tail.conj() * (metric @ tail), axis=0))
    if np.any(total <= 0) or not np.all(np.isfinite(total)) or (not np.all(np.isfinite(numerator))):
        raise ValueError('Invalid mode energy for spectral tail diagnostic')
    return (start, numerator / total)

#============
#Hemisphere shares
#============
def hemisphere_shares(sphere, base, mode, lmax, coefficients, nlat):
    x, weights = leggauss(nlat)
    energies = []
    for sign in (1, -1):
        mu = sign * (x + 1) / 2
        phi = np.arcsin(mu)
        B, _, _ = _basis(mode, lmax, mu)
        fields = np.einsum('cni,ij->cnj', B, coefficients, optimize=True)
        H = base(phi)[2]
        density = H[:, None] * (abs(fields[0]) ** 2 + abs(fields[1]) ** 2) + sphere.gravity * abs(fields[2]) ** 2
        energies.append(np.sum(weights[:, None] * density, axis=0) / 2)
    return energies[0] / (energies[0] + energies[1])

#============
#Match modes
#============
def match_modes(low_coefficients, high_coefficients, high_metric):
    low, high, metric = map(np.asarray, (low_coefficients, high_coefficients, high_metric))
    if low.shape[0] > high.shape[0] or metric.shape != (high.shape[0],) * 2:
        raise ValueError('Incompatible nested bases')
    padded = np.zeros((high.shape[0], low.shape[1]), dtype=complex)
    padded[:low.shape[0]] = low
    nl = np.real(np.sum(padded.conj() * (metric @ padded), axis=0))
    nh = np.real(np.sum(high.conj() * (metric @ high), axis=0))
    if np.any(nl <= 0) or np.any(nh <= 0):
        raise ValueError('Nonpositive mode energy')
    overlap = abs(padded.conj().T @ metric @ high) / np.sqrt(nl[:, None] * nh[None, :])
    rows, cols = linear_sum_assignment(-overlap)
    return [dict(low_index=int(i), high_index=int(j), overlap=float(overlap[i, j]), best_alternative_overlap=float(np.max(np.delete(overlap[i], j))) if overlap.shape[1] > 1 else 0.0) for i, j in zip(rows, cols)]

#============
#Compare saved
#============
def compare_saved(root):
    import hashlib
    import json
    from pathlib import Path
    from .stability_assessment import assess
    root = Path(root)
    assessment = assess(root)
    report = json.loads((root / 'report.json').read_text())
    cases = {}
    for c in report['cases']:
        if 'mode_archive' not in c:
            raise ValueError('Eigenvector archives required; old results remain valid spectra')
        path = root / c['mode_archive']
        if hashlib.sha256(path.read_bytes()).hexdigest() != c['mode_archive_sha256']:
            raise ValueError('Eigenvector archive checksum mismatch')
        cases[c['mode'], c['lmax'], c['nlat']] = c

    #============
    #Compare
    #============
    def compare(a, b, kind):
        with np.load(root / a['mode_archive'], allow_pickle=False) as lo, np.load(root / b['mode_archive'], allow_pickle=False) as hi:
            if not np.array_equal(lo['labels'], hi['labels'][:len(lo['labels'])]):
                raise ValueError('Basis labels are not nested')
            matches = match_modes(lo['coefficients'], hi['coefficients'], hi['energy_matrix'])
            for row in matches:
                i, j = (row['low_index'], row['high_index'])
                diff = hi['eigenvalues'][j] - lo['eigenvalues'][i]
                row.update(growth_difference_s_inverse=float(diff.real), frequency_difference_rad_s=float(diff.imag), high_north_energy_fraction=b['eigenvalues'][j]['north_energy_fraction'])
                row['high_tail_energy_norm_ratio'] = b['eigenvalues'][j].get('tail_energy_norm_ratio')
        return dict(kind=kind, mode=a['mode'], low_lmax=a['lmax'], high_lmax=b['lmax'], low_nlat=a['nlat'], high_nlat=b['nlat'], candidate_matches=matches)
    comparisons = []
    for m in sorted({k[0] for k in cases}):
        levels = sorted({k[1] for k in cases if k[0] == m})
        finest = []
        for ell in levels:
            qs = sorted((k[2] for k in cases if k[:2] == (m, ell)))
            comparisons.append(compare(cases[m, ell, qs[0]], cases[m, ell, qs[-1]], 'quadrature'))
            finest.append(cases[m, ell, qs[-1]])
        comparisons.extend((compare(a, b, 'truncation') for a, b in zip(finest, finest[1:])))
    assessment.update(candidate_branch_comparisons=comparisons, eigenbranches_matched=False, matching_note='Maximum total energy overlap assignment; no branch identity or convergence certification. Inspect overlaps, alternatives and eigenvalue changes.')
    return assessment
if __name__ == '__main__':
    import argparse
    import hashlib
    import json
    from pathlib import Path
    p = argparse.ArgumentParser(description='Stability modes')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = compare_saved(args.root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    result['assessment_code_sha256'] = {}
    for name in ('stability_modes.py', 'stability_assessment.py'):
        raw = Path(__file__).with_name(name).read_bytes()
        (output / name).write_bytes(raw)
        result['assessment_code_sha256'][name] = hashlib.sha256(raw).hexdigest()
    (output / 'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
