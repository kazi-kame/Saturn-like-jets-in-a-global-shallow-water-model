#============
#Imports and module setup
#============
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erf

#============
#Construct balanced smooth jets
#============
def smooth_base(sphere, mean_depth, jets):
    jets = np.asarray(jets, dtype=float)
    if jets.ndim != 2 or jets.shape[1] != 3 or len(jets) == 0 or (not np.all(np.isfinite(jets))) or np.any(jets[:, 2] <= 0) or np.any(np.abs(jets[:, 1]) >= 1) or (not math.isfinite(mean_depth)) or (mean_depth <= 0):
        raise ValueError('Invalid smooth jets or mean depth')
    terms = [(2 * sphere.radius * sphere.rotation * A, c, s) for A, c, s in jets]
    for A, c, s in jets:
        for B, d, t in jets:
            width = 1 / math.sqrt(1 / s ** 2 + 1 / t ** 2)
            center = width ** 2 * (c / s ** 2 + d / t ** 2)
            amplitude = A * B * math.exp(-(c - d) ** 2 / (2 * (s * s + t * t)))
            terms.append((amplitude, center, width))

    #============
    #Primitive
    #============
    def primitive(x):
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=float)
        for amplitude, c, s in terms:
            z = (x - c) / s
            out -= amplitude * (-s * s * np.exp(-z * z / 2) + c * s * math.sqrt(math.pi / 2) * erf(z / math.sqrt(2))) / sphere.gravity
        return out
    points = sorted(set((c for _, c, _ in terms)))
    mean, error = quad(lambda x: float(primitive(x)), -1, 1, points=points, epsabs=1e-08, epsrel=1e-12, limit=200)
    offset = mean_depth - mean / 2
    variation_bound = sum((abs(A) * math.sqrt(2 * math.pi) * s * (abs(c) + s * math.sqrt(2 / math.pi)) for A, c, s in terms)) / sphere.gravity
    lower = mean_depth - variation_bound - error / 2
    if sphere.rotation >= 0 and np.all(jets[:, 0] >= 0):
        lower = float(min(primitive(-1), primitive(1)) + offset - error / 2)
    if not math.isfinite(lower) or lower <= 0:
        raise ValueError('Conservative positive-depth bound failed')

    #============
    #Base
    #============
    def base(phi):
        x, co = (np.sin(phi), np.cos(phi))
        W, Wx = (np.zeros_like(x), np.zeros_like(x))
        for A, c, s in jets:
            term = A * np.exp(-0.5 * ((x - c) / s) ** 2)
            W += term
            Wx -= (x - c) / s ** 2 * term
        U = co * W
        Up = -x * W + co * co * Wx
        Hp = -co * (2 * sphere.radius * sphere.rotation * x * W + x * W * W) / sphere.gravity
        return (U, Up, primitive(x) + offset, Hp)
    return (base, float(lower))

#============
#Map jet parameters to smooth profiles
#============
def reconstruct(config):
    p = config['physical_constants']
    a, b = (p['equatorial_radius_m'], p['polar_radius_m'])

    #============
    #Convert
    #============
    def convert(deg):
        return np.arctan((b / a) ** 2 * np.tan(np.deg2rad(deg)))
    jets, targets = ([], [])
    for name in ('north', 'south'):
        j = config['jets'][name]
        latitude = j['latitude_planetographic_deg']
        half = j['fwhm_planetographic_deg'] / 2
        center = convert(latitude)
        sigma = abs(convert(latitude + half) - convert(latitude - half)) / 2.354820045
        amplitude = j['peak_eastward_m_s']
        targets.append((amplitude, center, sigma))
        jets.append([amplitude / np.cos(center), float(np.sin(center)), float(np.cos(center) * sigma)])

    #============
    #Original
    #============
    def original(phi):
        return sum((A * np.exp(-0.5 * ((phi - c) / s) ** 2) for A, c, s in targets))
    return (jets, original)

#============
#Prepare experiment inputs
#============
def prepare(source, output):
    import hashlib
    import json
    from pathlib import Path
    from .dynamics import Sphere
    from .stability import validate
    source, output = (Path(source), Path(output))
    raw = source.read_bytes()
    cfg = json.loads(raw)
    jets, original = reconstruct(cfg)
    p, n = (cfg['physical_constants'], cfg['numerics'])
    sphere = Sphere(p['equatorial_radius_m'], 2 * math.pi / p['rotation_period_s'], p['gravitational_acceleration_m_s2'])
    base, bound = smooth_base(sphere, n['mean_layer_depth_m'], jets)
    checks = []
    for count in (8193, 16385):
        phi = np.linspace(-math.pi / 2, math.pi / 2, count)
        u, up, h, hp = base(phi)
        old = original(phi)
        error = u - old
        rms = math.sqrt(float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(error ** 2 * np.cos(phi), phi) / 2))
        hemispheres = {}
        for name, mask in [('north', phi >= 0), ('south', phi < 0)]:
            indices = np.flatnonzero(mask)
            peak = indices[np.argmax(u[mask])]
            hemispheres[name] = dict(max_sampled_error_m_s=float(np.max(np.abs(error[mask]))), reconstructed_peak_m_s=float(u[peak]), peak_planetocentric_deg=float(np.rad2deg(phi[peak])))
        checks.append(dict(points=count, global_area_rms_error_m_s=rms, max_sampled_error_m_s=float(np.max(np.abs(error))), minimum_sampled_depth_m=float(np.min(h)), hemispheres=hemispheres))
    source_hash = hashlib.sha256(raw).hexdigest()
    provenance = dict(source_path=str(source.resolve()), source_sha256=source_hash, mapping='local latitude-width mapping; no least-squares or observational fit')
    config = dict(base_interpretation='maintained_smooth_proxy_not_observation', radius_m=sphere.radius, rotation_rad_s=sphere.rotation, gravity_m_s2=sphere.gravity, mean_depth_m=n['mean_layer_depth_m'], smooth_jets=jets, nu_m4_s=n['hyperviscosity_m4_s'], modes=list(range(0, 21)), truncations=[32, 40, 48], quadrature_factors=[3, 5])
    validate(config)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source_config.json').write_bytes(raw)
    (output / 'stability_config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    report = dict(source=provenance, positive_depth_lower_bound_m=bound, simulation_performed=False, eigenanalysis_performed=False, smooth_profile_equivalent_to_original=False, scientifically_validated=False)
    root = Path(__file__).resolve().parents[1]
    report['input_sha256'] = {}
    for relative in ('saturn_poly/jet_reference.py', 'saturn_poly/stability.py', 'saturn_poly/dynamics.py'):
        data = (root / relative).read_bytes()
        target = output / 'inputs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        report['input_sha256'][relative] = hashlib.sha256(data).hexdigest()
    (output / 'reconstruction.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savetxt(output / 'profile.csv', np.column_stack((np.rad2deg(phi), old, u, up, h, hp)), delimiter=',', header='latitude_planetocentric_deg,original_u_m_s,smooth_u_m_s,du_dphi_m_s,depth_m,dh_dphi_m', comments='')
    return report
if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Jet reference')
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out), indent=2))
