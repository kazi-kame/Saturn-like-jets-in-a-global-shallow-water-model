#============
#Imports and module setup
#============
from __future__ import annotations
import numpy as np

#============
#Planetographic to planetocentric
#============
def planetographic_to_planetocentric(degrees, a, b):
    phi = np.deg2rad(np.asarray(degrees, dtype=float))
    ans = np.arctan2((b / a) ** 2 * np.sin(phi), np.cos(phi))
    return np.rad2deg(ans)

#============
#Ring modes
#============
def ring_modes(anomaly, max_mode=20, min_amplitude_m=0.01, min_peak_ratio=2.0):
    y = np.asarray(anomaly, float).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError('Nonfinite or masked ring: separate coverage-aware estimator needed')
    if y.size < 2 * max_mode + 1:
        raise ValueError('Too few longitude points for max_mode')
    y = y - np.mean(y)
    harmonics = np.fft.rfft(y)
    amp = 2.0 / y.size * np.abs(harmonics)
    amp[0] = 0.0
    amp = amp[1:max_mode + 1]
    ranks = np.argsort(amp)[::-1]
    primary = int(ranks[0] + 1)
    top = float(amp[ranks[0]])
    second = float(amp[ranks[1]]) if len(ranks) > 1 else 0.0
    ratio = float(top / max(second, 1e-300))
    cls = 'none' if top < min_amplitude_m else 'mixed' if ratio < min_peak_ratio else 'dominant_mode'
    return {'classification': cls, 'm': primary if cls == 'dominant_mode' else None, 'strongest_m': primary, 'amplitude_m': top, 'peak_to_second': ratio, 'amplitudes_m': [float(x) for x in amp], 'phase_rad': float(np.angle(harmonics[primary]))}

#============
#Integrate mass and mechanical energy
#============
def global_mass_energy(u, v, h, lat, lon, radius_m, gravity):
    lat = np.asarray(lat).reshape(-1)
    lon = np.asarray(lon).reshape(-1)
    if lat.size != h.shape[0] or lon.size != h.shape[1]:
        raise ValueError('grid and state shape mismatch')
    dlon = 2 * np.pi / len(lon)
    edge = np.empty(lat.size + 1)
    edge[1:-1] = (lat[:-1] + lat[1:]) * 0.5
    edge[0], edge[-1] = (-np.pi / 2, np.pi / 2)
    if np.any(np.diff(edge) <= 0):
        raise ValueError('latitude must be ascending')
    weights = np.diff(np.sin(edge))[:, None] * dlon * radius_m ** 2
    mass = float(np.sum(h * weights))
    energy = float(np.sum((0.5 * h * (u * u + v * v) + 0.5 * gravity * h * h) * weights))
    return {'layer_volume_m3': mass, 'layer_mechanical_energy_J_per_density': energy, 'min_h_m': float(h.min()), 'max_speed_m_s': float(np.hypot(u, v).max())}

#============
#Verify config
#============
def verify_config(cfg):

    #============
    #Number
    #============
    def number(section, key, *, positive=False, nonnegative=False, integer=False):
        value = section[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not np.isfinite(value)) or (integer and (not isinstance(value, int))):
            raise ValueError(f'{key} must be a finite ' + ('integer' if integer else 'number'))
        if positive and value <= 0 or (nonnegative and value < 0):
            raise ValueError(f'{key} is outside its allowed range')
        return value
    try:
        p = cfg['physical_constants']
        n = cfg['numerics']
        f = cfg['forcing']
        d = cfg['diagnostics']
        for key in ('equatorial_radius_m', 'polar_radius_m', 'rotation_period_s', 'gravitational_acceleration_m_s2'):
            number(p, key, positive=True)
        galewsky = cfg.get('verification_case', {}).get('kind') == 'galewsky_adjustment'
        if p['polar_radius_m'] >= p['equatorial_radius_m'] and (not galewsky):
            raise ValueError('must be oblate')
        for key in ('nlon', 'ncolat', 'save_every_steps', 'checkpoint_every_steps', 'max_mode'):
            number(n, key, positive=True, integer=True)
        if n['nlon'] % 2 or n['ncolat'] % 2 or min(n['nlon'], n['ncolat']) < 4:
            raise ValueError('grid sizes must be even and at least 4')
        for key in ('mean_layer_depth_m', 'time_step_s', 'run_days', 'dealias'):
            number(n, key, positive=True)
        if n['dealias'] < 1.5:
            raise ValueError('quadratic nonlinear solver requires dealias >=1.5')
        if any((not np.isclose(n['dealias'] * n[key], round(n['dealias'] * n[key])) for key in ('nlon', 'ncolat'))):
            raise ValueError('dealiased grid sizes must be integers')
        number(n, 'hyperviscosity_m4_s', nonnegative=True)
        if 'verification_case' in cfg:
            case = cfg['verification_case']
            if galewsky:
                from galewsky import validate_case
                validate_case(cfg)
            elif case['kind'] == 'stationary_solid_rotation':
                number(case, 'equatorial_speed_m_s')
                if n['hyperviscosity_m4_s'] != 0:
                    raise ValueError('Exact stationary case requires zero diffusion')
            else:
                raise ValueError('Unknown verification case')
        if round(n['run_days'] * 86400 / n['time_step_s']) < 1:
            raise ValueError('configured run duration is shorter than one timestep')
        for key in ('acceleration_scale_m_s2', 'vortex_acceleration_scale_m_s2'):
            number(f, key, nonnegative=True)
        for key in ('correlation_time_s', 'forcing_fwhm_planetographic_deg', 'vortex_sigma_deg', 'vortex_begin_day', 'vortex_duration_days'):
            number(f, key, positive=True)
        number(f, 'rng_seed', nonnegative=True, integer=True)
        for key in ('mode_min', 'mode_max'):
            number(f, key, positive=True, integer=True)
        spectral_limit = min(n['nlon'] // 2 - 1, n['ncolat'] - 1)
        if not f['mode_min'] <= f['mode_max'] <= spectral_limit or n['max_mode'] > spectral_limit:
            raise ValueError('forcing/diagnostic modes must fit the retained spherical basis')
        center = number(f, 'vortex_center_planetographic_deg')
        number(f, 'vortex_center_lon_deg')
        if not 0 < abs(center) < 90:
            raise ValueError('anticyclone center must be away from equator and poles')
        if f['vortex_sigma_deg'] >= 90:
            raise ValueError('vortex width must be below 90 degrees')
        for pole in ('north', 'south'):
            jet = cfg['jets'][pole]
            latitude = number(jet, 'latitude_planetographic_deg')
            number(jet, 'peak_eastward_m_s')
            width = number(jet, 'fwhm_planetographic_deg', positive=True)
            if not 0 < abs(latitude) < 90 or (latitude > 0) != (pole == 'north'):
                raise ValueError('jet latitude must match its named hemisphere')
            for span in (width, f['forcing_fwhm_planetographic_deg']):
                if abs(latitude) + span / 2 >= 90:
                    raise ValueError('jet/forcing FWHM endpoints cross a pole')
        number(d, 'mode_detection_min_height_amplitude_m', positive=True)
        if number(d, 'min_peak_to_second_ratio') <= 1:
            raise ValueError('peak ratio must exceed one')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Missing or malformed configuration: {exc}') from exc
    return True

#============
#Gradient wind balanced height
#============
def gradient_wind_balanced_height(lat_radians, east_wind_m_s, radius_m, omega_s, g_m_s2, mean_depth_m):
    from scipy.integrate import cumulative_trapezoid, trapezoid
    lat = np.asarray(lat_radians, dtype=float)
    u = np.asarray(east_wind_m_s, dtype=float)
    if lat.ndim != 1 or u.shape != lat.shape or (not np.all(np.diff(lat) > 0)):
        raise ValueError('need 1D increasing latitude and matched u')
    f = 2 * omega_s * np.sin(lat)
    grad = -(radius_m * f * u + u * u * np.tan(lat)) / g_m_s2
    h = cumulative_trapezoid(grad, lat, initial=0)
    weights = np.cos(lat)
    offset = trapezoid(h * weights, lat) / trapezoid(weights, lat)
    return mean_depth_m + h - offset
