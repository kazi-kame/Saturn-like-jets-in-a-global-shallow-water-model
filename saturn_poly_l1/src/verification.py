#============
#Imports and module setup
#============
import numpy as np

#============
#Stationary state
#============
def stationary_state(latitude, cfg):
    case = cfg['verification_case']
    p = cfg['physical_constants']
    n = cfg['numerics']
    speed = case['equatorial_speed_m_s']
    omega = 2 * np.pi / p['rotation_period_s']
    coefficient = (p['equatorial_radius_m'] * omega * speed + speed ** 2 / 2) / p['gravitational_acceleration_m_s2']
    u = speed * np.cos(latitude)
    h = n['mean_layer_depth_m'] - coefficient * (np.sin(latitude) ** 2 - 1 / 3)
    if not np.isfinite(h).all() or np.min(h) <= 0:
        raise ValueError('Stationary case has nonpositive/nonfinite depth')
    return (u, np.zeros_like(u), h)
