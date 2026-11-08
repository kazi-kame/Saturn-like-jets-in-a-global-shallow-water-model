#============
#Imports and module setup
#============
import numpy as np
from scipy.special import roots_legendre
RADIUS = 6371220.0
OMEGA = 7.292e-05
GRAVITY = 9.80616
MEAN_DEPTH = 10000.0
PHI0 = np.pi / 7
PHI1 = np.pi / 2 - PHI0

#============
#Jet
#============
def jet(latitude):
    lat = np.asarray(latitude, dtype=float)
    u = np.zeros_like(lat)
    inside = (lat > PHI0) & (lat < PHI1)
    u[inside] = 80 * np.exp(1 / ((lat[inside] - PHI0) * (lat[inside] - PHI1)) + 4 / (PHI1 - PHI0) ** 2)
    return u

#============
#Balance gradient
#============
def balance_gradient(latitude):
    u = jet(latitude)
    return -(RADIUS * 2 * OMEGA * np.sin(latitude) * u + u * u * np.tan(latitude)) / GRAVITY

#============
#Balanced height
#============
def balanced_height(latitude, order=128):
    lat = np.asarray(latitude, dtype=float)
    nodes, weights = roots_legendre(order)
    upper = np.clip(lat, PHI0, PHI1)
    x = (upper[..., None] - PHI0) * (nodes + 1) / 2 + PHI0
    integral = (upper - PHI0) / 2 * np.sum(weights * balance_gradient(x), axis=-1)
    full = (PHI1 - PHI0) * (nodes + 1) / 2 + PHI0
    mean_integral = (PHI1 - PHI0) / 4 * np.sum(weights * balance_gradient(full) * (1 - np.sin(full)))
    return MEAN_DEPTH + integral - mean_integral

#============
#Perturbation
#============
def perturbation(longitude, latitude):
    wrapped = (np.asarray(longitude) + np.pi) % (2 * np.pi) - np.pi
    lat = np.asarray(latitude)
    return 120 * np.cos(lat) * np.exp(-(wrapped / (1 / 3)) ** 2 - ((np.pi / 4 - lat) / (1 / 15)) ** 2)

#============
#Initial state
#============
def initial_state(longitude, latitude):
    lon, lat = np.broadcast_arrays(longitude, latitude)
    unique, inverse = np.unique(lat, return_inverse=True)
    base = balanced_height(unique)[inverse].reshape(lat.shape)
    return (jet(lat), np.zeros_like(lat), base + perturbation(lon, lat))

#============
#Validate case
#============
def validate_case(cfg):
    p = cfg['physical_constants']
    n = cfg['numerics']
    for key, value in [('equatorial_radius_m', RADIUS), ('polar_radius_m', RADIUS), ('rotation_period_s', 2 * np.pi / OMEGA), ('gravitational_acceleration_m_s2', GRAVITY)]:
        if not np.isclose(p[key], value, rtol=1e-14, atol=0):
            raise ValueError(f'Galewsky source constant mismatch: {key}')
    if n['mean_layer_depth_m'] != MEAN_DEPTH or n['hyperviscosity_m4_s'] != 0:
        raise ValueError('Galewsky adjustment requires mean depth 10000 m and zero hyperdiffusion')
    if n['run_days'] > 0.25:
        raise ValueError('This benchmark is restricted to six-hour inviscid adjustment')

#============
#Plot adjustment
#============
def plot_adjustment(lat_deg, lon_deg, fields, outpath, time_s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lon = (np.asarray(lon_deg) + 180) % 360 - 180
    order = np.argsort(lon)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    for ax, name, label in zip(axes, ('height_minus_initial_balance', 'divergence'), ('Height minus initial balance [m]', 'Divergence [s^-1]')):
        a = fields[name][:, order]
        scale = max(float(np.max(abs(a))), 1e-15)
        mesh = ax.pcolormesh(lon[order], lat_deg, a, shading='auto', cmap='RdBu_r', vmin=-scale, vmax=scale)
        ax.set_ylabel('Latitude [deg]')
        fig.colorbar(mesh, ax=ax, label=label)
    axes[-1].set_xlabel('East longitude [deg]')
    fig.suptitle(f'Galewsky adjustment at {time_s / 3600:g} h')
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
