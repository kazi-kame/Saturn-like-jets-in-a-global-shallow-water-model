#============
#Imports and module setup
#============
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from diagnostics import global_mass_energy, planetographic_to_planetocentric, ring_modes, verify_config
from grid_io import grid_copy, set_grid, equation_namespace, IVP_EQUATIONS, SI_SOLVER_OPTIONS
from verification import stationary_state
from provenance import collect_provenance, load_checkpoint, RunRecord, atomic_json, package_versions, DEDALUS_VERSION

#============
#Read config
#============
def read_config(path):
    path = Path(path)
    data = json.loads(path.read_text())
    verify_config(data)
    return (data, hashlib.sha256(path.read_bytes()).hexdigest())

#============
#Jet profile
#============
def jet_profile(lat_radians, cfg):
    p = cfg['physical_constants']
    jets = cfg['jets']
    a = p['equatorial_radius_m']
    b = p['polar_radius_m']
    result = np.zeros_like(lat_radians, dtype=float)
    for pole in ['north', 'south']:
        j = jets[pole]
        mu = float(planetographic_to_planetocentric(j['latitude_planetographic_deg'], a, b))
        half = j['fwhm_planetographic_deg'] / 2
        plus = float(planetographic_to_planetocentric(j['latitude_planetographic_deg'] + half, a, b))
        minus = float(planetographic_to_planetocentric(j['latitude_planetographic_deg'] - half, a, b))
        sigma = np.deg2rad(abs(plus - minus)) / 2.354820045
        result += j['peak_eastward_m_s'] * np.exp(-0.5 * ((lat_radians - np.deg2rad(mu)) / sigma) ** 2)
    return result

#============
#Forcing
#============
class Forcing:

    #============
    #Initialize object state
    #============
    def __init__(self, cfg, lon, lat, arm):
        self.cfg = cfg
        self.arm = arm
        f = cfg['forcing']
        p = cfg['physical_constants']
        self.rng = np.random.default_rng(int(f['rng_seed']))
        self.modes = np.arange(int(f['mode_min']), int(f['mode_max']) + 1)
        self.coeff = np.zeros((len(self.modes), 2), float)
        self.lon = lon
        self.lat = lat
        self.coslat = np.maximum(np.cos(lat), 0.001)
        self.acc = float(f['acceleration_scale_m_s2'])
        self.corr = float(f['correlation_time_s'])
        self.profiles = []
        for pole in ('north', 'south'):
            c = cfg['jets'][pole]['latitude_planetographic_deg']
            w = f['forcing_fwhm_planetographic_deg']
            mu = np.deg2rad(float(planetographic_to_planetocentric(c, p['equatorial_radius_m'], p['polar_radius_m'])))
            left = np.deg2rad(float(planetographic_to_planetocentric(c - w / 2, p['equatorial_radius_m'], p['polar_radius_m'])))
            right = np.deg2rad(float(planetographic_to_planetocentric(c + w / 2, p['equatorial_radius_m'], p['polar_radius_m'])))
            sigma = abs(left - right) / 2.354820045
            g = np.exp(-0.5 * ((lat - mu) / sigma) ** 2)
            dg = -(lat - mu) / (sigma * sigma) * g
            self.profiles.append((g, dg))

    #============
    #Calculate
    #============
    def calculate(self, step, dt):
        lon = self.lon
        lat = self.lat
        fu = np.zeros(np.broadcast_shapes(lon.shape, lat.shape), dtype=float)
        fn = np.zeros_like(fu)
        f = self.cfg['forcing']
        if self.arm in ('B', 'C'):
            decay = np.exp(-dt / self.corr)
            self.coeff = decay * self.coeff + np.sqrt(-np.expm1(-2 * dt / self.corr)) * self.rng.standard_normal(self.coeff.shape)
            phase = self.modes[:, None, None] * lon[None, :, :]
            a = self.coeff[:, 0, None, None]
            b = self.coeff[:, 1, None, None]
            az = (a * np.cos(phase) + b * np.sin(phase)).sum(axis=0) / np.sqrt(len(self.modes))
            daz = (self.modes[:, None, None] * (-a * np.sin(phase) + b * np.cos(phase))).sum(axis=0) / np.sqrt(len(self.modes))
            for profile, dprofile in self.profiles:
                fu -= self.acc * dprofile * az
                fn += self.acc * profile * daz / self.coslat
        if self.arm == 'C':
            t_days = (step + 0.5) * dt / 86400
            if f['vortex_begin_day'] <= t_days < f['vortex_begin_day'] + f['vortex_duration_days']:
                a0 = self.cfg['physical_constants']['equatorial_radius_m']
                b0 = self.cfg['physical_constants']['polar_radius_m']
                center = np.deg2rad(float(planetographic_to_planetocentric(f['vortex_center_planetographic_deg'], a0, b0)))
                alpha = np.deg2rad(f['vortex_sigma_deg'])
                lo0 = np.deg2rad(f['vortex_center_lon_deg'])
                dlon = (lon - lo0 + np.pi) % (2 * np.pi) - np.pi
                dlat = lat - center
                cc = np.cos(center)
                gauss = np.exp(-0.5 * (dlat * dlat + (cc * dlon) ** 2) / alpha ** 2)
                amag = float(f['vortex_acceleration_scale_m_s2']) * np.sign(center)
                fu += amag * dlat / alpha ** 2 * gauss
                fn -= amag * (cc ** 2 * dlon) / (alpha ** 2 * self.coslat) * gauss
        return (fu, fn)

#============
#Get pole records
#============
def get_pole_records(h, lat_deg, lon_deg, cfg):
    p = cfg['physical_constants']
    anomaly = h - h.mean(axis=1, keepdims=True)
    out = {}
    for pole in ('north', 'south'):
        j = cfg['jets'][pole]
        lat0 = float(planetographic_to_planetocentric(j['latitude_planetographic_deg'], p['equatorial_radius_m'], p['polar_radius_m']))
        i = int(np.argmin(np.abs(lat_deg - lat0)))
        m = ring_modes(anomaly[i], max_mode=cfg['numerics']['max_mode'], min_amplitude_m=cfg['diagnostics']['mode_detection_min_height_amplitude_m'], min_peak_ratio=cfg['diagnostics']['min_peak_to_second_ratio'])
        out[pole] = {'sampled_lat_planetocentric_deg': float(lat_deg[i]), **m}
    return out

#============
#Write png maps
#============
def write_png_maps(lat_deg, lon_deg, h, config, output, step, time_s, modes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    anomaly = h - h.mean(axis=1, keepdims=True)
    theta = np.deg2rad(lon_deg)
    for pole in ('north', 'south'):
        selection = lat_deg >= 40 if pole == 'north' else lat_deg <= -40
        subset = anomaly[selection, :]
        radii = 90 - np.abs(lat_deg[selection])
        order = np.argsort(radii)
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        mesh = ax.pcolormesh(theta, radii[order], subset[order], shading='auto', cmap='RdBu_r')
        ax.set_ylim(0, 50)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1 if pole == 'north' else 1)
        ax.set_title(f'{pole.title()} polar height anomaly at {time_s / 3600:g} h', fontsize=10)
        fig.colorbar(mesh, ax=ax, label='Layer height minus zonal mean [m]', pad=0.12)
        fig.savefig(output / f'{pole}_step{step:08d}.png', dpi=130, bbox_inches='tight')
        plt.close(fig)

#============
#Save physical fields to NetCDF
#============
def save_state_nc(outpath, lat_deg, lon_deg, u, v, h, step, time_s, cfg, auxiliary=None):
    import xarray as xr
    ds = xr.Dataset({'u_east': (('latitude_planetocentric', 'longitude_east'), u), 'v_north': (('latitude_planetocentric', 'longitude_east'), v), 'layer_thickness': (('latitude_planetocentric', 'longitude_east'), h)}, coords={'latitude_planetocentric': lat_deg, 'longitude_east': lon_deg}, attrs={'model_tier': 'L1', 'planet': 'Saturn spherical surrogate', 'step': int(step), 'sim_time_s': float(time_s), 'profile_provenance': 'analytic Gaussian proxies, NOT digitized full wind profiles', 'gravity_m_s2': cfg['physical_constants']['gravitational_acceleration_m_s2']})
    for k in ds.data_vars:
        ds[k].attrs['units'] = 'm s-1' if k != 'layer_thickness' else 'm'
    case = cfg.get('verification_case', {}).get('kind')
    ds.attrs['experiment_kind'] = case or 'saturn_proxy'
    if case:
        ds.attrs['profile_provenance'] = case
    if case == 'galewsky_adjustment':
        ds.attrs['planet'] = 'Earth benchmark sphere'
    for name, values in (auxiliary or {}).items():
        ds[name] = (('latitude_planetocentric', 'longitude_east'), values)
        ds[name].attrs['units'] = 'm' if name == 'height_minus_initial_balance' else 's^-1'
    ds.longitude_east.attrs['units'] = 'degrees_east'
    ds.latitude_planetocentric.attrs['units'] = 'degrees_north'
    ds.to_netcdf(outpath, engine='scipy')

#============
#Check solver dependencies
#============
def check_environment():
    import importlib
    versions = package_versions()
    errors = []
    for name in ('dedalus.public', 'numpy', 'scipy', 'xarray', 'matplotlib', 'mpi4py.MPI', 'h5py'):
        try:
            importlib.import_module(name)
        except (ImportError, RuntimeError, OSError) as exc:
            errors.append(f'{name}: {exc}')
    if versions['dedalus'] != DEDALUS_VERSION:
        errors.append(f"Dedalus {DEDALUS_VERSION} required; found {versions['dedalus']}")
    return {'packages': versions, 'errors': errors, 'ready_to_attempt_build': not errors}

#============
#Build and evolve the shallow-water system
#============
def evolve(args, cfg, record, restart, d3):
    n = cfg['numerics']
    p = cfg['physical_constants']
    radius = p['equatorial_radius_m']
    g = p['gravitational_acceleration_m_s2']
    omega = 2 * np.pi / p['rotation_period_s']
    nu = n['hyperviscosity_m4_s']
    timestep = float(n['time_step_s'])
    H = float(n['mean_layer_depth_m'])
    is_galewsky = cfg.get('verification_case', {}).get('kind') == 'galewsky_adjustment'
    output = record.output
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=np.float64)
    if dist.comm.size != 1:
        raise RuntimeError('This runner requires exactly one MPI process')
    basis = d3.SphereBasis(coords, (n['nlon'], n['ncolat']), radius=radius, dealias=n['dealias'], dtype=np.float64)
    u = dist.VectorField(coords, name='u', bases=basis)
    h = dist.Field(name='h', bases=basis)
    F = dist.VectorField(coords, name='F', bases=basis)
    lon_grid, theta_grid = dist.local_grids(basis, scales=1)
    lat_grid = np.pi / 2 - theta_grid + 0 * lon_grid
    lon_grid = lon_grid + 0 * lat_grid
    lat = lat_grid[0, :]
    lon = lon_grid[:, 0]
    azidx = np.argsort(lat)
    lat_deg = np.rad2deg(lat[azidx])
    lon_deg = np.rad2deg(lon)
    force = Forcing(cfg, lon_grid, lat_grid, args.arm)
    zcross = lambda vector: d3.MulCosine(d3.skew(vector))
    step0 = 0
    t0 = 0.0
    if restart is not None:
        set_grid(u, restart['u'])
        set_grid(h, restart['h'])
        force.coeff = restart['forcing_coeff']
        force.rng.bit_generator.state = restart['rng_state']
        step0 = restart['step']
        t0 = restart['time_s']
    elif is_galewsky:
        from galewsky import initial_state
        east, north, depth = initial_state(lon_grid, lat_grid)
        set_grid(u, np.stack((east, -north)))
        set_grid(h, depth)
    elif 'verification_case' in cfg:
        east, north, depth = stationary_state(lat_grid, cfg)
        set_grid(u, np.stack((east, -north)))
        set_grid(h, depth)
    else:
        set_grid(u, np.stack((jet_profile(lat_grid, cfg), np.zeros_like(lat_grid))))
        c = dist.Field(name='c')
        lbvp = d3.LBVP([h, c], namespace=dict(h=h, c=c, g=g, u=u, omega=omega, zcross=zcross))
        lbvp.add_equation('g*lap(h) + c = -div(u@grad(u) + 2*omega*zcross(u))')
        lbvp.add_equation('ave(h) = 0')
        lbvp.build_solver(**SI_SOLVER_OPTIONS).solve()
        set_grid(h, grid_copy(h) + H)
    set_grid(F, np.zeros_like(grid_copy(u)))
    problem = d3.IVP([u, h], namespace=equation_namespace(u, h, F, g, omega, nu, H, zcross))
    for equation in IVP_EQUATIONS:
        problem.add_equation(equation)
    solver = problem.build_solver(d3.RK222, **SI_SOLVER_OPTIONS)
    solver.sim_time = t0

    #============
    #Checked fields
    #============
    def checked_fields():
        arr = grid_copy(u)
        depth = grid_copy(h)
        if not np.isfinite(arr).all() or not np.isfinite(depth).all() or depth.min() <= 0:
            raise FloatingPointError('Nonfinite velocity/depth or nonpositive layer depth')
        return (arr, depth)
    checked_fields()

    #============
    #Auxiliary fields
    #============
    def auxiliary_fields(depth):
        if not is_galewsky:
            return {}
        from galewsky import balanced_height
        return {'divergence': grid_copy(d3.div(u).evaluate()).T[azidx], 'relative_vorticity': -grid_copy(d3.div(d3.skew(u)).evaluate()).T[azidx], 'height_minus_initial_balance': depth - balanced_height(np.deg2rad(lat_deg))[:, None]}
    if args.build_only:
        arr, depth = checked_fields()
        save_state_nc(output / 'states/step00000000.nc', lat_deg, lon_deg, arr[0].T[azidx], -arr[1].T[azidx], depth.T[azidx], 0, 0, cfg, auxiliary_fields(depth.T[azidx]))
        record.update(status='built_not_evolved', steps_evolved=0)
        print('Solver constructed; initial fields finite and positive; no timesteps taken.')
        return 0
    steps = args.steps if args.steps is not None else 10000 if args.benchmark else int(round(n['run_days'] * 86400 / timestep))
    record.update(status='running', start_step=step0, requested_steps=steps, timing_scope='time loop including periodic output; excludes build and final output')
    with (output / 'diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'time_s', 'volume_m3', 'energy_per_density', 'min_h_m', 'max_speed_m_s', 'north_m', 'south_m', 'north_class', 'south_class'])
        writer.writeheader()

        #============
        #Analyze
        #============
        def analyze(step, write_output=True):
            arr, depth = checked_fields()
            u2 = arr[0].T[azidx]
            v2 = -arr[1].T[azidx]
            h2 = depth.T[azidx]
            budgets = global_mass_energy(u2, v2, h2, np.deg2rad(lat_deg), np.deg2rad(lon_deg), radius, g)
            modes = {pole: {'m': None, 'classification': 'not_applicable_benchmark'} for pole in ('north', 'south')} if is_galewsky else get_pole_records(h2, lat_deg, lon_deg, cfg)
            writer.writerow({'step': step, 'time_s': step * timestep, 'volume_m3': budgets['layer_volume_m3'], 'energy_per_density': budgets['layer_mechanical_energy_J_per_density'], 'min_h_m': budgets['min_h_m'], 'max_speed_m_s': budgets['max_speed_m_s'], 'north_m': modes['north']['m'], 'south_m': modes['south']['m'], 'north_class': modes['north']['classification'], 'south_class': modes['south']['classification']})
            stream.flush()
            if write_output and (not args.benchmark):
                auxiliary = auxiliary_fields(h2)
                if is_galewsky:
                    from galewsky import plot_adjustment
                    plot_adjustment(lat_deg, lon_deg, auxiliary, output / 'figures' / f'adjustment_step{step:08d}.png', step * timestep)
                else:
                    write_png_maps(lat_deg, lon_deg, h2, cfg, output / 'figures', step, step * timestep, modes)
                save_state_nc(output / 'states' / f'step{step:08d}.nc', lat_deg, lon_deg, u2, v2, h2, step, step * timestep, cfg, auxiliary)
            return (modes, budgets)
        modes, budgets = analyze(step0)
        last_saved = step0
        last_checkpoint = None
        tstart = time.perf_counter()
        for idx in range(steps):
            step = step0 + idx
            fu, fn = force.calculate(step, timestep)
            set_grid(F, np.stack((fu, -fn)))
            solver.step(timestep)
            arr, depth = checked_fields()
            if not args.benchmark and (step + 1) % n['save_every_steps'] == 0:
                modes, budgets = analyze(step + 1)
                last_saved = step + 1
            if not args.benchmark and (step + 1) % n['checkpoint_every_steps'] == 0:
                record.checkpoint(arr, depth, force, step + 1, (step + 1) * timestep)
                last_checkpoint = step + 1
        elapsed = time.perf_counter() - tstart
        final_step = step0 + steps
        if last_saved != final_step:
            modes, budgets = analyze(final_step)
        if not args.benchmark and last_checkpoint != final_step:
            arr, depth = checked_fields()
            record.checkpoint(arr, depth, force, final_step, final_step * timestep)
    summary = {'status': 'numerical_run_only_not_scientifically_validated', 'arm': args.arm, 'peak_rss_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), 'physical_days_per_wall_day': steps * timestep / elapsed, 'physical_seconds_evolved': steps * timestep, 'wall_seconds': elapsed, 'timing_scope': record.manifest['timing_scope'], 'measured_steps_per_wall_s': steps / elapsed, 'steps': steps, 'time_step_s': timestep, 'start_step': step0, 'final_step': final_step, 'experiment_kind': cfg.get('verification_case', {}).get('kind', 'saturn_proxy'), 'north': modes['north'], 'south': modes['south'], 'budgets_final': budgets, 'known_limitations': ['External reference-field agreement unverified', 'timestep/spatial convergence unverified', 'six-hour inviscid adjustment only'] if is_galewsky else ['Gaussian profile proxies', 'effective one-layer SW', 'unverified polar regularity', 'unvalidated stochastic forcing', 'approximate budget quadrature', 'independent benchmarks pending']}
    atomic_json(output / 'summary.json', summary)
    if args.benchmark:
        atomic_json(output / 'benchmark.json', summary)
    record.update(status='completed', final_step=final_step, steps_evolved=steps)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('north', 'south')}, indent=2))
    return 0

#============
#Command-line interface
#============
def main(argv=None):
    ap = argparse.ArgumentParser(description='Run l1')
    ap.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/saturn_working_1024x512.json'))
    ap.add_argument('--arm', choices=['A', 'B', 'C'], default='A')
    ap.add_argument('--out', default='runs/smoke', help='New directory; existing paths are refused')
    ap.add_argument('--steps', type=int, default=None, help='Additional steps, including on restart')
    ap.add_argument('--benchmark', action='store_true', help='10,000 steps by default; no intermediate output')
    ap.add_argument('--restart', help='Registered checkpoint; continuation uses a NEW output directory')
    ap.add_argument('--validate-config', action='store_true')
    ap.add_argument('--check-environment', action='store_true')
    ap.add_argument('--build-only', action='store_true', help='Construct balanced fields and solver without time evolution')
    args = ap.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        ap.error('--steps must be positive')
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    verify_config(cfg)
    if 'verification_case' in cfg and args.arm != 'A':
        ap.error('Verification cases require unforced arm A')
    if cfg.get('verification_case', {}).get('kind') == 'galewsky_adjustment':
        if args.benchmark:
            ap.error('Galewsky requires saved snapshots; --benchmark is unsupported')
        steps = args.steps if args.steps is not None else round(cfg['numerics']['run_days'] * 86400 / cfg['numerics']['time_step_s'])
        if not args.build_only and steps * cfg['numerics']['time_step_s'] > 21600:
            ap.error('Galewsky implementation is limited to the first six hours')
    if args.build_only and args.restart:
        ap.error('--build-only cannot be combined with --restart')
    manifest, inputs = collect_provenance(cfg, config_bytes, args.arm)
    if args.validate_config:
        print('Structural configuration and provenance checks passed; scientific validation pending.')
        return 0
    environment = check_environment()
    if args.check_environment:
        print(json.dumps(environment, indent=2))
        return 0 if environment['ready_to_attempt_build'] else 1
    if environment['errors']:
        raise SystemExit(json.dumps(environment, indent=2))
    import dedalus.public as d3
    restart = load_checkpoint(args.restart, manifest['compatibility_sha256'], cfg) if args.restart else None
    if cfg.get('verification_case', {}).get('kind') == 'galewsky_adjustment' and restart:
        if restart['time_s'] + steps * cfg['numerics']['time_step_s'] > 21600:
            ap.error('Restart would exceed the six-hour benchmark window')
    with RunRecord(args.out, manifest, inputs, config_bytes, restart) as record:
        return evolve(args, cfg, record, restart, d3)
if __name__ == '__main__':
    raise SystemExit(main())
