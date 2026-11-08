#============
#Imports and module setup
#============
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import roots_legendre
from assess_runs import ROOT, read_state, run_metadata, digest, write_report
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import planetographic_to_planetocentric

#============
#Modal history
#============
def modal_history(times, coefficients, minimum_amplitude_m):
    t = np.asarray(times, float)
    z = np.asarray(coefficients, complex)
    if t.ndim != 1 or z.ndim != 2 or len(t) != len(z) or (not np.isfinite(t).all()) or (not np.isfinite(z).all()):
        raise ValueError('Finite matched time and mode arrays required')
    if np.any(np.diff(t) <= 0) or minimum_amplitude_m <= 0 or (not np.isfinite(minimum_amplitude_m)):
        raise ValueError('Invalid times or threshold')
    amplitude = np.abs(z)
    intervals = []
    for i, dt in enumerate(np.diff(t)):
        modes = []
        for j in range(z.shape[1]):
            m = j + 1
            valid = bool(min(amplitude[i, j], amplitude[i + 1, j]) >= minimum_amplitude_m)
            angle = float(np.angle(z[i + 1, j] * np.conj(z[i, j]))) if valid else None
            modes.append({'m': m, 'phase_resolved_above_threshold': valid, 'principal_phase_drift_deg_per_hour': -angle / m / dt * 180 / np.pi * 3600 if valid else None, 'phase_branch_spacing_deg_per_hour': 360 / m / dt * 3600, 'log_amplitude_change_per_hour': float(np.log(amplitude[i + 1, j] / amplitude[i, j]) / dt * 3600) if valid else None})
        intervals.append({'start_s': float(t[i]), 'end_s': float(t[i + 1]), 'modes': modes})
    return {'times_s': t.tolist(), 'amplitudes_m': amplitude.tolist(), 'intervals': intervals, 'minimum_phase_amplitude_m': minimum_amplitude_m, 'phase_aliasing_resolved': False, 'coherent_pattern_speed_established': False, 'note': 'Rates do not bridge weak samples. Growth is observed amplitude change under forcing, not an instability eigenvalue.'}

#============
#Ring coefficients
#============
def ring_coefficients(h, lat, cfg):
    p = cfg['physical_constants']
    result = {}
    for pole in ('north', 'south'):
        target = float(planetographic_to_planetocentric(cfg['jets'][pole]['latitude_planetographic_deg'], p['equatorial_radius_m'], p['polar_radius_m']))
        index = int(np.argmin(abs(lat - target)))
        y = h[index] - h[index].mean()
        result[pole] = (float(lat[index]), 2 * np.fft.rfft(y)[1:cfg['numerics']['max_mode'] + 1] / y.size)
    return result

#============
#Snapshot series
#============
def snapshot_series(run):
    run = Path(run).resolve()
    m = run_metadata(run)
    cfg = json.loads((run / 'config.lock.json').read_text())
    files = sorted((run / 'states').glob('step*.nc'))
    if not files:
        raise ValueError('No saved snapshots')
    start = m.get('start_step', 0)
    cadence = cfg['numerics']['save_every_steps']
    expected = [start] + list(range((start // cadence + 1) * cadence, m['final_step'] + 1, cadence))
    if m['final_step'] not in expected:
        expected.append(m['final_step'])
    actual = [int(p.stem[4:]) for p in files]
    if actual != expected:
        raise ValueError('Missing or unexpected output cadence')
    return (m, cfg, files)

#============
#Post trigger summary
#============
def post_trigger_summary(rows, off_s):
    at_off = [r for r in rows if np.isclose(r['time_s'], off_s, rtol=0, atol=1e-09)]
    after = [r for r in rows if r['time_s'] > off_s]
    if not at_off or not after:
        return {'status': 'insufficient_post_trigger_snapshots', 'final_to_shutoff_ratios': None}
    ratios = {}
    for key in ('max_height_difference_m', 'global_height_rms_m', 'global_velocity_rms_m_s'):
        baseline = at_off[0][key]
        ratios[key] = after[-1][key] / baseline if baseline > 1e-12 else None
    return {'status': 'post_trigger_response_measured', 'shutoff_s': off_s, 'final_s': after[-1]['time_s'], 'final_to_shutoff_ratios': ratios, 'interpretation': 'Matched C-B response with ongoing stochastic forcing; not proof of a freely evolving vortex or instability.'}

#============
#Analyze
#============
def analyze(root):
    root = Path(root).resolve()
    names = [arm for arm in 'ABC' if (root / f'saturn_{arm}_6h').is_dir()]
    if not {'B', 'C'} <= set(names):
        raise ValueError('Matched B and C runs required')
    metadata = {}
    cfgs = {}
    paths = {}
    evidence = {}
    for arm in names:
        run = root / f'saturn_{arm}_6h'
        metadata[arm], cfgs[arm], paths[arm] = snapshot_series(run)
        if metadata[arm]['arm'] != arm:
            raise ValueError('Arm label mismatch')
        for p in (run / 'manifest.json', run / 'config.lock.json'):
            evidence[str(p)] = digest(p)
    ref = metadata['B']['compatibility'].copy()
    ref.pop('arm')
    for arm in names:
        c = metadata[arm]['compatibility'].copy()
        c.pop('arm')
        if c != ref:
            raise ValueError('Arms differ beyond intervention')
        if len(paths[arm]) != len(paths['B']):
            raise ValueError('Snapshot counts differ')
    records = {a: {pole: [] for pole in ('north', 'south')} for a in names}
    times = []
    differences = []
    latitude_records = {}
    final_states = {}
    previous_time = None
    cfg = cfgs['B']
    p = cfg['physical_constants']
    n = cfg['numerics']
    for i in range(len(paths['B'])):
        states = {}
        for arm in names:
            path = paths[arm][i]
            s = read_state(path)
            states[arm] = s
            evidence[str(path)] = digest(path)
            if s['layer_thickness'].shape != (n['ncolat'], n['nlon']):
                raise ValueError('Snapshot shape differs from config')
            if not np.isclose(s['time_s'], s['step'] * n['time_step_s'], rtol=0, atol=1e-09):
                raise ValueError('Time/step mismatch')
            if s['step'] != int(path.stem[4:]):
                raise ValueError('Filename/step mismatch')
            for pole, (lat, z) in ring_coefficients(s['layer_thickness'], s['latitude_planetocentric'], cfg).items():
                records[arm][pole].append(z)
                latitude_records[pole] = lat
        b = states['B']
        c = states['C']
        for s in states.values():
            for key in ('latitude_planetocentric', 'longitude_east', 'time_s', 'step'):
                if not np.array_equal(s[key], b[key]):
                    raise ValueError('Mismatched comparison coordinates/time')
        lon = b['longitude_east']
        lat = b['latitude_planetocentric']
        if not np.allclose(lon, np.arange(n['nlon']) * 360 / n['nlon'], rtol=0, atol=1e-10):
            raise ValueError('Complete uniform east-longitude grid required')
        nodes, weights = roots_legendre(n['ncolat'])
        if not np.allclose(np.sin(np.deg2rad(lat)), nodes, rtol=0, atol=1e-12):
            raise ValueError('Gaussian latitude grid required')
        if previous_time is not None and (not np.array_equal(lat, previous_lat)):
            raise ValueError('Latitude grid changed')
        time = b['time_s']
        times.append(time)
        previous_time = time
        previous_lat = lat
        dh = c['layer_thickness'] - b['layer_thickness']
        dv2 = (c['u_east'] - b['u_east']) ** 2 + (c['v_north'] - b['v_north']) ** 2
        rms = lambda squared: float(np.sqrt(np.dot(weights, np.mean(squared, axis=1)) / 2))
        differences.append({'time_s': time, 'max_height_difference_m': float(abs(dh).max()), 'global_height_rms_m': rms(dh * dh), 'global_velocity_rms_m_s': rms(dv2), 'north_cap_max_height_m': float(abs(dh[lat >= 40]).max()), 'south_cap_max_height_m': float(abs(dh[lat <= -40]).max())})
        if i == len(paths['B']) - 1:
            final_states = states
    histories = {a: {pole: modal_history(times, z, cfg['diagnostics']['mode_detection_min_height_amplitude_m']) for pole, z in poles.items()} for a, poles in records.items()}
    f = cfg['forcing']
    start = f['vortex_begin_day'] * 86400
    end = start + f['vortex_duration_days'] * 86400
    return ({'kind': 'saved_wave_history', 'histories': histories, 'C_minus_B': differences, 'sampled_planetocentric_latitudes': latitude_records, 'trigger_begin_s': start, 'trigger_end_s': end, 'post_trigger_snapshot_count': sum((t > end for t in times)), 'post_trigger_response': post_trigger_summary(differences, end), 'evidence_sha256': evidence, 'scientifically_validated': False, 'scope': 'Fixed jet-latitude height harmonics and matched-field differences; not closed-contour polygon detection or free-vortex proof.'}, final_states)

#============
#Plot results
#============
def plot_results(report, states, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, pole in zip(axes[:2], ('north', 'south')):
        for arm, style in [('B', '-'), ('C', '--')]:
            h = report['histories'][arm][pole]
            amps = np.asarray(h['amplitudes_m'])
            for mode, color in [(6, '#2680a8'), (10, '#c77924'), (16, '#7b4da7')]:
                if mode <= amps.shape[1]:
                    ax.plot(np.asarray(h['times_s']) / 3600, amps[:, mode - 1], style, color=color, label=f'{arm} m={mode}')
        ax.set_title(pole.title() + ' jet')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Height-mode amplitude [m]')
        ax.legend(fontsize=8)
    rows = report['C_minus_B']
    axes[2].plot([r['time_s'] / 3600 for r in rows], [r['max_height_difference_m'] for r in rows], label='Global max |C-B|')
    axes[2].plot([r['time_s'] / 3600 for r in rows], [r['global_height_rms_m'] for r in rows], label='Global RMS C-B')
    for ax in axes:
        for t, label in [(report['trigger_begin_s'], 'trigger on'), (report['trigger_end_s'], 'trigger off')]:
            if t <= rows[-1]['time_s']:
                ax.axvline(t / 3600, color='gray', alpha=0.6, linestyle=':')
    axes[2].set_title('Trigger response')
    axes[2].set_xlabel('Time [h]')
    axes[2].set_ylabel('Height difference [m]')
    axes[2].legend(fontsize=8)
    fig.suptitle('Jet-wave amplitudes and trigger response')
    fig.savefig(out / 'wave_history.png', dpi=300)
    fig.savefig(out / 'wave_history.pdf')
    plt.close(fig)
    names = list(states)
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 8), subplot_kw={'projection': 'polar'}, layout='constrained', squeeze=False)
    anomalies = {}
    for arm, s in states.items():
        h = s['layer_thickness']
        a = h - h.mean(axis=1, keepdims=True)
        a = np.where(abs(a) <= 64 * np.finfo(float).eps * max(1, float(abs(h).max())), 0, a)
        anomalies[arm] = a
    for row, pole in enumerate(('north', 'south')):
        lat = states['B']['latitude_planetocentric']
        mask = lat >= 40 if pole == 'north' else lat <= -40
        scale = max(max((float(abs(a[mask]).max()) for a in anomalies.values())), 1e-06)
        for col, arm in enumerate(names):
            s = states[arm]
            r = 90 - abs(lat[mask])
            order = np.argsort(r)
            ax = axes[row, col]
            mesh = ax.pcolormesh(np.deg2rad(s['longitude_east']), r[order], anomalies[arm][mask][order], shading='auto', cmap='RdBu_r', vmin=-scale, vmax=scale, rasterized=True)
            ax.set_ylim(0, 50)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1 if pole == 'north' else 1)
            ax.set_title(f'Arm {arm} — {pole.title()}')
            ax.set_yticks([10, 30, 50], labels=[f'{v}°' + ('N' if pole == 'north' else 'S') for v in (80, 60, 40)])
        fig.colorbar(mesh, ax=list(axes[row]), label='Height minus zonal mean [m]', shrink=0.7)
    fig.suptitle(f"Height anomalies at {rows[-1]['time_s'] / 3600:g} h")
    fig.savefig(out / 'final_common_scale.png', dpi=300)
    fig.savefig(out / 'final_common_scale.pdf')
    plt.close(fig)

#============
#Command-line interface
#============
def main():
    p = argparse.ArgumentParser(description='Wave history')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report, states = analyze(args.root)
    write_report(out / 'wave_history.json', report)
    plot_results(report, states, out)
    print(f'Saved {out}/wave_history.json and plots; no fluid evolution.')
if __name__ == '__main__':
    main()
