#============
#Imports and module setup
#============
import cmath
import math

#============
#Calculate the modal spectrum
#============
def spectrum(longitudes, displacement, max_mode):
    n = len(longitudes)
    if n != len(displacement) or n < 4:
        raise ValueError('Need matching arrays with at least four samples')
    if not isinstance(max_mode, int) or not 1 <= max_mode <= (n - 1) // 2:
        raise ValueError('max_mode must be positive and below Nyquist')
    if not all((math.isfinite(x) for x in [*longitudes, *displacement])):
        raise ValueError('Samples must be finite; partial coverage is unsupported')
    step = 2 * math.pi / n
    if not all((math.isclose(x, longitudes[0] + i * step, abs_tol=1e-10, rel_tol=0) for i, x in enumerate(longitudes))):
        raise ValueError('Require increasing uniform radians covering one period, endpoint excluded')
    mean = math.fsum(displacement) / n
    return {m: sum(((x - mean) * cmath.exp(-1j * m * lon) for lon, x in zip(longitudes, displacement))) / n for m in range(1, max_mode + 1)}

#============
#Summarize
#============
def summarize(coefficients, minimum_amplitude, minimum_share):
    if not math.isfinite(minimum_amplitude) or minimum_amplitude <= 0:
        raise ValueError('minimum_amplitude must be finite and positive')
    if not math.isfinite(minimum_share) or not 0.5 < minimum_share <= 1:
        raise ValueError('minimum_share must be in (0.5, 1]')
    if not coefficients or any((m < 1 or not math.isfinite(abs(a)) for m, a in coefficients.items())):
        raise ValueError('Need finite positive-mode coefficients')
    power = {m: abs(a) ** 2 for m, a in coefficients.items()}
    total = math.fsum(power.values())
    peak = max(power, key=power.get)
    amplitude = 2 * abs(coefficients[peak])
    share = power[peak] / total if total else 0.0
    classification = 'none' if amplitude < minimum_amplitude else 'candidate' if share >= minimum_share else 'mixed'
    return {'classification': classification, 'candidate_m': peak if classification == 'candidate' else None, 'peak_amplitude_m': amplitude, 'peak_power_share': share, 'scope': 'retained-mode spectrum only; not polygon confirmation'}

#============
#Phase speed
#============
def phase_speed(times, amplitudes, mode):
    if mode < 1 or not isinstance(mode, int) or len(times) != len(amplitudes) or (len(times) < 2):
        raise ValueError('Need positive integer mode and matching time series')
    if not all((math.isfinite(t) for t in times)) or any((b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('Times must be finite and strictly increasing')
    if any((not math.isfinite(abs(a)) or abs(a) == 0 for a in amplitudes)):
        raise ValueError('Phase requires finite nonzero amplitudes')
    phases = [cmath.phase(amplitudes[0])]
    for previous, current in zip(amplitudes, amplitudes[1:]):
        delta = cmath.phase(current) - cmath.phase(previous)
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        if math.isclose(abs(delta), math.pi, abs_tol=1e-10):
            raise ValueError('Ambiguous phase increment at pi')
        phases.append(phases[-1] + delta)
    centered_t = [t - times[0] for t in times]
    mean_t = math.fsum(centered_t) / len(times)
    mean_phase = math.fsum(phases) / len(phases)
    slope = math.fsum(((t - mean_t) * (p - mean_phase) for t, p in zip(centered_t, phases)))
    slope /= math.fsum(((t - mean_t) ** 2 for t in centered_t))
    return -slope / mode
