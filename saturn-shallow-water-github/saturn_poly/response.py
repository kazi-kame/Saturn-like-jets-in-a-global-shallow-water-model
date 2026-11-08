#============
#Imports and module setup
#============
import math
import numpy as np
from scipy.linalg import cholesky, expm, solve_triangular, svdvals

#============
#Energyresponse
#============
class EnergyResponse:

    #============
    #Initialize object state
    #============
    def __init__(self, operator, metric):
        L, M = (np.asarray(operator, dtype=complex), np.asarray(metric, dtype=complex))
        if L.ndim != 2 or L.shape[0] == 0 or L.shape[0] != L.shape[1] or (M.shape != L.shape):
            raise ValueError('Operator and metric must be nonempty square matrices of identical shape')
        if not np.all(np.isfinite(L)) or not np.all(np.isfinite(M)):
            raise ValueError('Operator and metric must be finite')
        metric_scale = np.linalg.norm(M)
        hermitian_error = np.linalg.norm(M - M.conj().T)
        if metric_scale == 0 or hermitian_error > 100 * np.finfo(float).eps * len(M) * metric_scale:
            raise ValueError('Energy metric must be Hermitian and nonzero')
        M = (M + M.conj().T) / 2
        try:
            C = cholesky(M, lower=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Energy metric must be positive definite') from exc
        self.metric_condition = float(np.linalg.cond(M))
        if not math.isfinite(self.metric_condition):
            raise ValueError('Numerically singular energy metric')
        self.matrix = solve_triangular(C.T, (C @ L).T, lower=True).T
        if not np.all(np.isfinite(self.matrix)):
            raise ValueError('Nonfinite energy-coordinate operator')

    #============
    #Transient gain
    #============
    def transient_gain(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError('Time must be finite and nonnegative')
        if time_s == 0:
            return 1.0
        try:
            with np.errstate(over='raise', invalid='raise'):
                propagator = expm(time_s * self.matrix)
                if not np.all(np.isfinite(propagator)):
                    raise ValueError('Nonfinite propagator; requested time exceeds numerical range')
                gain = float(svdvals(propagator)[0]) ** 2
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('Transient gain exceeds numerical range') from exc
        if not math.isfinite(gain):
            raise ValueError('Transient gain exceeds numerical range')
        return gain

    #============
    #Resolvent
    #============
    def resolvent(self, frequency_rad_s, relative_tolerance=1e-12):
        if not math.isfinite(frequency_rad_s):
            raise ValueError('Frequency must be finite')
        if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
            raise ValueError('Relative tolerance must lie in (0,1)')
        A = self.matrix
        inverse_resolvent = 1j * frequency_rad_s * np.eye(len(A)) - A
        singular_values = svdvals(inverse_resolvent)
        smallest = float(singular_values[-1])
        scale = max(float(svdvals(A)[0]), abs(frequency_rad_s))
        cutoff = relative_tolerance * scale
        unresolved = smallest <= cutoff
        gain = None if unresolved else 1.0 / smallest
        if gain is not None and (not math.isfinite(gain)):
            raise ValueError('Resolvent gain exceeds numerical range')
        return {'frequency_rad_s': frequency_rad_s, 'status': 'singular_or_numerically_unresolved' if unresolved else 'resolved', 'amplitude_gain_s': gain, 'smallest_singular_value_s_inverse': smallest, 'singularity_cutoff_s_inverse': cutoff, 'relative_tolerance': relative_tolerance}

    #============
    #Report
    #============
    def report(self, times_s=(), frequencies_rad_s=()):
        return {'metric_condition_number': self.metric_condition, 'time_domain_fluid_integration': False, 'transient_gain': [{'time_s': t, 'energy_gain': self.transient_gain(t)} for t in times_s], 'resolvent': [self.resolvent(w) for w in frequencies_rad_s], 'forcing_convention': 'exp(+i omega t); unrestricted state-tendency input; same energy weights for input and output', 'interpretation': 'Worst-case operator norms, not realized stochastic forcing or polygon probability. Inviscid neutral modes need not decay; resolvent is a harmonic particular response, not necessarily an attracting steady response.'}
