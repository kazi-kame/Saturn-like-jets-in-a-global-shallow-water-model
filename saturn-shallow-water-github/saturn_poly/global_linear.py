#============
#Imports and module setup
#============
from dataclasses import dataclass
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve
from .dynamics import Sphere

#============
#Linearoperator
#============
@dataclass
class LinearOperator:
    matrix: np.ndarray
    energy_matrix: np.ndarray
    gram_matrix: np.ndarray
    labels: list
    latitudes: np.ndarray
    quadrature_points: int

#============
#Basis
#============
def _basis(mode, lmax, mu):
    cosine = np.sqrt(1 - mu ** 2)
    tangent = mu / cosine
    columns, derivatives, labels = ([], [], [])
    diagonal = np.full_like(mu, 1 / math.sqrt(2))
    for order in range(1, mode + 1):
        diagonal *= -math.sqrt((2 * order + 1) / (2 * order)) * cosine
    previous = np.zeros_like(mu)
    current = diagonal
    for degree in range(mode, lmax + 1):
        if degree > mode:
            factor = math.sqrt((4 * degree ** 2 - 1) / (degree ** 2 - mode ** 2))
            back = math.sqrt((2 * degree + 1) * ((degree - 1) ** 2 - mode ** 2) / ((2 * degree - 3) * (degree ** 2 - mode ** 2))) if degree > mode + 1 else 0
            previous, current = (current, factor * mu * current - back * previous)
        Y = current
        coupling = math.sqrt((2 * degree + 1) / (2 * degree - 1) * (degree ** 2 - mode ** 2)) if degree > mode else 0
        dY = (coupling * previous - degree * mu * Y) / cosine
        ddY = tangent * dY + (mode ** 2 / cosine ** 2 - degree * (degree + 1)) * Y
        zero = np.zeros_like(Y, dtype=complex)
        if degree > 0:
            scale = math.sqrt(degree * (degree + 1))
            east = 1j * mode * Y / cosine / scale
            north = dY / scale
            east_phi = 1j * mode * (dY + Y * tangent) / cosine / scale
            north_phi = ddY / scale
            columns.extend((np.stack((east, north, zero)), np.stack((-north, east, zero))))
            derivatives.extend((np.stack((east_phi, north_phi, zero)), np.stack((-north_phi, east_phi, zero))))
            labels.extend((('gradient', degree), ('toroidal', degree)))
        columns.append(np.stack((zero, zero, Y)))
        derivatives.append(np.stack((zero, zero, dY)))
        labels.append(('height', degree))
    return (np.stack(columns, axis=2), np.stack(derivatives, axis=2), labels)

#============
#Build operator
#============
def build_operator(sphere: Sphere, mode, lmax, depth=None, base=None, nlat=None):
    if isinstance(mode, bool) or not isinstance(mode, int) or isinstance(lmax, bool) or (not isinstance(lmax, int)) or (not 0 <= mode <= lmax) or (not 1 <= lmax <= 256):
        raise ValueError('Require integer 0 <= mode <= lmax and 1 <= lmax <= 256')
    if (depth is None) == (base is None):
        raise ValueError('Supply exactly one of depth or a base profile')
    if nlat is None:
        nlat = 2 * lmax + 4
    if isinstance(nlat, bool) or not isinstance(nlat, int) or (not lmax + 2 <= nlat <= 2048):
        raise ValueError('Require lmax+2 <= quadrature points <= 2048')
    mu, weights = leggauss(nlat)
    phi = np.arcsin(mu)
    cosine = np.sqrt(1 - mu ** 2)
    tangent = mu / cosine
    a, g = (sphere.radius, sphere.gravity)
    f = 2 * sphere.rotation * mu
    if base is None:
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError('Rest depth must be positive and finite')
        U, Up, H, Hp = (np.zeros(nlat), np.zeros(nlat), np.full(nlat, depth), np.zeros(nlat))
    else:
        arrays = tuple((np.asarray(x, dtype=float) for x in base(phi)))
        if len(arrays) != 4 or any((x.shape != phi.shape for x in arrays)):
            raise ValueError('Base must return four arrays on the supplied latitude grid')
        U, Up, H, Hp = arrays
        if not all((np.all(np.isfinite(x)) for x in arrays)) or np.any(H <= 0):
            raise ValueError('Base must be finite with positive sampled depth')
        expected = -(a * f * U + U ** 2 * tangent) / g
        if not np.allclose(Hp, expected, rtol=1e-10, atol=1e-10):
            raise ValueError('Base does not satisfy sampled gradient-wind balance')
    B, dB, labels = _basis(mode, lmax, mu)
    p, q, r = B
    qp, rp = (dB[1], dB[2])
    ik = 1j * mode / (a * cosine)
    action = np.stack((-(ik * U)[:, None] * p + (f + (U * tangent - Up) / a)[:, None] * q - (ik * g)[:, None] * r, -(ik * U)[:, None] * q - (f + 2 * U * tangent / a)[:, None] * p - g * rp / a, -(ik * U)[:, None] * r - (ik * H)[:, None] * p - (H[:, None] * qp + (Hp - H * tangent)[:, None] * q) / a))
    energy_weights = a * a * weights[None, :] * np.stack((H, H, np.full(nlat, g)))
    M = np.einsum('cni,cn,cnj->ij', B.conj(), energy_weights, B, optimize=True)
    K = np.einsum('cni,cn,cnj->ij', B.conj(), energy_weights, action, optimize=True)
    gram = np.einsum('cni,n,cnj->ij', B.conj(), weights, B, optimize=True)
    L = solve(M, K, assume_a='pos')
    if not np.all(np.isfinite(L)):
        raise ValueError('Nonfinite operator; check scales and truncation')
    return LinearOperator(L, M, gram, labels, phi, nlat)

#============
#Reference report
#============
def reference_report(sphere, mode, lmax, depth, nlat=None, times_s=(), frequencies_rad_s=()):
    op = build_operator(sphere, mode, lmax, depth=depth, nlat=nlat)
    eigenvalues, vectors = np.linalg.eig(op.matrix)
    order = np.argsort(eigenvalues.imag)
    scale = np.linalg.norm(op.matrix)
    residuals = [np.linalg.norm(op.matrix @ v - s * v) / (max(scale, np.finfo(float).tiny) * np.linalg.norm(v)) for s, v in zip(eigenvalues, vectors.T)]
    ML = op.energy_matrix @ op.matrix
    energy_residual = np.linalg.norm(ML + ML.conj().T) / max(np.linalg.norm(ML), np.finfo(float).tiny)
    report = {'kind': 'global_linear_rest_reference_not_saturn_prediction', 'time_integration_performed': False, 'mode': mode, 'lmax': lmax, 'quadrature_points': op.quadrature_points, 'sphere_radius_m': sphere.radius, 'rotation_rad_s': sphere.rotation, 'effective_gravity_m_s2': sphere.gravity, 'depth_m': depth, 'basis_dimension': len(op.labels), 'max_eigenpair_relative_residual': float(max(residuals)), 'energy_skew_relative_residual': float(energy_residual), 'eigenvalues_s_inverse': [{'real': float(eigenvalues[i].real), 'imag': float(eigenvalues[i].imag), 'angular_pattern_speed_rad_s': -float(eigenvalues[i].imag) / mode if mode else None} for i in order], 'limitations': 'Rest-state reference only. Roundoff-sized real parts are not evidence of instability. No Saturn calibration or nonlinear solver.'}
    if times_s or frequencies_rad_s:
        from .response import EnergyResponse
        report['energy_response'] = EnergyResponse(op.matrix, op.energy_matrix).report(times_s, frequencies_rad_s)
    return report
