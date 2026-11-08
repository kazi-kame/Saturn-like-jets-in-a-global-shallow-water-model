#============
#Imports and module setup
#============
import math
import unittest
import numpy as np
from numpy.testing import assert_allclose
from saturn_poly.dynamics import Sphere
from saturn_poly.global_linear import build_operator

#============
#Globallineartests
#============
class GlobalLinearTests(unittest.TestCase):

    #============
    #Test analytic gravity wave spectrum
    #============
    def test_analytic_gravity_wave_spectrum(self):
        for mode in (0, 1, 3):
            a, g, H, lmax = (2.0, 3.0, 4.0, 6)
            op = build_operator(Sphere(a, 0, g), mode, lmax, depth=H)
            expected = []
            for degree in range(mode, lmax + 1):
                if degree:
                    frequency = math.sqrt(g * H * degree * (degree + 1)) / a
                    expected.extend((-frequency, 0, frequency))
                else:
                    expected.append(0)
            eigenvalues = np.linalg.eigvals(op.matrix)
            assert_allclose(np.sort(eigenvalues.imag), np.sort(expected), atol=2e-12)
            assert_allclose(eigenvalues.real, 0, atol=2e-12)

    #============
    #Test vector basis orthonormal and energy skew
    #============
    def test_vector_basis_orthonormal_and_energy_skew(self):
        for mode in (0, 1, 4):
            op = build_operator(Sphere(2, 0.7, 3), mode, 8, depth=4)
            assert_allclose(op.gram_matrix, np.eye(len(op.labels)), atol=2e-12)
            np.linalg.cholesky(op.energy_matrix)
            ML = op.energy_matrix @ op.matrix
            self.assertLess(np.linalg.norm(ML + ML.conj().T) / np.linalg.norm(ML), 2e-12)

    #============
    #Test mass conservation axisymmetric rest
    #============
    def test_mass_conservation_axisymmetric_rest(self):
        op = build_operator(Sphere(2, 0.7, 3), 0, 8, depth=4)
        constant_height = op.labels.index(('height', 0))
        assert_allclose(op.matrix[constant_height, :], 0, atol=2e-12)
        assert_allclose(op.matrix[:, constant_height], 0, atol=2e-12)

    #============
    #Test balanced jet quadrature refinement
    #============
    def test_balanced_jet_quadrature_refinement(self):
        sphere = Sphere(2, 0.3, 3)

        #============
        #Base
        #============
        def base(phi):
            u0, H0 = (0.2, 4)
            c = (sphere.radius * sphere.rotation * u0 + u0 * u0 / 2) / sphere.gravity
            return (u0 * np.cos(phi), -u0 * np.sin(phi), H0 - c * np.sin(phi) ** 2, -2 * c * np.sin(phi) * np.cos(phi))
        low = build_operator(sphere, 2, 6, base=base, nlat=24)
        high = build_operator(sphere, 2, 6, base=base, nlat=48)
        assert_allclose(low.matrix, high.matrix, atol=2e-12, rtol=2e-12)

    #============
    #Test reject invalid profiles and truncation
    #============
    def test_reject_invalid_profiles_and_truncation(self):
        sphere = Sphere(2, 0.7, 3)
        with self.assertRaises(ValueError):
            build_operator(sphere, 5, 3, depth=4)
        with self.assertRaises(ValueError):
            build_operator(sphere, 1, 4, depth=-1)
        with self.assertRaises(ValueError):
            build_operator(sphere, 1, 4, depth=1, base=lambda phi: ())
        with self.assertRaises(ValueError):
            build_operator(sphere, 1, 4, base=lambda phi: (np.ones_like(phi), np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)))
