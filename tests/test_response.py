#============
#Imports and module setup
#============
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_allclose
from saturn_poly.response import EnergyResponse

#============
#Responsetests
#============
class ResponseTests(unittest.TestCase):

    #============
    #Test diagonal decay and resolvent
    #============
    def test_diagonal_decay_and_resolvent(self):
        analyzer = EnergyResponse(np.diag([-1 + 2j, -3 - 1j]), np.diag([2, 7]))
        self.assertEqual(analyzer.transient_gain(0), 1)
        self.assertAlmostEqual(analyzer.transient_gain(0.4), math.exp(-0.8))
        self.assertAlmostEqual(analyzer.resolvent(2)['amplitude_gain_s'], 1)

    #============
    #Test non normal transient against closed form
    #============
    def test_non_normal_transient_against_closed_form(self):
        k, t = (8.0, 0.5)
        b = k * t
        expected = math.exp(-2 * t) * (2 + b * b + abs(b) * math.sqrt(b * b + 4)) / 2
        analyzer = EnergyResponse([[-1, k], [0, -1]], np.eye(2))
        self.assertAlmostEqual(analyzer.transient_gain(t), expected, places=12)
        self.assertGreater(expected, 1)

    #============
    #Test complex coordinate invariance
    #============
    def test_complex_coordinate_invariance(self):
        L = np.array([[-1, 3 + 2j], [0, -2]], dtype=complex)
        M = np.array([[2, 0.2j], [-0.2j, 1]])
        T = np.array([[2, 1j], [0.3, 0.7]])
        original = EnergyResponse(L, M)
        transformed = EnergyResponse(np.linalg.solve(T, L @ T), T.conj().T @ M @ T)
        self.assertAlmostEqual(original.transient_gain(0.2), transformed.transient_gain(0.2), places=12)
        self.assertAlmostEqual(original.resolvent(-0.7)['amplitude_gain_s'], transformed.resolvent(-0.7)['amplitude_gain_s'], places=12)

    #============
    #Test metric scaling and energy conserving case
    #============
    def test_metric_scaling_and_energy_conserving_case(self):
        L = np.array([[0, 2], [-0.5, 0]])
        for factor in (1e-20, 1, 1e+20):
            analyzer = EnergyResponse(L, factor * np.diag([1, 4]))
            self.assertAlmostEqual(analyzer.transient_gain(0.7), 1, places=12)

    #============
    #Test singular and near singular resolvent
    #============
    def test_singular_and_near_singular_resolvent(self):
        analyzer = EnergyResponse(np.diag([0, -1]), np.eye(2))
        for frequency in (0, 1e-14):
            result = analyzer.resolvent(frequency)
            self.assertEqual(result['status'], 'singular_or_numerically_unresolved')
            self.assertIsNone(result['amplitude_gain_s'])
        self.assertEqual(analyzer.resolvent(1e-06)['status'], 'resolved')
        zero = EnergyResponse([[0]], [[1]]).resolvent(0)
        self.assertIsNone(zero['amplitude_gain_s'])

    #============
    #Test invalid inputs
    #============
    def test_invalid_inputs(self):
        for metric in ([[1, 1], [0, 1]], [[1, 0], [0, -1]], [[0, 0], [0, 0]], [[float('nan'), 0], [0, 1]]):
            with self.assertRaises(ValueError):
                EnergyResponse(np.eye(2), metric)
        analyzer = EnergyResponse([[-1]], [[1]])
        for time in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                analyzer.transient_gain(time)
        with self.assertRaises(ValueError):
            analyzer.resolvent(float('nan'))
        with self.assertRaises(ValueError):
            analyzer.resolvent(0, relative_tolerance=0)

    #============
    #Test inverse norm matches explicit small matrix
    #============
    def test_inverse_norm_matches_explicit_small_matrix(self):
        L = np.array([[-1, 2j], [0.4, -3]])
        A = EnergyResponse(L, np.eye(2))
        expected = np.linalg.norm(np.linalg.inv(0.7j * np.eye(2) - L), 2)
        assert_allclose(A.resolvent(0.7)['amplitude_gain_s'], expected, rtol=1e-13)

    #============
    #Test reference report wiring with mock operator
    #============
    def test_reference_report_wiring_with_mock_operator(self):
        from saturn_poly.dynamics import Sphere
        from saturn_poly.global_linear import reference_report
        toy = SimpleNamespace(matrix=np.array([[-1 + 2j]]), energy_matrix=np.array([[3]]), labels=[('test', 0)], quadrature_points=8)
        with patch('saturn_poly.global_linear.build_operator', return_value=toy):
            report = reference_report(Sphere(1, 0, 1), 2, 4, 1, times_s=[0, 0.5], frequencies_rad_s=[2])
        self.assertEqual(report['eigenvalues_s_inverse'][0]['angular_pattern_speed_rad_s'], -1)
        self.assertAlmostEqual(report['energy_response']['transient_gain'][1]['energy_gain'], math.exp(-1))
        self.assertAlmostEqual(report['energy_response']['resolvent'][0]['amplitude_gain_s'], 1)
