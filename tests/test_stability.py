#============
#Imports and module setup
#============
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
from saturn_poly.dynamics import Sphere
from saturn_poly.stability import polynomial_base, maintained_operator, spectrum, run

#============
#Stabilitytests
#============
class StabilityTests(unittest.TestCase):

    #============
    #Test balance mean and solid rotation
    #============
    def test_balance_mean_and_solid_rotation(self):
        s = Sphere(2, 0.3, 3)
        mu, w = leggauss(40)
        for coefficients in ([0.2], [0.2, 0.03, -0.1]):
            base, bound = polynomial_base(s, 4, coefficients)
            phi = np.arcsin(mu)
            U, Up, H, Hp = base(phi)
            assert_allclose(Hp, -(2 * s.radius * s.rotation * mu * U + U ** 2 * np.tan(phi)) / s.gravity, atol=1e-14)
            self.assertAlmostEqual(np.dot(w, H) / 2, 4)
            self.assertTrue(np.all(H >= bound))
            eps = 1e-06
            assert_allclose((base(phi + eps)[0] - base(phi - eps)[0]) / (2 * eps), Up, atol=1e-10)
        base, _ = polynomial_base(s, 4, [0.2])
        c = (s.radius * s.rotation * 0.2 + 0.2 ** 2 / 2) / s.gravity
        assert_allclose(base(np.arcsin(mu))[2], 4 + c / 3 - c * mu ** 2, atol=1e-14)

    #============
    #Test damped rest analytic blocks
    #============
    def test_damped_rest_analytic_blocks(self):
        s = Sphere(2, 0, 3)
        base, _ = polynomial_base(s, 4, [0])
        nu = 0.01
        op = maintained_operator(s, 2, 5, base, nu)
        for ell in range(2, 6):
            ii = [op.labels.index((kind, ell)) for kind in ('gradient', 'height')]
            actual = np.linalg.eigvals(op.matrix[np.ix_(ii, ii)])
            dv = nu * (ell * (ell + 1) - 1) ** 2 / s.radius ** 4
            dh = nu * (ell * (ell + 1)) ** 2 / s.radius ** 4
            omega2 = s.gravity * 4 * ell * (ell + 1) / s.radius ** 2
            root = np.sqrt(complex((dv - dh) ** 2 / 4 - omega2))
            expected = np.array([-(dv + dh) / 2 - root, -(dv + dh) / 2 + root])
            assert_allclose(actual[np.argsort(actual.imag)], expected[np.argsort(expected.imag)], atol=1e-11)
            j = op.labels.index(('toroidal', ell))
            self.assertAlmostEqual(op.matrix[j, j].real, -dv)
        self.assertLess(max((x['relative_residual'] for x in spectrum(op, 2))), 1e-12)

    #============
    #Test rejections
    #============
    def test_rejections(self):
        s = Sphere(2, 0.3, 3)
        for depth, c in [(0, [0]), (1, [100]), (4, [float('nan')])]:
            with self.assertRaises(ValueError):
                polynomial_base(s, depth, c)
        base, _ = polynomial_base(s, 4, [0])
        with self.assertRaises(ValueError):
            maintained_operator(s, 1, 4, base, -1)

    #============
    #Test pipeline provenance and failure without scan
    #============
    def test_pipeline_provenance_and_failure_without_scan(self):
        config = Path(__file__).resolve().parents[1] / 'configs/stability_manufactured.json'
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'results'
            with patch('saturn_poly.stability.maintained_operator', return_value=None), patch('saturn_poly.stability.spectrum', return_value=[]):
                run(config, out)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(len(report['cases']), 28)
            self.assertFalse(report['convergence_assessed'])
            self.assertEqual((out / 'config.lock.json').read_bytes(), config.read_bytes())
            self.assertIn('saturn_poly/stability.py', report['input_sha256'])
            with self.assertRaises(FileExistsError):
                run(config, out)
            failed = Path(folder) / 'failed'
            with patch('saturn_poly.stability.maintained_operator', side_effect=ValueError('fixture')):
                with self.assertRaisesRegex(ValueError, 'fixture'):
                    run(config, failed)
            self.assertEqual(json.loads((failed / 'report.json').read_text())['status'], 'failed')
