#============
#Imports and module setup
#============
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from saturn_poly.dynamics import Sphere
from saturn_poly.jet_reference import smooth_base

#============
#Jetreferencetests
#============
class JetReferenceTests(unittest.TestCase):

    #============
    #Test balance derivatives and mean
    #============
    def test_balance_derivatives_and_mean(self):
        sphere = Sphere(2, 0.3, 3)
        jets = [[0.2, 0.7, 0.08], [0.3, -0.4, 0.12]]
        base, bound = smooth_base(sphere, 4, jets)
        phi = np.linspace(-1.55, 1.55, 501)
        U, Up, H, Hp = base(phi)
        assert_allclose(Hp, -(2 * sphere.radius * sphere.rotation * np.sin(phi) * U + U ** 2 * np.tan(phi)) / sphere.gravity, atol=1e-14)
        step = 1e-06
        assert_allclose((base(phi + step)[0] - base(phi - step)[0]) / (2 * step), Up, atol=1e-08)
        assert_allclose((base(phi + step)[2] - base(phi - step)[2]) / (2 * step), Hp, atol=1e-08)
        mean = quad(lambda x: float(base(np.arcsin(x))[2]), -1, 1, points=[-0.4, 0.7], epsabs=1e-10)[0] / 2
        self.assertAlmostEqual(mean, 4, places=10)
        self.assertTrue(np.all(H >= bound))

    #============
    #Test zero and invalid
    #============
    def test_zero_and_invalid(self):
        sphere = Sphere(2, 0.3, 3)
        base, bound = smooth_base(sphere, 4, [[0, 0, 0.1]])
        assert_allclose(base(np.array([-0.8, 0, 0.8]))[2], 4)
        self.assertEqual(bound, 4)
        for jets in ([[1, 0, 0]], [[1, 1, 0.1]], [[float('nan'), 0, 0.1]]):
            with self.assertRaises(ValueError):
                smooth_base(sphere, 4, jets)

    #============
    #Test endpoint minimum for eastward jets
    #============
    def test_endpoint_minimum_for_eastward_jets(self):
        s = Sphere(2, 0.3, 3)
        base, bound = smooth_base(s, 4, [[0.2, 0.7, 0.08], [0.3, -0.4, 0.12]])
        phi = np.linspace(-np.pi / 2, np.pi / 2, 1001)
        endpoint = min(base(phi)[2][0], base(phi)[2][-1])
        self.assertLessEqual(bound, endpoint)
        self.assertLess(endpoint - bound, 5e-09)

    #============
    #Test manufactured proxy conversion
    #============
    def test_manufactured_proxy_conversion(self):
        from saturn_poly.jet_reference import reconstruct
        config = dict(physical_constants=dict(equatorial_radius_m=2, polar_radius_m=2), jets={name: dict(latitude_planetographic_deg=lat, peak_eastward_m_s=1, fwhm_planetographic_deg=3) for name, lat in [('north', 60), ('south', -60)]})
        jets, old = reconstruct(config)
        self.assertAlmostEqual(jets[0][0], 2)
        self.assertAlmostEqual(jets[0][1], np.sqrt(3) / 2)
        assert_allclose(old(np.deg2rad(np.array([60, -60]))), 1, atol=1e-14)
