#============
#Imports and module setup
#============
import cmath
import math
import unittest
from saturn_poly.dynamics import Sphere, LocalField, balanced_height_gradient, nonlinear_tendency, linear_mode_tendency

#============
#Dynamicstests
#============
class DynamicsTests(unittest.TestCase):
    sphere = Sphere(radius=7.0, rotation=0.13, gravity=2.3)

    #============
    #Base
    #============
    def base(self, latitude):
        s, u0, h_eq = (self.sphere, 0.8, 4.0)
        U = u0 * math.cos(latitude)
        Uphi = -u0 * math.sin(latitude)
        c = (s.radius * s.rotation * u0 + u0 ** 2 / 2) / s.gravity
        H = h_eq - c * math.sin(latitude) ** 2
        Hphi = -2 * c * math.sin(latitude) * math.cos(latitude)
        return (LocalField(U, latitude=Uphi), LocalField(H, latitude=Hphi))

    #============
    #Test exact steady solid rotation
    #============
    def test_exact_steady_solid_rotation(self):
        for phi in (-1.2, -0.5, 0, 0.7, 1.2):
            u, h = self.base(phi)
            self.assertAlmostEqual(balanced_height_gradient(self.sphere, phi, u.value), h.latitude)
            for value in nonlinear_tendency(self.sphere, phi, u, LocalField(0), h):
                self.assertAlmostEqual(value, 0, places=13)

    #============
    #Test linearization against nonlinear directional derivative
    #============
    def test_linearization_against_nonlinear_directional_derivative(self):
        p, q, r = (0.3 + 0.2j, -0.4 + 0.1j, 0.2 - 0.3j)
        p_phi, q_phi, r_phi = (-0.2 + 0.15j, 0.1 + 0.4j, -0.3 + 0.2j)
        for phi in (-1.0, -0.3, 0.0, 0.8):
            U, H = self.base(phi)
            for mode in (0, 3, 6, 10):
                target = linear_mode_tendency(self.sphere, phi, mode, U, H, p, q, r, q_phi, r_phi)
                for angle in (0, 0.41):
                    phase = cmath.exp(1j * angle)

                    #============
                    #Tendency
                    #============
                    def tendency(eps):
                        fields = []
                        for base, z, dz in ((U, p, p_phi), (LocalField(0), q, q_phi), (H, r, r_phi)):
                            fields.append(LocalField(base.value + eps * (z * phase).real, eps * (1j * mode * z * phase).real, base.latitude + eps * (dz * phase).real))
                        return nonlinear_tendency(self.sphere, phi, *fields)
                    eps = 1e-05
                    plus, minus = (tendency(eps), tendency(-eps))
                    for a, b, expected in zip(plus, minus, target):
                        self.assertAlmostEqual((a - b) / (2 * eps), (expected * phase).real, places=9)

    #============
    #Test rest state and mass mode
    #============
    def test_rest_state_and_mass_mode(self):
        U, H = (LocalField(0), LocalField(4))
        for mode in (0, 4):
            tendency = linear_mode_tendency(self.sphere, 0.3, mode, U, H, 0, 0, 1, 0, 0)
            self.assertEqual(tendency[1], 0)
            self.assertEqual(tendency[2], 0)
            if mode == 0:
                self.assertEqual(tendency[0], 0)

    #============
    #Test reject invalid geometry and unbalanced state
    #============
    def test_reject_invalid_geometry_and_unbalanced_state(self):
        with self.assertRaises(ValueError):
            Sphere(0, 1, 1)
        with self.assertRaises(ValueError):
            balanced_height_gradient(self.sphere, math.pi / 2, 0)
        with self.assertRaises(ValueError):
            nonlinear_tendency(self.sphere, 0, LocalField(0), LocalField(0), LocalField(-1))
        with self.assertRaises(ValueError):
            linear_mode_tendency(self.sphere, 0.5, 2, LocalField(1), LocalField(4), 1, 1, 1, 0, 0)
