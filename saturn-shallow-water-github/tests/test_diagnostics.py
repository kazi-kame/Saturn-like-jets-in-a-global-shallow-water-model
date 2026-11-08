#============
#Imports and module setup
#============
import cmath
import math
import unittest
from saturn_poly.diagnostics import spectrum, summarize, phase_speed

#============
#Diagnosticstests
#============
class DiagnosticsTests(unittest.TestCase):

    #============
    #Setup
    #============
    def setUp(self):
        self.lon = [2 * math.pi * i / 128 for i in range(128)]

    #============
    #Test analytic modes and translation
    #============
    def test_analytic_modes_and_translation(self):
        for mode in (3, 6, 10, 17):
            for angle in (0, 0.37):
                values = [12 + 4 * math.cos(mode * (x - angle)) for x in self.lon]
                coefficients = spectrum(self.lon, values, 24)
                self.assertAlmostEqual(abs(coefficients[mode] - 2 * cmath.exp(-1j * mode * angle)), 0, places=12)
                self.assertEqual(summarize(coefficients, 0.1, 0.8)['candidate_m'], mode)

    #============
    #Test constant and mixed
    #============
    def test_constant_and_mixed(self):
        self.assertEqual(summarize(spectrum(self.lon, [7] * 128, 24), 0.1, 0.8)['classification'], 'none')
        values = [math.cos(3 * x) + math.cos(11 * x) for x in self.lon]
        self.assertEqual(summarize(spectrum(self.lon, values, 24), 0.1, 0.8)['classification'], 'mixed')

    #============
    #Test drift sign and phase wrap
    #============
    def test_drift_sign_and_phase_wrap(self):
        times = [float(t) for t in range(100)]
        for speed in (-0.02, 0, 0.02):
            amplitudes = [cmath.exp(-1j * 6 * speed * t) for t in times]
            self.assertAlmostEqual(phase_speed(times, amplitudes, 6), speed, places=12)

    #============
    #Test reject incomplete nonfinite and nyquist
    #============
    def test_reject_incomplete_nonfinite_and_nyquist(self):
        for longitude, values, max_mode in ((self.lon[:-1], [0] * 127, 24), (self.lon, [float('nan')] * 128, 24), (self.lon, [0] * 128, 64)):
            with self.assertRaises(ValueError):
                spectrum(longitude, values, max_mode)

    #============
    #Test reject undefined phase
    #============
    def test_reject_undefined_phase(self):
        with self.assertRaises(ValueError):
            phase_speed([0, 1], [0j, 1j], 6)
        with self.assertRaises(ValueError):
            phase_speed([1, 1], [1j, 1j], 6)
if __name__ == '__main__':
    unittest.main()
