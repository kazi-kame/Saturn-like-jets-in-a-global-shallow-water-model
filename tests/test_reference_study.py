#============
#Imports and module setup
#============
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from saturn_poly.reference_study import validate_config, assess_reference, run_reference_study
CONFIG = {'purpose': 'nonrotating_rest_numerical_reference_not_saturn', 'radius_m': 1, 'effective_gravity_m_s2': 1, 'depth_m': 1, 'rotation_rad_s': 0, 'modes': [0, 1], 'truncations': [2, 3], 'quadrature_factors': [2, 3], 'relative_tolerance': 1e-10}

#============
#Exact report
#============
def exact_report(sphere, mode, lmax, depth, nlat):
    values = []
    for degree in range(mode, lmax + 1):
        if degree:
            w = math.sqrt(sphere.gravity * depth * degree * (degree + 1)) / sphere.radius
            values += [-w, 0, w]
        else:
            values += [0]
    return {'mode': mode, 'lmax': lmax, 'quadrature_points': nlat, 'eigenvalues_s_inverse': [{'real': 0, 'imag': x} for x in values], 'max_eigenpair_relative_residual': 0, 'energy_skew_relative_residual': 0}

#============
#Referencestudytests
#============
class ReferenceStudyTests(unittest.TestCase):

    #============
    #Test config rejects physical or ambiguous settings
    #============
    def test_config_rejects_physical_or_ambiguous_settings(self):
        self.assertEqual(validate_config(CONFIG)['case_count'], 8)
        for key, value in (('purpose', 'saturn'), ('rotation_rad_s', 1), ('depth_m', -1), ('modes', [1, 1]), ('truncations', [2]), ('quadrature_factors', [1, 2]), ('relative_tolerance', float('nan')), ('unknown', 1)):
            config = copy.deepcopy(CONFIG)
            config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    #============
    #Test assessment detects wrong spectrum
    #============
    def test_assessment_detects_wrong_spectrum(self):
        from saturn_poly.dynamics import Sphere
        report = exact_report(Sphere(1, 0, 1), 1, 2, 1, 8)
        self.assertTrue(assess_reference(report, CONFIG)['passed'])
        report['eigenvalues_s_inverse'][0]['real'] = 0.1
        self.assertFalse(assess_reference(report, CONFIG)['passed'])

    #============
    #Make fixture
    #============
    def make_fixture(self, root):
        for name, content in (('saturn_poly/example.py', 'x = 1\n'), ('requirements.txt', 'numpy\n')):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        config_path = root / 'config.json'
        config_path.write_text(json.dumps(CONFIG))
        return config_path

    #============
    #Test mocked pipeline preserves inputs and refuses overwrite
    #============
    def test_mocked_pipeline_preserves_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'fixture'
            config_path = self.make_fixture(root)
            output = Path(folder) / 'result'
            with patch('saturn_poly.reference_study.reference_report', side_effect=exact_report) as computation:
                result = run_reference_study(config_path, output, root)
            self.assertTrue(result['reference_checks_passed'])
            self.assertEqual(computation.call_count, 8)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['config_sha256'], hashlib.sha256(config_path.read_bytes()).hexdigest())
            self.assertEqual((output / 'inputs/saturn_poly/example.py').read_bytes(), (root / 'saturn_poly/example.py').read_bytes())
            with self.assertRaises(FileExistsError):
                run_reference_study(config_path, output, root)

    #============
    #Test failed computation is recorded
    #============
    def test_failed_computation_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'fixture'
            config_path = self.make_fixture(root)
            output = Path(folder) / 'result'
            with patch('saturn_poly.reference_study.reference_report', side_effect=RuntimeError('test failure')):
                with self.assertRaises(RuntimeError):
                    run_reference_study(config_path, output, root)
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['status'], 'failed')
