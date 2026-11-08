#============
#Imports and module setup
#============
import unittest
import tempfile
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
from scipy.special import lpmv, gammaln
from saturn_poly.global_linear import _basis, build_operator
from saturn_poly.dynamics import Sphere
from saturn_poly.stability_modes import hemisphere_shares, match_modes, spectral_tail_norm

#============
#Extendedbasistests
#============
class ExtendedBasisTests(unittest.TestCase):

    #============
    #Test normalized recurrence against independent function
    #============
    def test_normalized_recurrence_against_independent_function(self):
        mu, _ = leggauss(180)
        for m in (0, 1, 6, 20):
            B, _, labels = _basis(m, 128, mu)
            for ell in (max(m, 1), 64, 128):
                normalization = np.exp(0.5 * (np.log((2 * ell + 1) / 2) + gammaln(ell - m + 1) - gammaln(ell + m + 1)))
                assert_allclose(B[2, :, labels.index(('height', ell))], normalization * lpmv(m, ell, mu), atol=2e-11, rtol=2e-10)

    #============
    #Test high degree orthonormality including high order
    #============
    def test_high_degree_orthonormality_including_high_order(self):
        mu, w = leggauss(300)
        for m in (0, 20, 80, 128):
            B, _, labels = _basis(m, 128, mu)
            gram = np.einsum('cni,n,cnj->ij', B.conj(), w, B, optimize=True)
            assert_allclose(gram, np.eye(len(labels)), atol=3e-10)

    #============
    #Test extended rest operator
    #============
    def test_extended_rest_operator(self):
        op = build_operator(Sphere(2, 0, 3), 6, 64, depth=4, nlat=196)
        values = np.linalg.eigvals(op.matrix)
        expected = []
        for ell in range(6, 65):
            freq = np.sqrt(3 * 4 * ell * (ell + 1)) / 2
            expected.extend([-freq, 0, freq])
        assert_allclose(np.sort(values.imag), sorted(expected), atol=1e-09)
        assert_allclose(values.real, 0, atol=1e-09)

    #============
    #Test localization and phase invariant matching
    #============
    def test_localization_and_phase_invariant_matching(self):
        mu, _ = leggauss(20)
        _, _, labels = _basis(2, 4, mu)
        coeff = np.eye(len(labels), dtype=complex)
        base = lambda phi: (0 * phi, 0 * phi, 0 * phi + 4, 0 * phi)
        shares = hemisphere_shares(Sphere(2, 0, 3), base, 2, 4, coeff, 32)
        assert_allclose(shares, 0.5, atol=1e-13)
        low = np.eye(3, dtype=complex)
        high = np.eye(4, dtype=complex)[:, [2, 0, 3, 1]] * np.exp(0.7j)
        matches = match_modes(low, high, np.eye(4))
        self.assertEqual([x['high_index'] for x in matches], [1, 3, 0])
        assert_allclose([x['overlap'] for x in matches], 1)

    #============
    #Test saved vectors are physical eigenvectors
    #============
    def test_saved_vectors_are_physical_eigenvectors(self):
        from saturn_poly.stability import spectrum
        sphere = Sphere(2, 0, 3)
        base = lambda phi: (0 * phi, 0 * phi, 0 * phi + 4, 0 * phi)
        op = build_operator(sphere, 2, 6, base=base)
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'modes.npz'
            rows = spectrum(op, 2, archive, sphere, base)
            with np.load(archive, allow_pickle=False) as data:
                c, s = (data['coefficients'], data['eigenvalues'])
                assert_allclose(op.matrix @ c, c * s[None, :], atol=1e-12)
                assert_allclose(np.real(np.sum(c.conj() * (op.energy_matrix @ c), axis=0)), 1, atol=1e-12)
                assert_allclose(s.real, [x['growth_s_inverse'] for x in rows])
            for row in rows:
                self.assertTrue(0 <= row['north_energy_fraction'] <= 1)

    #============
    #Test degree256 basis and derivatives
    #============
    def test_degree256_basis_and_derivatives(self):
        mu, w = leggauss(540)
        for m in (0, 40, 160, 256):
            B, dB, labels = _basis(m, 256, mu)
            self.assertTrue(np.all(np.isfinite(dB)))
            gram = np.einsum('cni,n,cnj->ij', B.conj(), w, B, optimize=True)
            assert_allclose(gram, np.eye(len(labels)), atol=2e-09)
            if m <= 40:
                ell = 256
                norm = np.exp(0.5 * (np.log((2 * ell + 1) / 2) + gammaln(ell - m + 1) - gammaln(ell + m + 1)))
                assert_allclose(B[2, :, labels.index(('height', ell))], norm * lpmv(m, ell, mu), atol=1e-10)

    #============
    #Test tail norm is not an additive fraction
    #============
    def test_tail_norm_is_not_an_additive_fraction(self):
        labels = [('height', i) for i in range(10)]
        c = np.zeros((10, 2), dtype=complex)
        c[0, 0] = 1
        c[-1, 1] = 1
        start, ratio = spectral_tail_norm(c, np.eye(10), labels)
        self.assertEqual(start, 9)
        assert_allclose(ratio, [0, 1])
        metric = np.eye(10)
        metric[0, 9] = metric[9, 0] = -0.9
        c = np.zeros((10, 1), dtype=complex)
        c[0] = c[9] = 1
        _, ratio = spectral_tail_norm(c, metric, labels)
        assert_allclose(ratio, [5])

    #============
    #Test degree256 rest gravity wave operator
    #============
    def test_degree256_rest_gravity_wave_operator(self):
        op = build_operator(Sphere(2, 0, 3), 40, 256, depth=4, nlat=540)
        expected = np.zeros_like(op.matrix)
        for ell in range(40, 257):
            g = op.labels.index(('gradient', ell))
            h = op.labels.index(('height', ell))
            wave = np.sqrt(ell * (ell + 1)) / 2
            expected[g, h] = -3 * wave
            expected[h, g] = 4 * wave
        assert_allclose(op.matrix, expected, atol=2e-08)
