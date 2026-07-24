"""Tests for the TOMAS-JAX adapter (ambrs.tomas_jax).

These tests exercise the pure mapping/binning helpers. They are skipped when the
optional tomas_jax dependency is not installed.
"""

import math
import unittest

import numpy as np

import ambrs.tomas_jax as tomas_jax

_HAVE_TOMAS = tomas_jax._TOMAS_AVAILABLE


@unittest.skipUnless(_HAVE_TOMAS, 'tomas_jax not installed')
class TestSpeciesMapping(unittest.TestCase):
    def test_mapped_species_go_to_expected_indices(self):
        from tomas_jax.core.config import SRTSO4, SRTNH4, SRTORG1
        names = ('SO4', 'OC', 'NH4')
        fracs = (0.5, 0.3, 0.2)
        frac_by_index, remapped = tomas_jax.map_species_fractions(names, fracs)
        self.assertEqual(remapped, ())
        self.assertAlmostEqual(frac_by_index[SRTSO4], 0.5)
        self.assertAlmostEqual(frac_by_index[SRTORG1], 0.3)  # OC -> organic bin
        self.assertAlmostEqual(frac_by_index[SRTNH4], 0.2)

    def test_multiple_organics_aggregate_into_one_slot(self):
        from tomas_jax.core.config import SRTORG1
        names = ('OC', 'ARO1', 'LIM2')
        fracs = (0.2, 0.3, 0.1)
        frac_by_index, remapped = tomas_jax.map_species_fractions(names, fracs)
        self.assertEqual(remapped, ())
        self.assertAlmostEqual(frac_by_index[SRTORG1], 0.6)

    def test_unmapped_species_warns_and_conserves_mass(self):
        from tomas_jax.core.config import SRTSO4, SRTORG1
        names = ('SO4', 'BC')
        fracs = (0.7, 0.3)
        with self.assertWarns(UserWarning):
            frac_by_index, remapped = tomas_jax.map_species_fractions(names, fracs)
        self.assertIn('BC', remapped)
        self.assertAlmostEqual(frac_by_index[SRTSO4], 0.7)
        # BC mass is lumped into the organic slot (mass conserved)
        self.assertAlmostEqual(frac_by_index[SRTORG1], 0.3)
        self.assertAlmostEqual(sum(frac_by_index.values()), 1.0)


@unittest.skipUnless(_HAVE_TOMAS, 'tomas_jax not installed')
class TestLognormalBinning(unittest.TestCase):
    def setUp(self):
        from tomas_jax.core.config import xk_boundaries
        self.xk = np.asarray(xk_boundaries(), dtype=float)
        self.boxvol = 1e6

    def test_number_is_conserved(self):
        N_total = 1e4          # # cm^-3
        gmd = 1e-7             # 100 nm, well inside the 1.7 nm - 17.5 um grid
        log10_gsd = math.log10(1.6)
        Nk = tomas_jax.lognormal_to_bins(
            self.xk, N_total, gmd, log10_gsd, self.boxvol)
        self.assertEqual(Nk.shape, (len(self.xk) - 1,))
        self.assertTrue(np.all(Nk >= 0.0))
        recovered = np.sum(Nk) / self.boxvol
        # midpoint integration over a fully-captured mode should be within a few %
        self.assertLess(abs(recovered - N_total) / N_total, 0.05)

    def test_distribute_mass_matches_total_and_indices(self):
        from tomas_jax.core.config import ICOMP, SRTSO4, SRTORG1
        N_total = 1e4
        gmd = 1e-7
        log10_gsd = math.log10(1.6)
        Nk = tomas_jax.lognormal_to_bins(
            self.xk, N_total, gmd, log10_gsd, self.boxvol)
        frac_by_index = {SRTSO4: 0.8, SRTORG1: 0.2}
        Mk = tomas_jax.distribute_mass(Nk, self.xk, frac_by_index)
        self.assertEqual(Mk.shape, (len(self.xk) - 1, ICOMP))
        m_mid = np.sqrt(self.xk[:-1] * self.xk[1:])
        expected_total = np.sum(Nk * m_mid)
        self.assertAlmostEqual(np.sum(Mk) / expected_total, 1.0, places=6)
        # split respects the requested fractions
        self.assertAlmostEqual(
            np.sum(Mk[:, SRTSO4]) / expected_total, 0.8, places=6)
        self.assertAlmostEqual(
            np.sum(Mk[:, SRTORG1]) / expected_total, 0.2, places=6)


if __name__ == '__main__':
    unittest.main()
