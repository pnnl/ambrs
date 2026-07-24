"""Tests for the TOMAS-JAX adapter (ambrs.tomas_jax).

These tests exercise the pure mapping/binning helpers. They are skipped when the
optional tomas_jax dependency is not installed.
"""

import math
import unittest
import warnings
from math import log10

import numpy as np
import scipy.stats

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
import ambrs.tomas_jax as tomas_jax

_HAVE_TOMAS = tomas_jax._TOMAS_AVAILABLE

# aerosol/gas species fixtures (names must be valid AMBRS species)
so4 = aerosol.AerosolSpecies(name='SO4', molar_mass=97.071, density=1770,
                             hygroscopicity=0.507)
oc = aerosol.AerosolSpecies(name='OC', molar_mass=12.01, density=1000,
                            hygroscopicity=0.1)
bc = aerosol.AerosolSpecies(name='BC', molar_mass=12.01, density=1800,
                            hygroscopicity=0.0)
so2 = gas.GasSpecies(name='SO2', molar_mass=64.07)
h2so4 = gas.GasSpecies(name='H2SO4', molar_mass=98.079)


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


@unittest.skipUnless(_HAVE_TOMAS, 'tomas_jax not installed')
class TestCreateInput(unittest.TestCase):
    def setUp(self):
        self.n = 8
        self.spec = ppe.EnsembleSpecification(
            name='tomas_ensemble',
            aerosols=(so4, oc, bc),
            gases=(so2, h2so4),
            size=aerosol.AerosolModalSizeDistribution(
                modes=[
                    aerosol.AerosolModeDistribution(
                        name='aitken',
                        species=[so4],
                        number=scipy.stats.loguniform(1e8, 1e9),
                        geom_mean_diam=scipy.stats.loguniform(2e-8, 4e-8),
                        log10_geom_std_dev=log10(1.6),
                        mass_fractions=[scipy.stats.uniform(0, 1)],
                    ),
                    aerosol.AerosolModeDistribution(
                        name='accumulation',
                        species=[so4, oc, bc],
                        number=scipy.stats.loguniform(1e8, 1e9),
                        geom_mean_diam=scipy.stats.loguniform(8e-8, 1.5e-7),
                        log10_geom_std_dev=log10(1.6),
                        mass_fractions=[scipy.stats.uniform(0, 1),
                                        scipy.stats.uniform(0, 1),
                                        scipy.stats.uniform(0, 1)],
                    ),
                ],
            ),
            gas_concs=(scipy.stats.loguniform(1e6, 1e7) for _ in range(2)),
            flux=scipy.stats.loguniform(1e-11, 1e-8),
            relative_humidity=scipy.stats.uniform(0.1, 0.8),
            temperature=scipy.stats.uniform(260, 300),
            pressure=101325,
            height=500,
        )
        self.ensemble = ppe.sample(self.spec, self.n)
        self.processes = aerosol.AerosolProcesses(
            coagulation=True, condensation=True, nucleation=True)

    def _model(self):
        return tomas_jax.AerosolModel(self.processes)

    def test_create_input_shapes_and_fields(self):
        from tomas_jax.core.config import ICOMP, N_GAS_SPECIES, SRTSO4
        model = self._model()
        scenario = self.ensemble.member(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # BC lumping warning is expected
            inp = model.create_input(scenario, dt=60.0, nstep=10)
        nbins = len(inp.xk) - 1
        self.assertEqual(inp.Nk.shape, (nbins,))
        self.assertEqual(inp.Mk.shape, (nbins, ICOMP))
        self.assertEqual(inp.Gc.shape, (N_GAS_SPECIES,))
        self.assertEqual(inp.dt, 60.0)
        self.assertEqual(inp.nstep, 10)
        # processes emitted in canonical operator-split order
        self.assertEqual(inp.processes, ('nucleation', 'coagulation', 'condensation'))
        # H2SO4 gas placed into Gc
        self.assertGreater(inp.Gc[SRTSO4], 0.0)
        # BC has no TOMAS slot -> recorded as dropped/remapped
        self.assertIn('BC', inp.dropped_species)
        self.assertTrue(np.all(inp.Nk >= 0.0))

    def test_number_and_mass_consistency(self):
        model = self._model()
        scenario = self.ensemble.member(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inp = model.create_input(scenario, dt=60.0, nstep=10)
        total_number = sum(m.number for m in scenario.size.modes)  # [# m^-3]
        recovered = np.sum(inp.Nk) / model.boxvol * 1e6            # [# m^-3]
        self.assertLess(abs(recovered - total_number) / total_number, 0.10)
        # mass fractions sum to 1 per mode (ppe normalizes), so total aerosol
        # mass equals sum(Nk * mean-bin-mass)
        xk = inp.xk
        m_mid = np.sqrt(xk[:-1] * xk[1:])
        self.assertAlmostEqual(np.sum(inp.Mk) / np.sum(inp.Nk * m_mid), 1.0, places=5)

    def test_create_inputs_over_ensemble(self):
        model = self._model()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inputs = model.create_inputs(self.ensemble, dt=60.0, nstep=10)
        self.assertEqual(len(inputs), self.n)
        self.assertTrue(all(isinstance(i, tomas_jax.Input) for i in inputs))

    def test_invalid_arguments(self):
        model = self._model()
        scenario = self.ensemble.member(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertRaises(ValueError, model.create_input, scenario, 0.0, 10)
            self.assertRaises(ValueError, model.create_input, scenario, 60.0, 0)
            bad = self.ensemble.member(0)
            bad.size = None
            self.assertRaises(TypeError, model.create_input, bad, 60.0, 10)


def _single_mode_scenario(number=1e12, gmd=5e-8, h2so4_conc=0.0):
    from ambrs.scenario import Scenario
    mode = aerosol.AerosolModeState(
        name='aitken', species=(so4,), number=number, geom_mean_diam=gmd,
        log10_geom_std_dev=log10(1.6), mass_fractions=(1.0,))
    size = aerosol.AerosolModalSizeState(modes=(mode,))
    return Scenario(
        aerosols=(so4,), gases=(h2so4,), size=size, gas_concs=(h2so4_conc,),
        flux=0.0, relative_humidity=0.5, temperature=298.0, pressure=101325.0,
        height=500.0)


@unittest.skipUnless(_HAVE_TOMAS, 'tomas_jax not installed')
class TestRun(unittest.TestCase):
    def test_run_returns_populated_output(self):
        from PyParticle import ParticlePopulation
        model = tomas_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation=True, condensation=True))
        sc = _single_mode_scenario(number=1e10, h2so4_conc=1e7)
        out = model.run(model.create_input(sc, dt=60.0, nstep=5),
                        scenario_name='s0')
        self.assertEqual(out.model_name, 'tomas-jax')
        self.assertEqual(out.scenario_name, 's0')
        self.assertIs(out.scenario, sc)
        self.assertEqual(out.timestep, 5)
        self.assertEqual(set(out.thermodynamics), {'T', 'p', 'RH'})
        self.assertIsInstance(out.particle_population, ParticlePopulation)
        self.assertGreater(out.particle_population.get_Ntot(), 0.0)

    def test_coagulation_conserves_dry_mass_and_reduces_number(self):
        model = tomas_jax.AerosolModel(aerosol.AerosolProcesses(coagulation=True))
        sc = _single_mode_scenario(number=1e12)  # high enough for active coag
        p_early = model.run(model.create_input(sc, dt=60.0, nstep=1)).particle_population
        p_late = model.run(model.create_input(sc, dt=60.0, nstep=40)).particle_population
        # coagulation reduces number
        self.assertLess(p_late.get_Ntot(), p_early.get_Ntot())
        # ...and conserves dry mass
        m0, m1 = p_early.get_tot_dry_mass(), p_late.get_tot_dry_mass()
        self.assertLess(abs(m1 - m0) / m0, 1e-6)

    def test_step_function_is_cached(self):
        model = tomas_jax.AerosolModel(aerosol.AerosolProcesses(coagulation=True))
        sc = _single_mode_scenario(number=1e10)
        model.run(model.create_input(sc, dt=60.0, nstep=2))
        model.run(model.create_input(sc, dt=60.0, nstep=3))
        self.assertEqual(len(model._step_cache), 1)  # one compiled step reused


if __name__ == '__main__':
    unittest.main()
