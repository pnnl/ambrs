"""Tests for the MAM4-JAX adapter (ambrs.mam4_jax).

These tests exercise the AMBRS->MAM4 mapping, the lognormal->state construction,
and a short in-process run. They are skipped when the optional mam4_jax
dependency is not installed.
"""

import os
import tempfile
import unittest
import warnings
from math import log10

import numpy as np
import scipy.stats

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
import ambrs.mam4_jax as mam4_jax
from ambrs.scenario import Scenario

_HAVE_MAM4_JAX = mam4_jax._MAM4_JAX_AVAILABLE

# aerosol/gas species fixtures (names must be valid AMBRS species)
so4 = aerosol.AerosolSpecies(name='SO4', molar_mass=97.071, density=1770,
                             hygroscopicity=0.507)
oc = aerosol.AerosolSpecies(name='OC', molar_mass=12.01, density=1000,
                            hygroscopicity=0.1)
bc = aerosol.AerosolSpecies(name='BC', molar_mass=12.01, density=1700,
                            hygroscopicity=1e-10)
na = aerosol.AerosolSpecies(name='Na', molar_mass=58.44, density=1900,
                            hygroscopicity=1.16)
oin = aerosol.AerosolSpecies(name='OIN', molar_mass=100.0, density=2600,
                             hygroscopicity=0.068)
so2 = gas.GasSpecies(name='SO2', molar_mass=64.07)
h2so4 = gas.GasSpecies(name='H2SO4', molar_mass=98.079)

# MAM4's fixed geometric std devs (sigma = 1.8, 1.6, 1.8, 1.6) as log10, so
# fixtures don't trip the mode-width warning.
_L = [log10(1.8), log10(1.6), log10(1.8), log10(1.6)]


def _mode_state(name, species, number, gmd, log10_gsd):
    n = len(species)
    return aerosol.AerosolModeState(
        name=name, species=tuple(species), number=number, geom_mean_diam=gmd,
        log10_geom_std_dev=log10_gsd, mass_fractions=tuple(1.0 / n for _ in species))


def _four_mode_scenario(numbers=(1e8, 1e9, 1e5, 2e8),
                        gmds=(1.1e-7, 2.6e-8, 2.0e-6, 5.0e-8),
                        h2so4_conc=1e-13, so2_conc=1e-9):
    size = aerosol.AerosolModalSizeState(modes=(
        _mode_state('accumulation', [so4, oc, bc, na, oin], numbers[0], gmds[0], _L[0]),
        _mode_state('aitken', [so4, na], numbers[1], gmds[1], _L[1]),
        _mode_state('coarse', [so4, na, oin], numbers[2], gmds[2], _L[2]),
        _mode_state('primary_carbon', [oc, bc], numbers[3], gmds[3], _L[3]),
    ))
    return Scenario(
        aerosols=(so4, oc, bc, na, oin), gases=(so2, h2so4), size=size,
        gas_concs=(so2_conc, h2so4_conc), flux=0.0, relative_humidity=0.5,
        temperature=273.0, pressure=1.0e5, height=500.0)


@unittest.skipUnless(_HAVE_MAM4_JAX, 'mam4_jax not installed')
class TestSpeciesMapping(unittest.TestCase):
    def test_species_map_to_expected_slots_in_accum_mode(self):
        # accumulation mode (index 0) has slots for all of these types
        names = ('SO4', 'OC', 'BC', 'Na', 'OIN')
        fracs = (0.4, 0.2, 0.1, 0.2, 0.1)
        frac_by_pcnst, remapped = mam4_jax.map_species_fractions(0, names, fracs)
        self.assertEqual(remapped, ())
        self.assertAlmostEqual(sum(frac_by_pcnst.values()), 1.0)
        slot_of_type = mam4_jax.SLOT_OF_TYPE[0]
        self.assertAlmostEqual(frac_by_pcnst[slot_of_type[mam4_jax._T_SULFATE]], 0.4)
        self.assertAlmostEqual(frac_by_pcnst[slot_of_type[mam4_jax._T_BC]], 0.1)

    def test_organics_aggregate_into_soa_slot(self):
        names = ('MSA', 'ARO1', 'LIM2')
        fracs = (0.2, 0.2, 0.2)
        frac_by_pcnst, remapped = mam4_jax.map_species_fractions(0, names, fracs)
        self.assertEqual(remapped, ())
        soa_pcnst = mam4_jax.SLOT_OF_TYPE[0][mam4_jax._T_SOA]
        self.assertAlmostEqual(frac_by_pcnst[soa_pcnst], 1.0)

    def test_ammonium_and_nitrate_lump_into_sulfate(self):
        # NH4/NO3 have no slot in any mode -> lump into sulfate, mass conserved
        names = ('SO4', 'NH4', 'NO3')
        fracs = (0.5, 0.3, 0.2)
        with self.assertWarns(UserWarning):
            frac_by_pcnst, remapped = mam4_jax.map_species_fractions(0, names, fracs)
        self.assertIn('NH4', remapped)
        self.assertIn('NO3', remapped)
        sulfate_pcnst = mam4_jax.SLOT_OF_TYPE[0][mam4_jax._T_SULFATE]
        self.assertAlmostEqual(frac_by_pcnst[sulfate_pcnst], 1.0)
        self.assertAlmostEqual(sum(frac_by_pcnst.values()), 1.0)

    def test_species_without_slot_in_mode_lumps(self):
        # aitken mode (index 1) has no black-carbon slot -> BC lumped, warned
        names = ('SO4', 'BC')
        fracs = (0.7, 0.3)
        with self.assertWarns(UserWarning):
            frac_by_pcnst, remapped = mam4_jax.map_species_fractions(1, names, fracs)
        self.assertIn('BC', remapped)
        self.assertAlmostEqual(sum(frac_by_pcnst.values()), 1.0)


@unittest.skipUnless(_HAVE_MAM4_JAX, 'mam4_jax not installed')
class TestLognormalToQ(unittest.TestCase):
    def test_shapes_and_number_round_trip(self):
        modes = _four_mode_scenario().size.modes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state, dropped = mam4_jax.lognormal_to_q(modes, 273.0, 1.0e5)
        self.assertEqual(state['q'].shape, (1, 1, mam4_jax.PCNST))
        self.assertEqual(state['dgncur_a'].shape, (1, 1, mam4_jax.NMODES))
        rho_air = mam4_jax._air_density(273.0, 1.0e5)
        for m, mode in enumerate(modes):
            q_num = state['q'][0, 0, mam4_jax.NUMPTR_AMODE[m]]
            recovered = q_num * rho_air     # [# m^-3]
            self.assertAlmostEqual(recovered / mode.number, 1.0, places=6)
            self.assertAlmostEqual(state['dgncur_a'][0, 0, m], mode.geom_mean_diam)

    def test_mass_volume_consistency(self):
        modes = _four_mode_scenario().size.modes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state, _ = mam4_jax.lognormal_to_q(modes, 273.0, 1.0e5)
        q = state['q'][0, 0]
        rho_air = mam4_jax._air_density(273.0, 1.0e5)
        for m, mode in enumerate(modes):
            # volume implied by the (fixed-width) lognormal
            v_num = mode.number * mam4_jax.DUMFAC_AMODE[m] * mode.geom_mean_diam ** 3
            # volume implied by the per-species masses in q
            v_mass = 0.0
            for slot in range(int(mam4_jax.NSPEC_AMODE[m])):
                pcnst = mam4_jax.LMASSPTR_AMODE[m, slot]
                t = mam4_jax.LSPECTYPE_AMODE[m, slot]
                v_mass += q[pcnst] * rho_air / mam4_jax.SPECDENS_AMODE[t]
            self.assertAlmostEqual(v_mass / v_num, 1.0, places=5)

    def test_gsd_mismatch_warns(self):
        # a mode with a geometric std dev far from MAM4's fixed sigma warns
        modes = (
            _mode_state('accumulation', [so4], 1e8, 1.1e-7, log10(1.3)),
            _mode_state('aitken', [so4], 1e9, 2.6e-8, _L[1]),
            _mode_state('coarse', [so4], 1e5, 2.0e-6, _L[2]),
            _mode_state('primary_carbon', [oc], 2e8, 5.0e-8, _L[3]),
        )
        with self.assertWarns(UserWarning):
            mam4_jax.lognormal_to_q(modes, 273.0, 1.0e5)


@unittest.skipUnless(_HAVE_MAM4_JAX, 'mam4_jax not installed')
class TestCreateInput(unittest.TestCase):
    def _model(self, **kw):
        return mam4_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation=True, condensation=True,
                                     nucleation=True), **kw)

    def test_shapes_flags_and_gas(self):
        model = self._model()
        sc = _four_mode_scenario(h2so4_conc=3e-13, so2_conc=2e-9)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inp = model.create_input(sc, dt=60.0, nstep=10)
        self.assertEqual(inp.q.shape, (1, 1, mam4_jax.PCNST))
        self.assertEqual(inp.dgncur_a.shape, (1, 1, mam4_jax.NMODES))
        self.assertEqual((inp.dt, inp.nstep), (60.0, 10))
        # process flags mapped from AerosolProcesses
        self.assertEqual(
            (inp.mdo_gasaerexch, inp.mdo_newnuc, inp.mdo_coag, inp.mdo_rename),
            (1, 1, 1, 1))
        # gas mixing ratios placed into q
        self.assertAlmostEqual(inp.q[0, 0, mam4_jax._H2SO4_IDX], 3e-13)
        self.assertAlmostEqual(inp.so2_mmr, 2e-9)

    def test_requires_four_modes(self):
        model = self._model()
        one_mode = aerosol.AerosolModalSizeState(
            modes=(_mode_state('accumulation', [so4], 1e8, 1.1e-7, _L[0]),))
        sc = Scenario(aerosols=(so4,), gases=(h2so4,), size=one_mode,
                      gas_concs=(1e-13,), flux=0.0, relative_humidity=0.5,
                      temperature=273.0, pressure=1.0e5, height=500.0)
        self.assertRaises(TypeError, model.create_input, sc, 60.0, 10)

    def test_invalid_dt_nstep(self):
        model = self._model()
        sc = _four_mode_scenario()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertRaises(ValueError, model.create_input, sc, 0.0, 10)
            self.assertRaises(ValueError, model.create_input, sc, 60.0, 0)

    def test_create_inputs_over_ensemble(self):
        spec = ppe.EnsembleSpecification(
            name='mam4jax', aerosols=(so4, oc, bc, na, oin), gases=(so2, h2so4),
            size=aerosol.AerosolModalSizeDistribution(modes=[
                aerosol.AerosolModeDistribution(
                    name='accumulation', species=[so4, oc, bc, na, oin],
                    number=scipy.stats.loguniform(1e8, 1e9),
                    geom_mean_diam=scipy.stats.loguniform(8e-8, 1.5e-7),
                    log10_geom_std_dev=_L[0],
                    mass_fractions=[scipy.stats.uniform(0, 1)] * 5),
                aerosol.AerosolModeDistribution(
                    name='aitken', species=[so4, na],
                    number=scipy.stats.loguniform(1e9, 1e10),
                    geom_mean_diam=scipy.stats.loguniform(2e-8, 4e-8),
                    log10_geom_std_dev=_L[1],
                    mass_fractions=[scipy.stats.uniform(0, 1)] * 2),
                aerosol.AerosolModeDistribution(
                    name='coarse', species=[so4, na, oin],
                    number=scipy.stats.loguniform(1e4, 1e6),
                    geom_mean_diam=scipy.stats.loguniform(1e-6, 3e-6),
                    log10_geom_std_dev=_L[2],
                    mass_fractions=[scipy.stats.uniform(0, 1)] * 3),
                aerosol.AerosolModeDistribution(
                    name='primary_carbon', species=[oc, bc],
                    number=scipy.stats.loguniform(1e8, 1e9),
                    geom_mean_diam=scipy.stats.loguniform(3e-8, 6e-8),
                    log10_geom_std_dev=_L[3],
                    mass_fractions=[scipy.stats.uniform(0, 1)] * 2),
            ]),
            gas_concs=(scipy.stats.loguniform(1e-10, 1e-8),
                       scipy.stats.loguniform(1e-14, 1e-12)),
            flux=scipy.stats.loguniform(1e-11, 1e-8),
            relative_humidity=scipy.stats.uniform(0.1, 0.8),
            temperature=scipy.stats.uniform(260, 40), pressure=101325, height=500)
        ensemble = ppe.sample(spec, 4)
        model = self._model()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inputs = model.create_inputs(ensemble, dt=60.0, nstep=5)
        self.assertEqual(len(inputs), 4)
        self.assertTrue(all(isinstance(i, mam4_jax.Input) for i in inputs))


@unittest.skipUnless(_HAVE_MAM4_JAX, 'mam4_jax not installed')
class TestRun(unittest.TestCase):
    def test_run_returns_populated_output(self):
        from PyParticle import ParticlePopulation
        model = mam4_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation=True, condensation=True))
        sc = _four_mode_scenario()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = model.run(model.create_input(sc, dt=60.0, nstep=3),
                            scenario_name='s0')
        self.assertEqual(out.model_name, 'mam4-jax')
        self.assertEqual(out.scenario_name, 's0')
        self.assertIs(out.scenario, sc)
        self.assertEqual(out.timestep, 3)
        self.assertEqual(set(out.thermodynamics), {'T', 'p', 'RH'})
        self.assertIsInstance(out.particle_population, ParticlePopulation)
        self.assertGreater(out.particle_population.get_Ntot(), 0.0)
        self.assertGreater(out.particle_population.get_tot_dry_mass(), 0.0)

    def test_passthrough_conserves_dry_mass(self):
        # all processes off: calcsize/wateruptake run but conserve total dry mass
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses())
        sc = _four_mode_scenario()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m1 = model.run(model.create_input(sc, dt=60.0, nstep=1)) \
                .particle_population.get_tot_dry_mass()
            m8 = model.run(model.create_input(sc, dt=60.0, nstep=8)) \
                .particle_population.get_tot_dry_mass()
        self.assertLess(abs(m8 - m1) / m1, 1e-4)

    def test_step_function_is_cached(self):
        model = mam4_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation=True, condensation=True))
        sc = _four_mode_scenario()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.run(model.create_input(sc, dt=60.0, nstep=2))
            model.run(model.create_input(sc, dt=60.0, nstep=3))
        self.assertEqual(len(model._step_cache), 1)  # one compiled step reused


@unittest.skipUnless(_HAVE_MAM4_JAX, 'mam4_jax not installed')
class TestDirectRunner(unittest.TestCase):
    def test_runs_ensemble_in_process(self):
        from PyParticle import ParticlePopulation
        from ambrs.runners import DirectRunner
        scenarios = [_four_mode_scenario(numbers=(n, 10 * n, n / 1e3, 2 * n))
                     for n in (1e8, 2e8, 3e8)]
        ensemble = ppe.ensemble_from_scenarios(scenarios)
        model = mam4_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation=True, condensation=True))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inputs = model.create_inputs(ensemble, dt=60.0, nstep=3)
            outputs = DirectRunner(model).run(inputs)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertIsInstance(out.particle_population, ParticlePopulation)
            self.assertGreater(out.particle_population.get_Ntot(), 0.0)
        self.assertEqual([o.scenario_name for o in outputs], ['1', '2', '3'])
        self.assertEqual(len(model._step_cache), 1)

    def test_empty_and_type_validation(self):
        from ambrs.runners import DirectRunner
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses(coagulation=True))
        self.assertEqual(DirectRunner(model).run([]), [])
        self.assertRaises(TypeError, DirectRunner(model).run, 'not-a-list')


@unittest.skipUnless(_HAVE_MAM4_JAX, 'mam4_jax not installed')
class TestFilePersistence(unittest.TestCase):
    def test_write_read_input_round_trip(self):
        model = mam4_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation=True, condensation=True))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inp = model.create_input(_four_mode_scenario(), dt=60.0, nstep=7)
        with tempfile.TemporaryDirectory() as d:
            model.write_input_files(inp, d, 'run0')
            self.assertTrue(os.path.exists(os.path.join(d, 'run0.npz')))
            self.assertTrue(os.path.exists(os.path.join(d, 'run0_params.txt')))
            loaded = model.read_input(d, 'run0')
        np.testing.assert_allclose(loaded.q, inp.q)
        np.testing.assert_allclose(loaded.dgncur_a, inp.dgncur_a)
        self.assertEqual(loaded.nstep, inp.nstep)
        self.assertEqual(loaded.dt, inp.dt)
        self.assertEqual(loaded.mdo_coag, inp.mdo_coag)

    def test_write_requires_existing_dir(self):
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses(coagulation=True))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inp = model.create_input(_four_mode_scenario(), dt=60.0, nstep=2)
        self.assertRaises(OSError, model.write_input_files, inp,
                          '/no/such/dir', 'run0')


@unittest.skipUnless(_HAVE_MAM4_JAX, 'mam4_jax not installed')
class TestInvocationUnsupported(unittest.TestCase):
    def test_invocation_raises(self):
        from ambrs.aerosol_model import NotImplementedError as AmbrsNotImpl
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses(coagulation=True))
        self.assertRaises(AmbrsNotImpl, model.invocation, 'exe', 'prefix')


if __name__ == '__main__':
    unittest.main()
