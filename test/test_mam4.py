# unit tests for the ambrs.mam4 package

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
from ambrs.scenario import Scenario
import ambrs.mam4 as mam4

from math import log10
import numpy as np
import os
import scipy.stats
import tempfile
import unittest

# relevant aerosol and gas species
so4 = aerosol.AerosolSpecies(
    name='SO4',
    molar_mass = 97.071, # NOTE: 1000x smaller than "molecular weight"!
    density = 1770,
    hygroscopicity = 0.507,
)
pom = aerosol.AerosolSpecies(
    name='OC',
    molar_mass = 12.01,
    density = 1000,
    hygroscopicity = 0.5,
)
soa = aerosol.AerosolSpecies(
    name='OC',
    molar_mass = 12.01,
    density = 1000,
    hygroscopicity = 0.5,
)
bc = aerosol.AerosolSpecies(
    name='BC',
    molar_mass = 12.01,
    density = 1000,
    hygroscopicity = 0.5,
)
dst = aerosol.AerosolSpecies(
    name='OIN',
    molar_mass = 135.065,
    density = 1000,
    hygroscopicity = 0.5,
)
ncl = aerosol.AerosolSpecies(
    name='Na',
    molar_mass = 58.44,
    density = 1000,
    hygroscopicity = 0.5,
)

so2 = gas.GasSpecies(
    name='SO2',
    molar_mass = 64.07,
)
h2so4 = gas.GasSpecies(
    name='H2SO4',
    molar_mass = 98.079,
)
#soag = gas.GasSpecies(
#    name='soag',
#    molar_mass = 12.01,
#)

# reference pressure and height
p0 = 101325 # [Pa]
h0 = 500    # [m]

class TestMAM4AerosolModel(unittest.TestCase):
    """Unit tests for ambr.mam4.AerosolModel"""

    def setUp(self):
        self.n = 100
        self.ensemble_spec = ppe.EnsembleSpecification(
            name = 'mam4_ensemble',
            aerosols = (so4, pom, soa, bc, dst, ncl),
            gases = (so2, h2so4),#, soag),
            size = aerosol.AerosolModalSizeDistribution(
                modes = [
                    aerosol.AerosolModeDistribution(
                        name = "accumulation",
                        species = [so4, pom, soa, bc, dst, ncl],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(0.5e-7, 1.1e-7),
                        log10_geom_std_dev = log10(1.6),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # so4
                            scipy.stats.uniform(0, 1), # pom
                            scipy.stats.uniform(0, 1), # soa
                            scipy.stats.uniform(0, 1), # bc
                            scipy.stats.uniform(0, 1), # dst
                            scipy.stats.uniform(0, 1), # ncl
                        ],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "aitken",
                        species = [so4, soa, ncl],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(0.5e-8, 3e-8),
                        log10_geom_std_dev = log10(1.6),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # so4
                            scipy.stats.uniform(0, 1), # soa
                            scipy.stats.uniform(0, 1), # ncl
                        ],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "coarse",
                        species = [dst, ncl, so4, bc, pom, soa],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(1e-6, 2e-6),
                        log10_geom_std_dev = log10(1.8),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # dst
                            scipy.stats.uniform(0, 1), # ncl
                            scipy.stats.uniform(0, 1), # so4
                            scipy.stats.uniform(0, 1), # bc
                            scipy.stats.uniform(0, 1), # pom
                            scipy.stats.uniform(0, 1), # soa
                        ],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "primary carbon",
                        species = [pom, bc],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(1e-8, 6e-8),
                        log10_geom_std_dev = log10(1.8),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # pom
                            scipy.stats.uniform(0, 1), # bc
                        ],
                    ),
                ],
            ),
            gas_concs = (scipy.stats.loguniform(1e5, 1e6) for g in range(3)),
            flux = scipy.stats.loguniform(1e-2*1e-9, 1e1*1e-9),
            relative_humidity = scipy.stats.loguniform(1e-5, 0.99),
            temperature = scipy.stats.uniform(240, 310),
            pressure = p0,
            height = h0,
        )
        self.ensemble = ppe.sample(self.ensemble_spec, self.n)

    def test_create_input(self):
        processes = aerosol.AerosolProcesses(
            aging = True,
            coagulation = True,
        )
        scenario = self.ensemble.member(0)
        dt = 4.0
        nstep = 100
        model = mam4.AerosolModel(processes)
        input = model.create_input(scenario, dt, nstep)

        # timestepping parameters
        self.assertTrue(abs(dt - input.mam_dt) < 1e-12)
        self.assertTrue(nstep, input.mam_nstep)

        # aerosol processes
        self.assertFalse(input.mdo_gaschem)
        self.assertFalse(input.mdo_gasaerexch)
        self.assertTrue(input.mdo_rename)
        self.assertFalse(input.mdo_newnuc)
        self.assertTrue(input.mdo_coag)

        # atmospheric conditions
        self.assertTrue(abs(scenario.temperature - input.temp) < 1e-12)
        self.assertTrue(abs(scenario.pressure - input.press) < 1e-12)
        self.assertTrue(abs(scenario.relative_humidity - input.RH_CLEA) < 1e-12)

        # modal number concentrations
        self.assertTrue(abs(scenario.size.modes[0].number - input.numc1) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[1].number - input.numc2) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[2].number - input.numc3) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[3].number - input.numc4) < 1e-12)

        # accumulation mode species mass fractions
        self.assertEqual('accumulation', scenario.size.modes[0].name)
        self.assertTrue(abs(scenario.size.modes[0].mass_fractions[0] - input.mfso41) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[0].mass_fractions[1] - input.mfpom1) < 1e-12)
        # FIXME: vvv we need to disambiguate POM and SOA (both OC in MOSAIC)
        #self.assertTrue(abs(scenario.size.modes[0].mass_fractions[2] - input.mfsoa1) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[0].mass_fractions[3] - input.mfbc1)  < 1e-12)
        self.assertTrue(abs(scenario.size.modes[0].mass_fractions[4] - input.mfdst1) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[0].mass_fractions[5] - input.mfncl1) < 1e-12)

        # aitken mode species mass fractions
        self.assertEqual('aitken', scenario.size.modes[1].name)
        self.assertTrue(abs(scenario.size.modes[1].mass_fractions[0] - input.mfso42) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[1].mass_fractions[1] - input.mfsoa2) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[1].mass_fractions[2] - input.mfncl2) < 1e-12)

        # coarse mode species mass fractions
        self.assertEqual('coarse', scenario.size.modes[2].name)
        self.assertTrue(abs(scenario.size.modes[2].mass_fractions[0] - input.mfdst3) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[2].mass_fractions[1] - input.mfncl3) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[2].mass_fractions[2] - input.mfso43) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[2].mass_fractions[3] - input.mfbc3)  < 1e-12)
        self.assertTrue(abs(scenario.size.modes[2].mass_fractions[4] - input.mfpom3) < 1e-12)
        # FIXME: vvv we need to disambiguate POM and SOA (both OC in MOSAIC)
        #self.assertTrue(abs(scenario.size.modes[2].mass_fractions[5] - input.mfsoa3) < 1e-12)

        # primary carbon species mass fractions
        self.assertEqual('primary carbon', scenario.size.modes[3].name)
        self.assertTrue(abs(scenario.size.modes[3].mass_fractions[0] - input.mfpom4) < 1e-12)
        self.assertTrue(abs(scenario.size.modes[3].mass_fractions[1] - input.mfbc4)  < 1e-12)

        # test that passing invalid parameters raises exceptions appropriately
        bad_scenario = scenario
        bad_scenario.size = None
        bad_args = [bad_scenario, dt, nstep]
        self.assertRaises(TypeError, model.create_input, *bad_args)

        bad_dt = 0
        bad_args = [scenario, bad_dt, nstep]
        self.assertRaises(ValueError, model.create_input, *bad_args)

        bad_nstep = 0
        bad_args = [scenario, dt, bad_nstep]
        self.assertRaises(ValueError, model.create_input, *bad_args)

    def test_create_inputs(self):
        processes = aerosol.AerosolProcesses(
            aging = True,
            coagulation = True,
        )
        dt = 4.0
        nstep = 100

        model = mam4.AerosolModel(processes)
        inputs = model.create_inputs(self.ensemble, dt, nstep)
        for i, input in enumerate(inputs):
            scenario = self.ensemble.member(i)

            # timestepping parameters
            self.assertTrue(abs(dt - input.mam_dt) < 1e-12)
            self.assertTrue(nstep, input.mam_nstep)

            # aerosol processes
            self.assertFalse(input.mdo_gaschem)
            self.assertFalse(input.mdo_gasaerexch)
            self.assertTrue(input.mdo_rename)
            self.assertFalse(input.mdo_newnuc)
            self.assertTrue(True, input.mdo_coag)

            # atmospheric conditions
            self.assertTrue(abs(scenario.temperature - input.temp) < 1e-12)
            self.assertTrue(abs(scenario.pressure - input.press) < 1e-12)
            self.assertTrue(abs(scenario.relative_humidity - input.RH_CLEA) < 1e-12)

            # modal number concentrations
            self.assertTrue(abs(scenario.size.modes[0].number - input.numc1) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[1].number - input.numc2) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[2].number - input.numc3) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[3].number - input.numc4) < 1e-12)

            # accumulation mode species mass fractions
            self.assertEqual('accumulation', scenario.size.modes[0].name)
            self.assertTrue(abs(scenario.size.modes[0].mass_fractions[0] - input.mfso41) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[0].mass_fractions[1] - input.mfpom1) < 1e-12)
            # FIXME: vvv we need to disambiguate POM and SOA (both OC in MOSAIC)
            #self.assertTrue(abs(scenario.size.modes[0].mass_fractions[2] - input.mfsoa1) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[0].mass_fractions[3] - input.mfbc1)  < 1e-12)
            self.assertTrue(abs(scenario.size.modes[0].mass_fractions[4] - input.mfdst1) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[0].mass_fractions[5] - input.mfncl1) < 1e-12)

            # aitken mode species mass fractions
            self.assertEqual('aitken', scenario.size.modes[1].name)
            self.assertTrue(abs(scenario.size.modes[1].mass_fractions[0] - input.mfso42) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[1].mass_fractions[1] - input.mfsoa2) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[1].mass_fractions[2] - input.mfncl2) < 1e-12)

            # coarse mode species mass fractions
            self.assertEqual('coarse', scenario.size.modes[2].name)
            self.assertTrue(abs(scenario.size.modes[2].mass_fractions[0] - input.mfdst3) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[2].mass_fractions[1] - input.mfncl3) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[2].mass_fractions[2] - input.mfso43) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[2].mass_fractions[3] - input.mfbc3)  < 1e-12)
            self.assertTrue(abs(scenario.size.modes[2].mass_fractions[4] - input.mfpom3) < 1e-12)
            # FIXME: vvv we need to disambiguate POM and SOA (both OC in MOSAIC)
            #self.assertTrue(abs(scenario.size.modes[2].mass_fractions[5] - input.mfsoa3) < 1e-12)

            # primary carbon species mass fractions
            self.assertEqual('primary carbon', scenario.size.modes[3].name)
            self.assertTrue(abs(scenario.size.modes[3].mass_fractions[0] - input.mfpom4) < 1e-12)
            self.assertTrue(abs(scenario.size.modes[3].mass_fractions[1] - input.mfbc4)  < 1e-12)

        # test that passing invalid parameters raises exceptions appropriately
        bad_ensemble = ppe.Ensemble(
            aerosols = self.ensemble.aerosols,
            gases = self.ensemble.gases,
            size = None,
            gas_concs = self.ensemble.gas_concs,
            flux = self.ensemble.flux,
            relative_humidity = self.ensemble.relative_humidity,
            temperature = self.ensemble.temperature,
            pressure = self.ensemble.pressure,
            height = self.ensemble.height,
            specification = self.ensemble.specification,
        )
        bad_args = [bad_ensemble, dt, nstep]
        self.assertRaises(TypeError, model.create_inputs, *bad_args)

        bad_dt = 0
        bad_args = [self.ensemble, bad_dt, nstep]
        self.assertRaises(ValueError, model.create_inputs, *bad_args)

        bad_nstep = 0
        bad_args = [self.ensemble, dt, bad_nstep]
        self.assertRaises(ValueError, model.create_inputs, *bad_args)

    def test_write_input_files(self):
        processes = aerosol.AerosolProcesses(
            aging = True,
            coagulation = True,
        )
        scenario = self.ensemble.member(0)
        dt = 4.0
        nstep = 100
        model = mam4.AerosolModel(processes)
        input = model.create_input(scenario, dt, nstep)
        temp_dir = tempfile.TemporaryDirectory()
        model.write_input_files(input, temp_dir.name, 'namelist')
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, 'namelist')))
        temp_dir.cleanup()


# MAM4 maps SOA via the placeholder 'MSA' species (see ambrs.mam4), so a scenario
# used with create_input needs every mapped species present in each mode.
msa = aerosol.AerosolSpecies(
    name='MSA',
    molar_mass = 96.11,
    density = 1000,
    hygroscopicity = 0.5,
)

def _four_mode_scenario():
    """A 4-mode scenario carrying every species ambrs.mam4 maps, in MAM4's mode
order (accumulation, aitken, coarse, primary carbon)."""
    species = (so4, pom, msa, bc, dst, ncl)
    fracs = tuple(1.0 / len(species) for _ in species)
    modes = tuple(
        aerosol.AerosolModeState(
            name = name, species = species, number = number,
            geom_mean_diam = gmd, log10_geom_std_dev = log10(gsd),
            mass_fractions = fracs)
        for name, number, gmd, gsd in (
            ('accumulation', 1e8, 1.1e-7, 1.8),
            ('aitken', 1e9, 2.6e-8, 1.6),
            ('coarse', 1e5, 2.0e-6, 1.8),
            ('primary carbon', 2e8, 5.0e-8, 1.6),
        ))
    return Scenario(
        aerosols = species, gases = (so2, h2so4),
        size = aerosol.AerosolModalSizeState(modes = modes),
        gas_concs = (1e-9, 1e-13), flux = 0.0, relative_humidity = 0.5,
        temperature = 273.0, pressure = p0, height = h0)


class TestMAM4Output(unittest.TestCase):
    """Unit tests for ambrs.mam4.AerosolModel output handling"""

    def setUp(self):
        self.model = mam4.AerosolModel(
            aerosol.AerosolProcesses(coagulation = True))
        self.input = self.model.create_input(_four_mode_scenario(), 4.0, 100)

    def test_create_input_retains_scenario(self):
        # read_output_files needs the originating scenario to build an Output
        self.assertIsNotNone(self.input.scenario)
        self.assertEqual(len(self.input.scenario.size.modes), 4)

    def test_read_output_files_without_output_raises(self):
        # with no mam_output.nc present, reading output fails cleanly rather
        # than returning bogus data
        with tempfile.TemporaryDirectory() as root:
            dir = os.path.join(root, '1')
            os.mkdir(dir)
            self.model.write_input_files(self.input, dir, '1')
            self.assertRaises(
                OSError, self.model.read_output_files, self.input, dir, '1')

if __name__ == '__main__':
    unittest.main()
