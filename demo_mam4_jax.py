"""Minimal MAM4-JAX example for AMBRS.

Builds a small perturbed-parameter ensemble of 4-mode (accumulation, Aitken,
coarse, primary-carbon) scenarios, runs it *in process* through the MAM4-JAX box
model via ambrs.runners.DirectRunner (MAM4-JAX has no command-line interface, so
it is not run with PoolRunner), and prints a one-line summary of each scenario's
final aerosol state.

Requires the optional mam4_jax dependency (and PyParticle) to be installed. Run
from the repository root:

    python demo_mam4_jax.py
"""
import logging
from math import log10

import scipy.stats as stats

import ambrs
import ambrs.aerosol as aerosol
import ambrs.gas as gas

logging.basicConfig(level=logging.INFO)

# aerosol species (MAM4 represents sulfate, organics, black carbon, sea salt, dust)
so4 = aerosol.AerosolSpecies(name='SO4', molar_mass=97.071, density=1770,
                             hygroscopicity=0.507)
oc = aerosol.AerosolSpecies(name='OC', molar_mass=12.011, density=1000,
                            hygroscopicity=0.01)
bc = aerosol.AerosolSpecies(name='BC', molar_mass=12.011, density=1700,
                            hygroscopicity=1e-10)
na = aerosol.AerosolSpecies(name='Na', molar_mass=58.44, density=1900,
                            hygroscopicity=1.16)
oin = aerosol.AerosolSpecies(name='OIN', molar_mass=100.0, density=2600,
                             hygroscopicity=0.068)

# gas precursors (MAM4 gas concentrations are mixing ratios [kg gas / kg dry air])
so2 = gas.GasSpecies(name='SO2', molar_mass=64.07)
h2so4 = gas.GasSpecies(name='H2SO4', molar_mass=98.079)

# MAM4's fixed geometric std devs per mode (sigma = 1.8, 1.6, 1.8, 1.6)
log10_sigma = [log10(1.8), log10(1.6), log10(1.8), log10(1.6)]


def _mode(name, species, number, gmd, log10_gsd):
    """A single lognormal mode with equal mass fractions across its species."""
    frac = 1.0 / len(species)
    return aerosol.AerosolModeDistribution(
        name=name, species=species, number=number, geom_mean_diam=gmd,
        log10_geom_std_dev=log10_gsd,
        mass_fractions=[stats.uniform(frac, 0) for _ in species])


spec = ambrs.EnsembleSpecification(
    name='mam4_jax_demo',
    aerosols=(so4, oc, bc, na, oin),
    gases=(so2, h2so4),
    size=aerosol.AerosolModalSizeDistribution(modes=[
        _mode('accumulation', [so4, oc, bc, na, oin],
              stats.loguniform(1e8, 1e9), stats.loguniform(8e-8, 1.5e-7),
              log10_sigma[0]),
        _mode('aitken', [so4, na],
              stats.loguniform(1e9, 1e10), stats.loguniform(2e-8, 4e-8),
              log10_sigma[1]),
        _mode('coarse', [so4, na, oin],
              stats.loguniform(1e4, 1e6), stats.loguniform(1e-6, 3e-6),
              log10_sigma[2]),
        _mode('primary_carbon', [oc, bc],
              stats.loguniform(1e8, 1e9), stats.loguniform(3e-8, 6e-8),
              log10_sigma[3]),
    ]),
    # MAM4 gas mixing ratios [kg gas / kg dry air]
    gas_concs=(stats.loguniform(1e-10, 1e-8),   # SO2
               stats.loguniform(1e-14, 1e-12)),  # H2SO4
    flux=stats.loguniform(1e-11, 1e-8),
    relative_humidity=stats.uniform(0.2, 0.6),
    temperature=stats.uniform(270, 30),
    pressure=101325,
    height=500,
)

ensemble = ambrs.sample(spec, 5)

# condensation + coagulation + nucleation over one simulated hour
processes = ambrs.AerosolProcesses(coagulation=True, condensation=True,
                                   nucleation=True)
model = ambrs.mam4_jax.AerosolModel(processes)
inputs = model.create_inputs(ensemble, dt=60.0, nstep=60)

# run every scenario in process (one compiled step is reused across the ensemble)
outputs = ambrs.DirectRunner(model).run(inputs)

print()
print(f'{"scenario":>10} {"N_tot [/m3]":>16} {"dry mass conc":>16}')
for out in outputs:
    pop = out.particle_population
    print(f'{out.scenario_name:>10} {pop.get_Ntot():16.4e} '
          f'{pop.get_tot_dry_mass():16.4e}')
