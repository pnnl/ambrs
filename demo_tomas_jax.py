"""Minimal TOMAS-JAX example for AMBRS.

Builds a small perturbed-parameter ensemble, runs it *in process* through the
TOMAS-JAX box model via ambrs.runners.DirectRunner (TOMAS-JAX has no command-line
interface, so it is not run with PoolRunner), and prints a one-line summary of
each scenario's final aerosol state.

Requires the optional tomas_jax dependency (and PyParticle) to be installed. Run
from the repository root:

    python demo_tomas_jax.py
"""
import logging
from math import log10

import scipy.stats as stats

import ambrs
import ambrs.aerosol as aerosol
import ambrs.gas as gas

logging.basicConfig(level=logging.INFO)

# a single sulfate species and its condensable gas precursor
so4 = aerosol.AerosolSpecies(name='SO4', molar_mass=97.071, density=1770,
                             hygroscopicity=0.507)
h2so4 = gas.GasSpecies(name='H2SO4', molar_mass=98.079)

# an ensemble of Aitken-mode sulfate scenarios
spec = ambrs.EnsembleSpecification(
    name='tomas_demo',
    aerosols=(so4,),
    gases=(h2so4,),
    size=aerosol.AerosolModalSizeDistribution(modes=[
        aerosol.AerosolModeDistribution(
            name='aitken', species=[so4],
            number=stats.loguniform(1e9, 1e11),          # [# m^-3]
            geom_mean_diam=stats.loguniform(3e-8, 8e-8),  # [m]
            log10_geom_std_dev=log10(1.6),
            mass_fractions=[stats.uniform(0, 1)]),
    ]),
    gas_concs=(stats.loguniform(1e6, 1e8) for _ in range(1)),  # H2SO4 [molec cm^-3]
    flux=stats.loguniform(1e-11, 1e-8),
    relative_humidity=stats.uniform(0.2, 0.6),
    temperature=stats.uniform(270, 300),
    pressure=101325,
    height=500,
)

ensemble = ambrs.sample(spec, 5)

# coagulation + condensation over one simulated hour
processes = ambrs.AerosolProcesses(coagulation=True, condensation=True)
model = ambrs.tomas_jax.AerosolModel(processes, h2so4_production=1e5)
inputs = model.create_inputs(ensemble, dt=60.0, nstep=60)

# run every scenario in process (one compiled step is reused across the ensemble)
outputs = ambrs.DirectRunner(model).run(inputs)

print()
print(f'{"scenario":>10} {"N_tot [/m3]":>16} {"dry mass conc":>16}')
for out in outputs:
    pop = out.particle_population
    print(f'{out.scenario_name:>10} {pop.get_Ntot():16.4e} '
          f'{pop.get_tot_dry_mass():16.4e}')
