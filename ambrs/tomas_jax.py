"""ambrs.tomas_jax -- data types and functions integrating TOMAS-JAX, a JAX
re-implementation of the TwO-Moment Aerosol Sectional (TOMAS) box model
(https://github.com/reflective-org/tomas-jax), into the AMBRS framework.

Unlike the other AMBRS models (PartMC, MAM4), TOMAS-JAX is a pure Python/JAX
library with no native file-based input/output and no scenario-driven command
line. It is therefore run *in process* via ambrs.runners.DirectRunner, which
calls AerosolModel.run() directly rather than shelling out to an executable.

The TOMAS-JAX package is an optional dependency: importing ambrs never requires
it. When it is not installed, this module still imports (so the rest of AMBRS
works), but constructing a tomas_jax.AerosolModel raises ImportError. The
`_TOMAS_AVAILABLE` flag reflects whether the library could be imported.
"""

from .aerosol import AerosolProcesses, AerosolModalSizePopulation, \
                     AerosolModalSizeState
from .aerosol_model import BaseAerosolModel, NotImplementedError
from .analysis import Output
from .gas import GasSpecies
from .scenario import Scenario
from .ppe import Ensemble

from dataclasses import dataclass, field
from typing import Any
import os.path
import warnings

import numpy as np

# TOMAS-JAX is an optional dependency. Its config module MUST be imported before
# any other use of JAX because it enables float64 globally
# (jax.config.update("jax_enable_x64", True)).
try:
    import tomas_jax.core.config as _tomas_config  # noqa: F401  (import first!)
    from tomas_jax.core.config import (
        xk_boundaries,
        NBINS, ICOMP, N_GAS_SPECIES,
        SRTSO4, SRTSO2, SRTNH4, SRTH2O, SRTORG1, IORG,
        MW_H2SO4, MW_SO2, AVOGADRO, PI,
    )
    from tomas_jax.solvers.condensation import make_step

    # Map AMBRS aerosol species names onto TOMAS-JAX's species layout
    # (index 0 = SO4, 1..IORG = organics, NH4, H2O). TOMAS has no slot for
    # black carbon, dust, or inorganic salts, so those are handled via
    # UNMAPPED_SPECIES below.
    AEROSOL_SPECIES_MAP = {
        'SO4': SRTSO4,
        'NH4': SRTNH4,
        'H2O': SRTH2O,
        # organics are lumped into the first organic bin (SRTORG1)
        'OC': SRTORG1, 'MSA': SRTORG1,
        'ARO1': SRTORG1, 'ARO2': SRTORG1, 'ALK1': SRTORG1, 'OLE1': SRTORG1,
        'API1': SRTORG1, 'API2': SRTORG1, 'LIM1': SRTORG1, 'LIM2': SRTORG1,
    }

    # Gas species mapped onto TOMAS's gas array Gc. NH3 is not a Gc slot; it is
    # passed to the nucleation scheme as nh3_conc (see create_input).
    GAS_SPECIES_MAP = {
        'H2SO4': SRTSO4,
        'SO2': SRTSO2,
    }

    # Molar masses [g/mol] for converting gas number concentrations to the
    # kg-per-grid-cell units TOMAS stores in Gc.
    _GAS_MOLAR_MASS = {
        'H2SO4': MW_H2SO4,
        'SO2': MW_SO2,
    }

    _TOMAS_AVAILABLE = True
except ImportError:
    _TOMAS_AVAILABLE = False

# TOMAS operator-split order; create_input emits its processes tuple in this order.
_PROCESS_ORDER = ('so2_chemistry', 'nucleation', 'coagulation', 'condensation',
                  'dilution')


def _gas_conc_to_kg_per_cell(conc_molec_cm3, boxvol, molar_mass_g_mol):
    """Convert a gas number concentration [molec cm^-3] to [kg per grid cell]."""
    return conc_molec_cm3 * boxvol * (molar_mass_g_mol / 1000.0) / AVOGADRO

# AMBRS aerosol species with no TOMAS-JAX analog. Their mass is lumped into the
# organic slot so total aerosol mass is conserved, with a warning (the speciation
# is approximate; TOMAS condensation acts on H2SO4 only regardless).
UNMAPPED_SPECIES = frozenset({'BC', 'OIN', 'NO3', 'Cl', 'Na', 'Ca', 'CO3'})


def map_species_fractions(species_names, mass_fractions):
    """Map an AMBRS mode's (species_names, mass_fractions) onto TOMAS species
indices, conserving total mass.

Returns (frac_by_index, remapped) where frac_by_index is {tomas_index: fraction}
and remapped is the tuple of species names that had no TOMAS slot and were lumped
into the organic slot.
"""
    frac_by_index = {}
    remapped = []
    for name, frac in zip(species_names, mass_fractions):
        idx = AEROSOL_SPECIES_MAP.get(name)
        if idx is None:
            idx = SRTORG1  # conserve mass by lumping into the organic slot
            remapped.append(name)
        frac_by_index[idx] = frac_by_index.get(idx, 0.0) + float(frac)
    if remapped:
        warnings.warn(
            'tomas_jax: aerosol species %s have no TOMAS-JAX slot; their mass '
            'was lumped into the organic bin (speciation is approximate).'
            % ', '.join(remapped),
            stacklevel=2,
        )
    return frac_by_index, tuple(remapped)


def lognormal_to_bins(xk, number, geom_mean_diam, log10_geom_std_dev,
                      boxvol, dens=1770.0):
    """Integrate a single log-normal mode onto the TOMAS mass-doubling grid.

Replicates tomas_jax's own initializer (tests/conftest.make_lognormal) but takes
AMBRS mode parameters: geom_mean_diam in metres and log10_geom_std_dev (base-10
log of the geometric standard deviation).

Parameters:
    * xk: bin boundary masses [kg], shape (nbins+1,)
    * number: total number concentration [# cm^-3]
    * geom_mean_diam: geometric mean diameter [m]
    * log10_geom_std_dev: log10 of the geometric standard deviation
    * boxvol: grid-cell volume [cm^3]
    * dens: particle density used to relate bin mass to diameter [kg m^-3]

Returns Nk, the number per bin [# per grid cell], shape (nbins,).
"""
    xk_np = np.asarray(xk, dtype=float)
    nbins = len(xk_np) - 1
    gsd = 10.0 ** log10_geom_std_dev
    gmd_um = geom_mean_diam * 1e6  # the integral below works in micrometres
    Nk = np.zeros(nbins)
    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (dens * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (dens * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)
        Nk[k] = ((number * boxvol)
                 / (np.sqrt(2 * PI) * Dk * np.log(gsd))
                 * np.exp(-(np.log(Dk / gmd_um) ** 2 / (2 * np.log(gsd) ** 2)))
                 * (Dh - Dl))
    return Nk


def distribute_mass(Nk, xk, frac_by_index):
    """Distribute each bin's total mass across TOMAS species indices.

Parameters:
    * Nk: number per bin [# per grid cell], shape (nbins,)
    * xk: bin boundary masses [kg], shape (nbins+1,)
    * frac_by_index: {tomas_index: mass_fraction} for this mode

Returns Mk [kg per grid cell], shape (nbins, ICOMP). Bin particle mass is taken
as the geometric-mean bin mass sqrt(xk[k]*xk[k+1]), matching tomas_jax's own
initializer.
"""
    xk_np = np.asarray(xk, dtype=float)
    nbins = len(xk_np) - 1
    Mk = np.zeros((nbins, ICOMP))
    m_mid = np.sqrt(xk_np[:-1] * xk_np[1:])  # mean particle mass per bin [kg]
    bin_mass = np.asarray(Nk, dtype=float) * m_mid  # total mass per bin [kg/cell]
    for idx, frac in frac_by_index.items():
        Mk[:, idx] += bin_mass * frac
    return Mk


@dataclass
class Input:
    """ambrs.tomas_jax.Input -- an input dataclass for the TOMAS-JAX box model.

Because TOMAS-JAX is run in process, this dataclass holds the model's in-memory
state arrays directly (Nk, Mk, Gc, xk) together with the atmospheric state,
timestepping, active processes, and precursor parameters needed to drive a run.
"""

    # in-memory TOMAS state arrays
    Nk: Any            # number per bin [# per grid cell], shape (nbins,)
    Mk: Any            # mass per bin per species [kg per grid cell], shape (nbins, ICOMP)
    Gc: Any            # gas concentrations [kg per grid cell], shape (N_GAS_SPECIES,)
    xk: Any            # bin boundary masses [kg], shape (nbins+1,)

    # atmospheric state
    temp: float        # temperature [K]
    pres: float        # pressure [Pa]
    rh: float          # relative humidity [0-1]
    boxvol: float      # grid-cell volume [cm^3]
    alpha: float       # mass accommodation coefficient [-]

    # timestepping
    dt: float          # time step [s]
    nstep: int         # number of time steps

    # active processes, in TOMAS operator-split order (subset of
    # {'so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'})
    processes: tuple[str, ...]

    # nucleation precursor inputs (ricco_dunne scheme)
    org_conc: float    # condensable organic vapor [molec/cm^3]
    nh3_conc: float    # ammonia [molec/cm^3]
    fion: float        # ion-pair production rate [ion pairs/cm^3/s]
    fn_scale: float    # nucleation rate scale factor [-]

    # H2SO4 gas production rate added to Gc[SRTSO4] each step [molec/cm^3/s]
    h2so4_production: float

    # bookkeeping (not consumed by the solver)
    aerosols: tuple = field(default_factory=tuple)       # scenario aerosol species
    dropped_species: tuple = field(default_factory=tuple)  # species with no TOMAS slot
    scenario: Any = None   # originating Scenario, used to build the Output


class AerosolModel(BaseAerosolModel):
    """ambrs.tomas_jax.AerosolModel -- an in-process AMBRS adapter for TOMAS-JAX.

Run scenarios with ambrs.runners.DirectRunner rather than PoolRunner: TOMAS-JAX
has no executable to invoke, so create_input()/run() are used directly.
"""

    def __init__(self,
                 processes: AerosolProcesses,
                 boxvol: float = 1e6,
                 alpha: float = 1.0,
                 density: float = 1770.0,
                 cond_method: str = 'ppm_jit',
                 nucl_scheme: str = 'ricco_dunne',
                 org_conc: float = 0.0,
                 nh3_conc: float = 0.0,
                 fion: float = 0.0,
                 fn_scale: float = 1.0,
                 h2so4_production: float = 0.0):
        """Construct a TOMAS-JAX adapter.

Parameters (beyond the AerosolProcesses flags):
    * boxvol: grid-cell volume [cm^3] used to scale number/mass to per-cell units
    * alpha: mass accommodation coefficient [-]
    * density: particle density used to relate bin mass to diameter [kg m^-3]
    * cond_method: TOMAS H2SO4 condensation solver ('ppm_jit','tfl_jit','ppm','tfl')
    * nucl_scheme: nucleation scheme ('ricco_dunne' or 'zhao2024')
    * org_conc, nh3_conc, fion, fn_scale: nucleation precursor inputs (molec cm^-3,
      molec cm^-3, ion pairs cm^-3 s^-1, and a dimensionless rate scale). AMBRS's
      Scenario does not carry these, so they default to model-level constants; a
      Scenario NH3 gas concentration, if present, overrides nh3_conc.
    * h2so4_production: H2SO4 gas produced each step [molec cm^-3 s^-1]
"""
        if not _TOMAS_AVAILABLE:
            raise ImportError(
                'tomas_jax is not installed; install it to use '
                'ambrs.tomas_jax.AerosolModel (see requirements.txt).')
        BaseAerosolModel.__init__(self, 'tomas-jax', processes)
        self.boxvol = boxvol
        self.alpha = alpha
        self.density = density
        self.cond_method = cond_method
        self.nucl_scheme = nucl_scheme
        self.org_conc = org_conc
        self.nh3_conc = nh3_conc
        self.fion = fion
        self.fn_scale = fn_scale
        self.h2so4_production = h2so4_production

    def _active_processes(self) -> tuple:
        """The TOMAS process names implied by self.processes, in split order."""
        active = {
            'coagulation': self.processes.coagulation,
            'condensation': self.processes.condensation,
            'nucleation': self.processes.nucleation,
        }
        return tuple(p for p in _PROCESS_ORDER if active.get(p, False))

    def create_input(self,
                     scenario: Scenario,
                     dt: float,
                     nstep: int) -> Input:
        """Convert an AMBRS scenario into a TOMAS-JAX Input.

Each log-normal mode is integrated onto the standard TOMAS mass-doubling grid and
its composition mapped onto TOMAS species indices; modes are summed. Gas number
concentrations (assumed [molec cm^-3]) are placed into Gc.
"""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if nstep <= 0:
            raise ValueError("nstep must be positive")
        if not isinstance(scenario.size, AerosolModalSizeState):
            raise TypeError('Non-modal aerosol particle size state cannot be '
                            'used to create tomas-jax input!')

        xk = np.asarray(xk_boundaries(), dtype=float)
        nbins = len(xk) - 1
        Nk = np.zeros(nbins)
        Mk = np.zeros((nbins, ICOMP))
        dropped = set()
        for mode in scenario.size.modes:
            species_names = tuple(s.name for s in mode.species)
            frac_by_index, remapped = map_species_fractions(
                species_names, mode.mass_fractions)
            dropped.update(remapped)
            # AMBRS mode.number is [# m^-3]; TOMAS bins number in [# cm^-3].
            mode_Nk = lognormal_to_bins(
                xk, mode.number * 1e-6, mode.geom_mean_diam,
                mode.log10_geom_std_dev, self.boxvol, self.density)
            Nk += mode_Nk
            Mk += distribute_mass(mode_Nk, xk, frac_by_index)

        # gas phase: map known gases into Gc; let a Scenario NH3 override nh3_conc
        Gc = np.zeros(N_GAS_SPECIES)
        nh3_conc = self.nh3_conc
        for gas, conc in zip(scenario.gases, scenario.gas_concs):
            idx = GAS_SPECIES_MAP.get(gas.name)
            if idx is not None:
                Gc[idx] += _gas_conc_to_kg_per_cell(
                    conc, self.boxvol, _GAS_MOLAR_MASS[gas.name])
            elif gas.name == 'NH3':
                nh3_conc = conc

        return Input(
            Nk=Nk, Mk=Mk, Gc=Gc, xk=xk,
            temp=scenario.temperature,
            pres=scenario.pressure,
            rh=scenario.relative_humidity,
            boxvol=self.boxvol,
            alpha=self.alpha,
            dt=dt,
            nstep=nstep,
            processes=self._active_processes(),
            org_conc=self.org_conc,
            nh3_conc=nh3_conc,
            fion=self.fion,
            fn_scale=self.fn_scale,
            h2so4_production=self.h2so4_production,
            aerosols=tuple(scenario.aerosols),
            dropped_species=tuple(sorted(dropped)),
            scenario=scenario,
        )

    def _step_fn(self, processes):
        """Build (once) and cache the compiled TOMAS step for a process set."""
        if not hasattr(self, '_step_cache'):
            self._step_cache = {}
        key = (processes, self.cond_method, self.nucl_scheme)
        if key not in self._step_cache:
            self._step_cache[key] = make_step(
                list(processes),
                cond_method=self.cond_method,
                nucl_scheme=self.nucl_scheme)
        return self._step_cache[key]

    def run(self, input, scenario_name: str = 'tomas') -> Output:
        """Run TOMAS-JAX in process and return an ambrs.analysis.Output.

Integrates the Input state for input.nstep steps (reusing one compiled step
across calls), then builds a PyParticle ParticlePopulation from the final per-bin
number/mass and a GasMixture from the final gas phase.
"""
        import jax.numpy as jnp

        Nk = jnp.asarray(input.Nk)
        Mk = jnp.asarray(input.Mk)
        Gc = jnp.asarray(input.Gc)
        xk = jnp.asarray(input.xk)

        prod_kg = _gas_conc_to_kg_per_cell(
            input.h2so4_production * input.dt, input.boxvol, MW_H2SO4)
        nuc_kwargs = {}
        if 'nucleation' in input.processes:
            nuc_kwargs = dict(org_conc=input.org_conc, nh3_conc=input.nh3_conc,
                              fion=input.fion, fn_scale=input.fn_scale)

        step = self._step_fn(input.processes) if input.processes else None
        for _ in range(input.nstep):
            if input.h2so4_production:
                Gc = Gc.at[SRTSO4].add(prod_kg)
            if step is not None:
                Nk, Mk, Gc = step(
                    Nk, Mk, Gc, xk, input.temp, input.pres, input.boxvol,
                    input.rh, input.alpha, input.dt, **nuc_kwargs)

        particle_population = self._build_population(
            np.asarray(Nk), np.asarray(Mk), np.asarray(xk), input.boxvol)
        gas_mixture = self._build_gas_mixture(
            np.asarray(Gc), input.temp, input.pres, input.boxvol)
        thermodynamics = {'T': input.temp, 'p': input.pres, 'RH': input.rh}

        return Output(
            model_name=self.name,
            scenario_name=scenario_name,
            scenario=input.scenario,
            timestep=input.nstep,
            particle_population=particle_population,
            gas_mixture=gas_mixture,
            thermodynamics=thermodynamics,
        )

    def _build_population(self, Nk, Mk, xk, boxvol):
        """Build a PyParticle ParticlePopulation from TOMAS per-bin state.

One particle is created per non-empty bin: diameter from the bin-mean mass and
particle density, composition from the bin's mass across TOMAS species (SO4, a
lumped organic, NH4, H2O), and number concentration Nk/boxvol (same units as the
scenario's mode number).
"""
        from PyParticle import ParticlePopulation, make_particle
        from PyParticle.species.registry import get_species

        # aggregate TOMAS species indices back to AMBRS dry-species names
        # (H2O is not carried as a dry species; PyParticle adds water from RH)
        candidate_names = ['SO4', 'OC', 'NH4']
        masses = np.column_stack([
            Mk[:, SRTSO4],
            Mk[:, SRTORG1:SRTNH4].sum(axis=1),  # all organics -> OC proxy
            Mk[:, SRTNH4],
        ])
        m_dry = masses.sum(axis=1)  # total dry mass per bin [kg per cell]

        # TOMAS is two-moment: the mean single-particle mass in a bin is Mk/Nk,
        # not the geometric bin-mean mass. Derive the diameter from Mk/Nk so
        # total mass is preserved; fall back to the bin geometric mass for
        # (near-)empty bins, which carry negligible number.
        m_geom = np.sqrt(xk[:-1] * xk[1:])
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_mass = np.where(Nk > 0.0, m_dry / Nk, 0.0)
        mass_for_diam = np.where(mean_mass > 0.0, mean_mass, m_geom)
        dp_mid = np.cbrt(mass_for_diam / self.density * 6.0 / np.pi)  # [m]

        # Nk/boxvol is [# cm^-3]; report as [# m^-3] to match the scenario's
        # mode number units and the other AMBRS models.
        n_conc = (Nk / boxvol) * 1e6

        totals = masses.sum(axis=0)
        keep = [i for i in range(len(candidate_names)) if i == 0 or totals[i] > 0.0]
        names = [candidate_names[i] for i in keep]
        masses = masses[:, keep]

        species_list = tuple(get_species(n, None) for n in names)
        population = ParticlePopulation(
            species=species_list, spec_masses=[], num_concs=[], ids=[],
            species_modifications={})

        part_id = 0
        for k in range(len(dp_mid)):
            if n_conc[k] <= 0.0:
                continue
            row = masses[k]
            total = row.sum()
            fracs = (row / total).tolist() if total > 0.0 \
                else [1.0] + [0.0] * (len(names) - 1)  # empty bin -> nominal SO4
            particle = make_particle(
                dp_mid[k], species_list, list(fracs),
                specdata_path=None, species_modifications={}, D_is_wet=False)
            part_id += 1
            population.set_particle(particle, part_id, float(n_conc[k]))
        return population

    def _build_gas_mixture(self, Gc, temp, pres, boxvol):
        """Build a GasMixture (SO2/H2SO4 mole ratios) from TOMAS's Gc [kg/cell]."""
        from .gas import build_gas_mixture
        R = 8.314462618                       # J mol^-1 K^-1
        boxvol_m3 = boxvol * 1e-6
        n_air = pres * boxvol_m3 / (R * temp)  # mol of air in the cell
        so2_ratio = (Gc[SRTSO2] / (MW_SO2 / 1000.0)) / n_air
        h2so4_ratio = (Gc[SRTSO4] / (MW_H2SO4 / 1000.0)) / n_air
        return build_gas_mixture(
            {'SO2': float(so2_ratio), 'H2SO4': float(h2so4_ratio),
             'units': 'ratio'})

    def write_input_files(self, input, dir, prefix) -> None:
        """Persist an Input to `<prefix>.npz` (plus a human-readable
`<prefix>_params.txt`) for provenance. TOMAS-JAX has no native input format;
this simply records the in-memory state so a run can be reproduced or inspected.
The originating Scenario is not serialized."""
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')
        np.savez(
            os.path.join(dir, prefix + '.npz'),
            Nk=input.Nk, Mk=input.Mk, Gc=input.Gc, xk=input.xk,
            temp=input.temp, pres=input.pres, rh=input.rh, boxvol=input.boxvol,
            alpha=input.alpha, dt=input.dt, nstep=input.nstep,
            processes=np.asarray(input.processes),
            org_conc=input.org_conc, nh3_conc=input.nh3_conc, fion=input.fion,
            fn_scale=input.fn_scale, h2so4_production=input.h2so4_production,
            dropped_species=np.asarray(input.dropped_species),
        )
        with open(os.path.join(dir, prefix + '_params.txt'), 'w') as f:
            f.write('# generated by ambrs.tomas_jax.AerosolModel.write_input_files\n')
            f.write(f'processes = {list(input.processes)}\n')
            for name in ('temp', 'pres', 'rh', 'boxvol', 'alpha', 'dt', 'nstep',
                         'org_conc', 'nh3_conc', 'fion', 'fn_scale',
                         'h2so4_production'):
                f.write(f'{name} = {getattr(input, name)}\n')
            f.write(f'dropped_species = {list(input.dropped_species)}\n')

    def read_input(self, dir: str, prefix: str) -> Input:
        """Reconstruct an Input from a `<prefix>.npz` written by
write_input_files. The originating Scenario is not restored (scenario=None)."""
        d = np.load(os.path.join(dir, prefix + '.npz'))
        return Input(
            Nk=d['Nk'], Mk=d['Mk'], Gc=d['Gc'], xk=d['xk'],
            temp=float(d['temp']), pres=float(d['pres']), rh=float(d['rh']),
            boxvol=float(d['boxvol']), alpha=float(d['alpha']),
            dt=float(d['dt']), nstep=int(d['nstep']),
            processes=tuple(str(p) for p in d['processes'].tolist())
                      if d['processes'].size else (),
            org_conc=float(d['org_conc']), nh3_conc=float(d['nh3_conc']),
            fion=float(d['fion']), fn_scale=float(d['fn_scale']),
            h2so4_production=float(d['h2so4_production']),
            dropped_species=tuple(str(s) for s in d['dropped_species'].tolist())
                            if d['dropped_species'].size else (),
            scenario=None,
        )

    def read_output_files(self, input, dir: str, prefix: str) -> Output:
        """Produce the Output for a scenario. As TOMAS-JAX is in-process there are
no external output files: this runs the model in process (loading the Input from
`<prefix>.npz` if `input` is None)."""
        if input is None:
            input = self.read_input(dir, prefix)
        return self.run(input, scenario_name=prefix)

    def invocation(self, exe: str, prefix: str) -> str:
        raise NotImplementedError(
            'tomas-jax is an in-process model with no executable; '
            'run it with ambrs.runners.DirectRunner instead of PoolRunner.')
