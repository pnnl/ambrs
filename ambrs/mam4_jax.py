"""ambrs.mam4_jax -- data types and functions integrating MAM4-JAX, a JAX
re-implementation of the 4-mode MAM4 (Modal Aerosol Model) box model
(https://github.com/reflective-org/MAM4-JAX), into the AMBRS framework.

Like TOMAS-JAX (and unlike PartMC or the executable MAM4), MAM4-JAX is a pure
Python/JAX library with no native file-based input/output and no scenario-driven
command line. It is therefore run *in process* via ambrs.runners.DirectRunner,
which calls AerosolModel.run() directly rather than shelling out to an executable.

MAM4-JAX is an optional dependency: importing ambrs never requires it. When it is
not installed this module still imports (so the rest of AMBRS works), but
constructing a mam4_jax.AerosolModel raises ImportError. The `_MAM4_JAX_AVAILABLE`
flag reflects whether the library could be imported.

Because MAM4 is a fixed-structure, 4-mode modal model, this adapter reuses the
AMBRS->MAM4 species/gas conventions of ambrs.mam4 and builds an ambrs.analysis
Output the same way (a PyParticle binned-lognormal population + a GasMixture).
"""

from .aerosol import AerosolProcesses, AerosolModalSizeState
from .aerosol_model import BaseAerosolModel, NotImplementedError
from .analysis import Output
from .gas import GasSpecies, build_gas_mixture
from .scenario import Scenario

from dataclasses import dataclass, field
from typing import Any
import os.path
import warnings

import numpy as np

# MAM4-JAX is an optional dependency. `import mam4_jax` MUST run before any other
# use of JAX: importing it enables float64 globally
# (jax.config.update("jax_enable_x64", True)).
try:
    import mam4_jax as _mam4_jax  # noqa: F401  (import first!)
    from mam4_jax import data as _data
    from mam4_jax.processes.amicphys import (
        amicphys as _amicphys,
        configure_condensation as _configure_condensation,
        configure_gas_netprod as _configure_gas_netprod,
    )
    from mam4_jax.processes.calcsize import calcsize as _calcsize
    from mam4_jax.processes.wateruptake import wateruptake as _wateruptake

    # --- MAM4 structural constants, read from the library ---------------------
    PCNST = int(_data.PCNST)                                # tracers in q
    NMODES = int(_data.NTOT_AMODE)                          # 4
    NSPEC_AMODE = np.asarray(_data.NSPEC_AMODE, dtype=int)  # species per mode
    NUMPTR_AMODE = np.asarray(_data.NUMPTR_AMODE, dtype=int)      # q number idx
    LMASSPTR_AMODE = np.asarray(_data.LMASSPTR_AMODE, dtype=int)  # q mass idx (mode,slot)
    LSPECTYPE_AMODE = np.asarray(_data.LSPECTYPE_AMODE, dtype=int)  # slot -> type
    SPECDENS_AMODE = np.asarray(_data.SPECDENS_AMODE, dtype=float)  # kg/m^3 per type
    SIGMAG_AMODE = np.asarray(_data.SIGMAG_AMODE, dtype=float)      # fixed mode widths
    DUMFAC_AMODE = np.asarray(_data.DUMFAC_AMODE, dtype=float)  # (pi/6) exp(4.5 ln^2 sigma)
    # gas pcnst indices: LMAP_GAS = [SOAG, H2SO4]
    _SOAG_IDX = int(_data.LMAP_GAS[0])
    _H2SO4_IDX = int(_data.LMAP_GAS[1])
    _SO2_IDX = 7  # SO2 lives at pcnst 7; inert in this MAM4-MOM build

    # MAM4 species-type indices into SPECNAME_AMODE
    # ("sulfate","ammonium","nitrate","p-organic","s-organic","black-c",
    #  "seasalt","dust","m-organic")
    _T_SULFATE, _T_AMMONIUM, _T_NITRATE = 0, 1, 2
    _T_POM, _T_SOA, _T_BC, _T_NCL, _T_DUST, _T_MOM = 3, 4, 5, 6, 7, 8

    # AMBRS aerosol species name -> MAM4 species-type index. Ammonium (NH4) and
    # nitrate (NO3) map to their true MAM4 types, but this MAM4-MOM configuration
    # has NO ammonium or nitrate slot in any mode, so they fall through to the
    # mass-conserving fallback lump (into sulfate) with a warning. Organics all
    # lump into the secondary-organic type; sea-salt covers Na and Cl; dust
    # covers OIN (and Ca/CO3 as a mineral proxy).
    AEROSOL_SPECIES_MAP = {
        'SO4': _T_SULFATE,
        'NH4': _T_AMMONIUM,  # no ammonium slot anywhere -> lumped (warned)
        'NO3': _T_NITRATE,   # no nitrate slot anywhere  -> lumped (warned)
        'OC': _T_POM,
        'BC': _T_BC,
        'OIN': _T_DUST, 'Ca': _T_DUST, 'CO3': _T_DUST,
        'Na': _T_NCL, 'Cl': _T_NCL,
        # secondary organics
        'MSA': _T_SOA, 'ARO1': _T_SOA, 'ARO2': _T_SOA, 'ALK1': _T_SOA,
        'OLE1': _T_SOA, 'API1': _T_SOA, 'API2': _T_SOA, 'LIM1': _T_SOA,
        'LIM2': _T_SOA,
    }

    # Reverse map (MAM4 type -> AMBRS name) for building the output population.
    # s-organic reports as 'MSA' and sea-salt as 'Na', matching ambrs.mam4's
    # conventions; m-organic (if ever nonzero) reports as 'OC' with a warning.
    TYPE_TO_AMBRS = {
        _T_SULFATE: 'SO4', _T_POM: 'OC', _T_SOA: 'MSA', _T_BC: 'BC',
        _T_NCL: 'Na', _T_DUST: 'OIN', _T_MOM: 'OC',
    }

    # Per-mode {type index -> q pcnst index} and {pcnst index -> type index}.
    SLOT_OF_TYPE = []
    PCNST_TO_TYPE = {}
    for _m in range(NMODES):
        _slot_of_type = {}
        for _slot in range(int(NSPEC_AMODE[_m])):
            _t = int(LSPECTYPE_AMODE[_m, _slot])
            _pcnst = int(LMASSPTR_AMODE[_m, _slot])
            _slot_of_type[_t] = _pcnst
            PCNST_TO_TYPE[_pcnst] = _t
        SLOT_OF_TYPE.append(_slot_of_type)

    _MAM4_JAX_AVAILABLE = True
except (ImportError, OSError):
    # ImportError: mam4_jax not installed. OSError/FileNotFoundError: mam4_jax is
    # installed but a data file it loads at import (e.g. _coag_tables.npz) is
    # missing -- a broken/partial install that must not take down `import ambrs`.
    _MAM4_JAX_AVAILABLE = False

# universal gas constant [J mol^-1 K^-1] and dry-air molar mass [kg/mol]
_R = 8.314462618
_MW_DRY_AIR = 28.966e-3


def _air_density(temperature, pressure):
    """Dry-air mass density [kg m^-3] from temperature [K] and pressure [Pa]."""
    return pressure * _MW_DRY_AIR / (_R * temperature)


def _fallback_pcnst(mode_index):
    """A q pcnst index to lump into when a species' MAM4 type has no slot in a
mode. Prefer sulfate, then primary organic, then m-organic, else the first slot."""
    slot_of_type = SLOT_OF_TYPE[mode_index]
    for t in (_T_SULFATE, _T_POM, _T_MOM):
        if t in slot_of_type:
            return slot_of_type[t]
    return int(LMASSPTR_AMODE[mode_index, 0])


def map_species_fractions(mode_index, species_names, mass_fractions):
    """Map one mode's (species_names, mass_fractions) onto MAM4 q pcnst indices,
conserving mass and normalizing the placed fractions to sum to 1.

Returns (frac_by_pcnst, remapped) where frac_by_pcnst is {pcnst_index: fraction}
and remapped is the tuple of AMBRS species names that had no native MAM4 slot in
this mode and were lumped elsewhere (aerosol water is dropped, not lumped)."""
    frac_by_pcnst = {}
    remapped = []
    slot_of_type = SLOT_OF_TYPE[mode_index]
    for name, frac in zip(species_names, mass_fractions):
        frac = float(frac)
        if frac == 0.0 or name == 'H2O':
            continue
        tidx = AEROSOL_SPECIES_MAP.get(name)
        if tidx is not None and tidx in slot_of_type:
            pcnst = slot_of_type[tidx]
        else:
            # no native slot for this species/type in this mode: lump to conserve
            pcnst = _fallback_pcnst(mode_index)
            remapped.append(name)
        frac_by_pcnst[pcnst] = frac_by_pcnst.get(pcnst, 0.0) + frac
    total = sum(frac_by_pcnst.values())
    if total > 0.0:
        frac_by_pcnst = {p: f / total for p, f in frac_by_pcnst.items()}
    if remapped:
        warnings.warn(
            'mam4_jax: aerosol species %s have no native MAM4 slot in mode %d; '
            'their mass was lumped into another slot (speciation is approximate).'
            % (', '.join(remapped), mode_index),
            stacklevel=2)
    return frac_by_pcnst, tuple(remapped)


def lognormal_to_q(modes, temperature, pressure):
    """Build MAM4 initial state arrays from AMBRS modal lognormal state.

Given the (exactly 4) AMBRS modes and the atmospheric temperature [K] and
pressure [Pa], returns a dict of float64 numpy arrays shaped for the MAM4-JAX
state dict: q (1,1,PCNST), dgncur_a (1,1,NMODES), and the auxiliary fields
(qqcw, dgncur_awet, qaerwat, wetdens) seeded so that MAM4-JAX's calcsize +
wateruptake (which run before amicphys) make them self-consistent.

MAM4 number tracers are stored as [# per kg dry air] and mass tracers as
[kg species / kg dry air]. MAM4 also enforces a *fixed* geometric std dev per
mode (SIGMAG_AMODE); the mode volume is therefore built with that fixed width so
that calcsize reproduces the input geometric mean diameter as a fixed point,
preserving both the mode number and mean diameter on step 0.
"""
    rho_air = _air_density(temperature, pressure)

    q = np.zeros((1, 1, PCNST), dtype=float)
    dgncur_a = np.zeros((1, 1, NMODES), dtype=float)
    dropped = []
    for m, mode in enumerate(modes):
        gmd = float(mode.geom_mean_diam)
        gsd_in = 10.0 ** float(mode.log10_geom_std_dev)
        if abs(gsd_in - SIGMAG_AMODE[m]) > 0.02 * SIGMAG_AMODE[m]:
            warnings.warn(
                'mam4_jax: mode %d geometric std dev %.3f differs from MAM4\'s '
                'fixed sigma %.3f; MAM4 uses the fixed value (mode width is not '
                'a free parameter).' % (m, gsd_in, SIGMAG_AMODE[m]),
                stacklevel=2)

        # number: [# m^-3] -> [# kg^-1 dry air]
        q[0, 0, NUMPTR_AMODE[m]] = mode.number / rho_air
        dgncur_a[0, 0, m] = gmd

        species_names = tuple(s.name for s in mode.species)
        frac_by_pcnst, remapped = map_species_fractions(
            m, species_names, mode.mass_fractions)
        dropped.extend(remapped)
        if not frac_by_pcnst:
            continue

        # total dry mode volume from the fixed-width lognormal [m^3 aerosol / m^3 air]
        v_mode = mode.number * DUMFAC_AMODE[m] * gmd ** 3
        # V = M_total * sum(f_i / rho_i)  ->  M_total [kg / m^3 air]
        denom = sum(f / SPECDENS_AMODE[PCNST_TO_TYPE[p]]
                    for p, f in frac_by_pcnst.items())
        m_total = v_mode / denom if denom > 0.0 else 0.0
        for pcnst, frac in frac_by_pcnst.items():
            q[0, 0, pcnst] = frac * m_total / rho_air  # -> mass mixing ratio [kg/kg]

    return {
        'q': q,
        'dgncur_a': dgncur_a,
        'qqcw': np.zeros((1, 1, PCNST), dtype=float),
        'dgncur_awet': dgncur_a.copy(),
        'qaerwat': np.zeros((1, 1, NMODES), dtype=float),
        'wetdens': np.zeros((1, 1, NMODES), dtype=float),
    }, tuple(sorted(set(dropped)))


@dataclass
class Input:
    """ambrs.mam4_jax.Input -- an input dataclass for the MAM4-JAX box model.

Because MAM4-JAX is run in process, this dataclass holds the model's in-memory
state arrays directly (q, dgncur_a and the wet/auxiliary fields) together with
the atmospheric state, timestepping, and active-process flags needed to drive a
run."""

    # in-memory MAM4 state arrays (numpy; converted to JAX in run())
    q: Any               # tracer array [various], shape (1,1,PCNST)
    dgncur_a: Any        # dry modal diameter [m], shape (1,1,NMODES)
    qqcw: Any            # cloud-borne tracer mirror, shape (1,1,PCNST)
    dgncur_awet: Any     # wet modal diameter [m], shape (1,1,NMODES)
    qaerwat: Any         # per-mode aerosol water, shape (1,1,NMODES)
    wetdens: Any         # per-mode wet density [kg/m^3], shape (1,1,NMODES)

    # atmospheric state
    temp: float          # temperature [K]
    press: float         # pressure [Pa]
    rh: float            # relative humidity [0-1]
    zmid: float          # mid-layer height [m] (PBL nucleation)
    pblh: float          # boundary-layer height [m]

    # timestepping
    dt: float            # time step [s]
    nstep: int           # number of time steps

    # MAM4 process toggles (mdo_*) and the H2SO4 gas-production rate [mol/mol/s]
    mdo_gasaerexch: int
    mdo_rename: int
    mdo_newnuc: int
    mdo_coag: int
    gaschem_rate: float

    # bookkeeping (not consumed by the solver)
    so2_mmr: float = 0.0            # input SO2 mixing ratio, echoed to output
    aerosols: tuple = field(default_factory=tuple)
    dropped_species: tuple = field(default_factory=tuple)
    scenario: Any = None


class AerosolModel(BaseAerosolModel):
    """ambrs.mam4_jax.AerosolModel -- an in-process AMBRS adapter for MAM4-JAX.

Run scenarios with ambrs.runners.DirectRunner rather than PoolRunner: MAM4-JAX
has no executable to invoke, so create_input()/run() are used directly."""

    def __init__(self,
                 processes: AerosolProcesses,
                 cond_backend: str = 'diffrax',
                 n_substeps: int = None,
                 zmid: float = 3000.0,
                 pblh: float = 1100.0,
                 N_bins: int = 1000):
        """Construct a MAM4-JAX adapter.

Parameters (beyond the AerosolProcesses flags):
    * cond_backend: MAM4-JAX condensation solver ('diffrax','substep','astem')
    * n_substeps: fixed substeps for the 'substep' backend (ignored otherwise)
    * zmid, pblh: mid-layer and boundary-layer heights [m]; MAM4's PBL
      nucleation depends on these and AMBRS's Scenario does not carry them, so
      they are model-level constants.
    * N_bins: number of size bins used to build the output particle population
"""
        if not _MAM4_JAX_AVAILABLE:
            raise ImportError(
                'mam4_jax is not installed; install it to use '
                'ambrs.mam4_jax.AerosolModel (see requirements.txt).')
        BaseAerosolModel.__init__(self, 'mam4-jax', processes)
        self.zmid = zmid
        self.pblh = pblh
        self.N_bins = N_bins
        _configure_condensation(backend=cond_backend, n_substeps=n_substeps)

    def _mdo(self) -> dict:
        """The MAM4 mdo_* flags implied by self.processes. mdo_rename is always
on (mirrors ambrs.mam4). gas_phase_chemistry is not an mdo flag; it is handled
via the H2SO4 production rate (see _gaschem_rate)."""
        return {
            'mdo_gasaerexch': 1 if self.processes.condensation else 0,
            'mdo_rename': 1,
            'mdo_newnuc': 1 if self.processes.nucleation else 0,
            'mdo_coag': 1 if self.processes.coagulation else 0,
        }

    def _gaschem_rate(self) -> float:
        """H2SO4 other-process production rate [mol/mol/s]. MAM4-JAX injects it
inside gas-aerosol exchange, so it only has an effect when condensation is on."""
        if self.processes.gas_phase_chemistry:
            if not self.processes.condensation:
                warnings.warn(
                    'mam4_jax: gas_phase_chemistry has no effect unless '
                    'condensation is enabled (MAM4-JAX injects H2SO4 production '
                    'inside gas-aerosol exchange).', stacklevel=2)
            return 1e-16
        return 0.0

    def create_input(self,
                     scenario: Scenario,
                     dt: float,
                     nstep: int) -> Input:
        """Convert an AMBRS scenario into a MAM4-JAX Input.

The scenario must carry a 4-mode modal size state (accumulation, Aitken, coarse,
primary-carbon, in that order). Gas concentrations are treated as MAM4 mixing
ratios [kg gas / kg dry air], matching ambrs.mam4."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if nstep <= 0:
            raise ValueError("nstep must be positive")
        if not isinstance(scenario.size, AerosolModalSizeState):
            raise TypeError('Non-modal aerosol particle size state cannot be '
                            'used to create mam4-jax input!')
        if len(scenario.size.modes) != NMODES:
            raise TypeError(
                f'{len(scenario.size.modes)}-mode aerosol particle size state '
                f'cannot be used to create mam4-jax input (need {NMODES}).')

        state, dropped = lognormal_to_q(
            scenario.size.modes, scenario.temperature, scenario.pressure)

        # gas phase: place known gas mixing ratios into q (SO2 is inert here)
        so2_mmr = 0.0
        iso2 = GasSpecies.find(scenario.gases, 'SO2')
        if iso2 != -1:
            so2_mmr = float(scenario.gas_concs[iso2])
            state['q'][0, 0, _SO2_IDX] = so2_mmr
        ih2so4 = GasSpecies.find(scenario.gases, 'H2SO4')
        if ih2so4 != -1:
            state['q'][0, 0, _H2SO4_IDX] = float(scenario.gas_concs[ih2so4])

        mdo = self._mdo()
        return Input(
            q=state['q'], dgncur_a=state['dgncur_a'], qqcw=state['qqcw'],
            dgncur_awet=state['dgncur_awet'], qaerwat=state['qaerwat'],
            wetdens=state['wetdens'],
            temp=scenario.temperature,
            press=scenario.pressure,
            rh=scenario.relative_humidity,
            zmid=self.zmid,
            pblh=self.pblh,
            dt=dt,
            nstep=nstep,
            mdo_gasaerexch=mdo['mdo_gasaerexch'],
            mdo_rename=mdo['mdo_rename'],
            mdo_newnuc=mdo['mdo_newnuc'],
            mdo_coag=mdo['mdo_coag'],
            gaschem_rate=self._gaschem_rate(),
            so2_mmr=so2_mmr,
            aerosols=tuple(scenario.aerosols),
            dropped_species=dropped,
            scenario=scenario,
        )

    def _step_fn(self, mdo, gaschem_rate):
        """Build (once) and cache the compiled MAM4 step for an mdo/gaschem set.

The step is calcsize -> wateruptake -> amicphys(mdo). The mdo_* ints are closed
over the jitted function, so they are static (baked into the compiled binary);
gaschem_rate is part of the cache key and set via configure_gas_netprod before
the step is first traced (it is read at trace time)."""
        if not hasattr(self, '_step_cache'):
            self._step_cache = {}
        key = (mdo['mdo_gasaerexch'], mdo['mdo_rename'],
               mdo['mdo_newnuc'], mdo['mdo_coag'], gaschem_rate)
        if key not in self._step_cache:
            import jax
            _configure_gas_netprod(h2so4=gaschem_rate)

            @jax.jit
            def step(state):
                state = _calcsize(state)
                state = _wateruptake(state)
                state = _amicphys(state, **mdo)
                return state

            self._step_cache[key] = step
        return self._step_cache[key]

    def run(self, input, scenario_name: str = 'mam4-jax') -> Output:
        """Run MAM4-JAX in process and return an ambrs.analysis.Output.

Integrates the Input state for input.nstep steps (reusing one compiled step
across calls), then builds a PyParticle binned-lognormal population and a
GasMixture from the final state."""
        import jax.numpy as jnp

        state = {
            'q': jnp.asarray(input.q),
            'qqcw': jnp.asarray(input.qqcw),
            'dgncur_a': jnp.asarray(input.dgncur_a),
            'dgncur_awet': jnp.asarray(input.dgncur_awet),
            'qaerwat': jnp.asarray(input.qaerwat),
            'wetdens': jnp.asarray(input.wetdens),
            't': jnp.full((1, 1), input.temp),
            'pmid': jnp.full((1, 1), input.press),
            'cldn': jnp.zeros((1, 1)),           # cloudy path unimplemented
            'zmid': jnp.full((1, 1), input.zmid),
            'pblh': jnp.full((1, 1), input.pblh),
            'relhum': jnp.full((1, 1), input.rh),
            'deltat': jnp.asarray(float(input.dt)),
        }

        mdo = {
            'mdo_gasaerexch': input.mdo_gasaerexch,
            'mdo_rename': input.mdo_rename,
            'mdo_newnuc': input.mdo_newnuc,
            'mdo_coag': input.mdo_coag,
        }
        step = self._step_fn(mdo, input.gaschem_rate)
        for _ in range(input.nstep):
            state = step(state)

        q = np.asarray(state['q']).reshape(-1, PCNST)[0]
        dgncur_a = np.asarray(state['dgncur_a']).reshape(-1, NMODES)[0]

        particle_population = self._build_population(
            q, dgncur_a, input.temp, input.press)
        gas_mixture = build_gas_mixture({
            'H2SO4': float(q[_H2SO4_IDX]),
            'SO2': float(input.so2_mmr),
            'units': 'kg_per_kg',
        })
        thermodynamics = {'T': input.temp, 'p': input.press, 'RH': input.rh}

        return Output(
            model_name=self.name,
            scenario_name=scenario_name,
            scenario=input.scenario,
            timestep=input.nstep,
            particle_population=particle_population,
            gas_mixture=gas_mixture,
            thermodynamics=thermodynamics,
        )

    def _build_population(self, q, dgncur_a, temperature, pressure):
        """Build a PyParticle binned-lognormal population from MAM4's final state.

One lognormal mode is emitted per MAM4 mode: number from q's number tracer
(converted [# kg^-1] -> [# m^-3]), geometric mean diameter from dgncur_a, the
fixed MAM4 geometric std dev, and per-species dry-mass fractions from q."""
        from PyParticle import build_population

        rho_air = _air_density(temperature, pressure)
        Ns, GMDs, GSDs, names, fracs = [], [], [], [], []
        lumped_mom = False
        for m in range(NMODES):
            Ns.append(float(q[NUMPTR_AMODE[m]] * rho_air))   # [# m^-3]
            GMDs.append(float(dgncur_a[m]))
            GSDs.append(float(SIGMAG_AMODE[m]))
            mode_names, mode_mass = [], []
            for slot in range(int(NSPEC_AMODE[m])):
                w = float(q[LMASSPTR_AMODE[m, slot]])
                if w <= 0.0:
                    continue
                t = int(LSPECTYPE_AMODE[m, slot])
                if t == _T_MOM:
                    lumped_mom = True
                mode_names.append(TYPE_TO_AMBRS[t])
                mode_mass.append(w)
            total = sum(mode_mass)
            if total > 0.0:
                names.append(mode_names)
                fracs.append([w / total for w in mode_mass])
            else:
                names.append(['SO4'])   # empty mode -> nominal sulfate
                fracs.append([1.0])
        if lumped_mom:
            warnings.warn(
                'mam4_jax: nonzero m-organic mass reported as OC in the output '
                'population.', stacklevel=2)

        cfg = {
            'type': 'binned_lognormals',
            'D_min': 1e-9, 'D_max': 1e-4, 'N_bins': self.N_bins,
            'N': Ns, 'GMD': GMDs, 'GSD': GSDs,
            'aero_spec_names': names, 'aero_spec_fracs': fracs,
        }
        return build_population(cfg)

    def write_input_files(self, input, dir, prefix) -> None:
        """Persist an Input to `<prefix>.npz` (plus a human-readable
`<prefix>_params.txt`) for provenance. MAM4-JAX has no native input format; this
records the in-memory state so a run can be reproduced or inspected. The
originating Scenario is not serialized."""
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')
        np.savez(
            os.path.join(dir, prefix + '.npz'),
            q=input.q, dgncur_a=input.dgncur_a, qqcw=input.qqcw,
            dgncur_awet=input.dgncur_awet, qaerwat=input.qaerwat,
            wetdens=input.wetdens,
            temp=input.temp, press=input.press, rh=input.rh,
            zmid=input.zmid, pblh=input.pblh, dt=input.dt, nstep=input.nstep,
            mdo_gasaerexch=input.mdo_gasaerexch, mdo_rename=input.mdo_rename,
            mdo_newnuc=input.mdo_newnuc, mdo_coag=input.mdo_coag,
            gaschem_rate=input.gaschem_rate, so2_mmr=input.so2_mmr,
            dropped_species=np.asarray(input.dropped_species),
        )
        with open(os.path.join(dir, prefix + '_params.txt'), 'w') as f:
            f.write('# generated by ambrs.mam4_jax.AerosolModel.write_input_files\n')
            for name in ('temp', 'press', 'rh', 'zmid', 'pblh', 'dt', 'nstep',
                         'mdo_gasaerexch', 'mdo_rename', 'mdo_newnuc', 'mdo_coag',
                         'gaschem_rate', 'so2_mmr'):
                f.write(f'{name} = {getattr(input, name)}\n')
            f.write(f'dropped_species = {list(input.dropped_species)}\n')

    def read_input(self, dir: str, prefix: str) -> Input:
        """Reconstruct an Input from a `<prefix>.npz` written by
write_input_files. The originating Scenario is not restored (scenario=None)."""
        d = np.load(os.path.join(dir, prefix + '.npz'))
        return Input(
            q=d['q'], dgncur_a=d['dgncur_a'], qqcw=d['qqcw'],
            dgncur_awet=d['dgncur_awet'], qaerwat=d['qaerwat'],
            wetdens=d['wetdens'],
            temp=float(d['temp']), press=float(d['press']), rh=float(d['rh']),
            zmid=float(d['zmid']), pblh=float(d['pblh']),
            dt=float(d['dt']), nstep=int(d['nstep']),
            mdo_gasaerexch=int(d['mdo_gasaerexch']),
            mdo_rename=int(d['mdo_rename']),
            mdo_newnuc=int(d['mdo_newnuc']), mdo_coag=int(d['mdo_coag']),
            gaschem_rate=float(d['gaschem_rate']), so2_mmr=float(d['so2_mmr']),
            dropped_species=tuple(str(s) for s in d['dropped_species'].tolist())
                            if d['dropped_species'].size else (),
            scenario=None,
        )

    def read_output_files(self, input, dir: str, prefix: str) -> Output:
        """Produce the Output for a scenario. As MAM4-JAX is in-process there are
no external output files: this runs the model in process (loading the Input from
`<prefix>.npz` if `input` is None)."""
        if input is None:
            input = self.read_input(dir, prefix)
        return self.run(input, scenario_name=prefix)

    def invocation(self, exe: str, prefix: str) -> str:
        raise NotImplementedError(
            'mam4-jax is an in-process model with no executable; '
            'run it with ambrs.runners.DirectRunner instead of PoolRunner.')
