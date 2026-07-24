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
        MW_H2SO4, AVOGADRO, PI,
    )
    from tomas_jax.solvers.condensation import make_step
    _TOMAS_AVAILABLE = True
except ImportError:
    _TOMAS_AVAILABLE = False


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


class AerosolModel(BaseAerosolModel):
    """ambrs.tomas_jax.AerosolModel -- an in-process AMBRS adapter for TOMAS-JAX.

Run scenarios with ambrs.runners.DirectRunner rather than PoolRunner: TOMAS-JAX
has no executable to invoke, so create_input()/run() are used directly.
"""

    def __init__(self,
                 processes: AerosolProcesses):
        if not _TOMAS_AVAILABLE:
            raise ImportError(
                'tomas_jax is not installed; install it to use '
                'ambrs.tomas_jax.AerosolModel (see requirements.txt).')
        BaseAerosolModel.__init__(self, 'tomas-jax', processes)

    def create_input(self,
                     scenario: Scenario,
                     dt: float,
                     nstep: int) -> Input:
        raise NotImplementedError('tomas_jax.AerosolModel.create_input not yet implemented!')

    def run(self, input) -> Output:
        raise NotImplementedError('tomas_jax.AerosolModel.run not yet implemented!')

    def write_input_files(self, input, dir, prefix) -> None:
        raise NotImplementedError('tomas_jax.AerosolModel.write_input_files not yet implemented!')

    def read_output_files(self, input, dir: str, prefix: str) -> Output:
        raise NotImplementedError('tomas_jax.AerosolModel.read_output_files not yet implemented!')

    def invocation(self, exe: str, prefix: str) -> str:
        raise NotImplementedError(
            'tomas-jax is an in-process model with no executable; '
            'run it with ambrs.runners.DirectRunner instead of PoolRunner.')
