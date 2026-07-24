# TOMAS-JAX in AMBRS

[TOMAS-JAX](https://github.com/reflective-org/tomas-jax) is a JAX re-implementation
of the TwO-Moment Aerosol Sectional (TOMAS) box model. Unlike PartMC and MAM4 —
external executables driven by `PoolRunner` — TOMAS-JAX is a pure Python/JAX
library with no command-line interface, so AMBRS runs it **in process** through a
dedicated runner.

## Installation

TOMAS-JAX is an **optional** dependency; `import ambrs` never requires it. To use
the adapter, install it (it pulls in `jax`, `jaxlib`, and `diffrax`):

```bash
pip install "git+ssh://git@github.com/reflective-org/tomas-jax.git@dev"
```

`ambrs.tomas_jax._TOMAS_AVAILABLE` reports whether the import succeeded;
constructing `ambrs.tomas_jax.AerosolModel` without it raises `ImportError`.

### PyParticle dependency caveat

This branch's `ambrs.analysis` imports the population library as `PyParticle`, but
the current `pnnl/PyParticle` package installs its module as **`part2pop`** (the
project was renamed 2025-10-15). Until AMBRS is migrated to `part2pop` (or an
older PyParticle is pinned), alias the module during development, e.g. a
`sitecustomize.py` on the path containing:

```python
import sys, part2pop
sys.modules.setdefault("PyParticle", part2pop)
```

Note also that `part2pop.analysis` exposes `build_variable` rather than the
`compute_variable` that `ambrs.analysis.Output.compute_variable` expects; this
API drift affects size-distribution post-processing for **all** models on this
branch, not just TOMAS-JAX.

## Usage

```python
import ambrs, ambrs.aerosol as aerosol, ambrs.gas as gas

so4 = aerosol.AerosolSpecies(name='SO4', molar_mass=97.071, density=1770,
                             hygroscopicity=0.507)
h2so4 = gas.GasSpecies(name='H2SO4', molar_mass=98.079)

model = ambrs.tomas_jax.AerosolModel(
    ambrs.AerosolProcesses(coagulation=True, condensation=True, nucleation=True))

inputs = model.create_inputs(ensemble, dt=60.0, nstep=60)   # 1 h at 60 s steps
outputs = ambrs.DirectRunner(model).run(inputs)             # list[analysis.Output]
```

`DirectRunner` runs scenarios sequentially, reusing a single JIT-compiled step
across the whole ensemble. Each `Output` carries a PyParticle `ParticlePopulation`
(`out.particle_population`) and a `GasMixture`. See `demo_tomas_jax.py` for a
complete example.

`write_input_files(input, dir, prefix)` persists an `Input` to `<prefix>.npz`
(with a readable `<prefix>_params.txt`) for provenance; `read_output_files` runs
the model in process (loading the Input from disk if none is passed).
`invocation()` is unsupported (there is no executable).

## Modelling assumptions (v1)

- **Processes.** `AerosolProcesses.coagulation`, `.condensation` (H2SO4 → SO4
  only), and `.nucleation` (the `ricco_dunne` scheme) are wired, applied in
  TOMAS operator-split order.
- **Nucleation precursors.** `Scenario` does not carry `org_conc`, `nh3_conc`, or
  `fion`; they are model-level constructor defaults (0 by default). A `Scenario`
  `NH3` gas concentration, if present, overrides `nh3_conc`. An optional
  `h2so4_production` [molec cm⁻³ s⁻¹] adds H2SO4 gas each step.
- **Composition mapping.** AMBRS species map onto TOMAS's `SO4 / organics / NH4 /
  H2O` layout: `SO4→SO4`, `NH4→NH4`, `H2O→H2O`, and all organics
  (`OC, MSA, ARO*, ALK1, OLE1, API*, LIM*`) into a single organic bin. Species
  with no TOMAS slot (`BC, OIN, NO3, Cl, Na, Ca, CO3`) have their **mass lumped
  into the organic bin to conserve total mass**, with a warning — the speciation
  is approximate.
- **Units.** `Scenario` mode number is `[# m⁻³]`; gas concentrations are assumed
  `[molec cm⁻³]`. TOMAS is two-moment, so per-bin particle diameters in the output
  population are derived from the actual `Mk/Nk` mean mass, which conserves dry
  mass under coagulation to machine precision.

## Continuous integration

The AMBRS test suite guards all TOMAS-JAX tests with `skipUnless`, so they are
skipped when the library is absent and CI stays green. Running the TOMAS-JAX
tests in CI requires making the private `tomas-jax` (and PyParticle) repositories
installable there, e.g. via a deploy key or token.
