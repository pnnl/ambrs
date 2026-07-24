# MAM4-JAX in AMBRS

[MAM4-JAX](https://github.com/reflective-org/MAM4-JAX) is a JAX re-implementation
of the 4-mode MAM4 (Modal Aerosol Model) box model. Like TOMAS-JAX — and unlike
PartMC and the executable MAM4 driven by `PoolRunner` — MAM4-JAX is a pure
Python/JAX library with no command-line interface, so AMBRS runs it **in process**
through `ambrs.runners.DirectRunner`.

## Installation

MAM4-JAX is an **optional** dependency; `import ambrs` never requires it. To use
the adapter, install it (it pulls in `jax`, `jaxlib`, and `diffrax`):

```bash
pip install "git+https://github.com/reflective-org/MAM4-JAX.git@main"
```

The repository is public, so no credentials are needed.
`ambrs.mam4_jax._MAM4_JAX_AVAILABLE` reports whether the import succeeded;
constructing `ambrs.mam4_jax.AerosolModel` without it raises `ImportError`.

## Usage

```python
import ambrs, ambrs.aerosol as aerosol, ambrs.gas as gas

model = ambrs.mam4_jax.AerosolModel(
    ambrs.AerosolProcesses(coagulation=True, condensation=True, nucleation=True))

inputs = model.create_inputs(ensemble, dt=60.0, nstep=60)   # 1 h at 60 s steps
outputs = ambrs.DirectRunner(model).run(inputs)             # list[analysis.Output]
```

The scenario must carry a **4-mode** modal size state, in MAM4's order:
accumulation, Aitken, coarse, primary-carbon. `DirectRunner` runs scenarios
sequentially, reusing a single JIT-compiled step across the whole ensemble. Each
`Output` carries a PyParticle binned-lognormal `ParticlePopulation`
(`out.particle_population`) and a `GasMixture`. See `demo_mam4_jax.py` for a
complete example.

`write_input_files(input, dir, prefix)` persists an `Input` to `<prefix>.npz`
(with a readable `<prefix>_params.txt`) for provenance; `read_output_files` runs
the model in process (loading the Input from disk if none is passed).
`invocation()` is unsupported (there is no executable).

### `AerosolModel` constructor options

- `cond_backend` — MAM4-JAX condensation solver (`'diffrax'` default, `'substep'`,
  `'astem'`).
- `n_substeps` — fixed substeps for the `'substep'` backend.
- `zmid`, `pblh` — mid-layer and boundary-layer heights [m]; MAM4's PBL
  nucleation depends on them and `Scenario` does not carry them, so they are
  model-level constants (defaults `3000` and `1100`).
- `N_bins` — number of size bins used to build the output population.

## Modelling assumptions (v1)

- **Processes.** `AerosolProcesses.condensation`, `.coagulation`, and
  `.nucleation` map to MAM4's `mdo_gasaerexch`, `mdo_coag`, and `mdo_newnuc`
  (`mdo_rename` is always on). `.gas_phase_chemistry` enables MAM4-JAX's H2SO4
  other-process production term, which is injected *inside* gas-aerosol exchange —
  it therefore only has an effect when condensation is also on (a warning is
  emitted otherwise).
- **State construction.** MAM4-JAX ships no initial-condition builder, so the
  adapter constructs MAM4's tracer array `q` itself: per-mode number as
  `[# kg⁻¹ dry air]` (`N / ρ_air`) at `NUMPTR_AMODE`, and per-species mass mixing
  ratios `[kg kg⁻¹]` at `LMASSPTR_AMODE`, with the mode dry volume derived from the
  lognormal `N · (π/6) · exp(4.5 ln²σ) · D_g³`.
- **Fixed mode width.** MAM4 enforces a fixed geometric std dev per mode
  (`SIGMAG_AMODE = 1.8, 1.6, 1.8, 1.6`). The mode volume is built with that fixed
  width so MAM4's `calcsize` reproduces the input geometric mean diameter as a
  fixed point — **number and mean diameter are preserved on step 0, but the
  scenario's per-mode GSD is discarded** (a warning is emitted on mismatch).
- **Composition mapping.** AMBRS species map onto MAM4's per-mode species set:
  `SO4→sulfate`, `OC→p-organic`, `MSA` and the SOA precursors
  (`ARO*, ALK1, OLE1, API*, LIM*`) `→ s-organic`, `BC→black-c`, `Na`/`Cl`
  `→ seasalt`, `OIN` (and `Ca`/`CO3`) `→ dust`. **This MAM4-MOM build has no
  ammonium or nitrate slots**, so `NH4`/`NO3` are lumped into sulfate (an
  inorganic proxy) to conserve mass, with a warning. Any species whose type has
  no slot in a given mode is lumped into that mode's sulfate (or organic)
  slot, with a warning — the speciation is approximate.
- **Gas units.** Gas concentrations in the `Scenario` are treated as MAM4 **mass
  mixing ratios** `[kg gas / kg dry air]` (matching `ambrs.mam4`), not number
  concentrations. `H2SO4` is placed into `q` and evolves; `SO2` is carried but is
  **inert** in this build and is echoed unchanged into the output gas mixture.
- **Clear-sky only.** MAM4-JAX's cloudy microphysics path is unimplemented, so the
  cloud fraction `cldn` is hard-coded to 0.

## Continuous integration

The AMBRS test suite guards all MAM4-JAX tests with `skipUnless`, so they are
skipped when the library is absent and CI stays green. Because MAM4-JAX is public,
running its tests in CI only requires adding the `pip install` above to the
workflow (no deploy key needed).
