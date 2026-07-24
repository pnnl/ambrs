# Running the executable box models (PartMC, MAM4)

AMBRS supports two kinds of box model:

* **In-process models** are Python libraries that AMBRS imports and steps
  directly. They need no build step and no executable.
* **Executable models** — **PartMC** and **MAM4** — are compiled Fortran/C
  programs. AMBRS writes their native input files, runs them as subprocesses,
  and parses their output. They must be **built separately** before AMBRS can
  run them.

This document covers the executable models: how to build them and how to drive
them from `ambrs`.

> **Note on verification.** The `ambrs`-side usage below is taken directly from
> the adapters in `ambrs/partmc.py` and `ambrs/mam4.py`. The build steps are
> taken from [ambuilder](https://github.com/AMBRS-project/ambuilder)'s own
> documentation and have **not** been re-verified here — treat exact compiler and
> library requirements as ambuilder's to define, and defer to its README if the
> two ever disagree.

> **Prerequisite.** `import ambrs` is currently broken on `analysis_clean`: the
> particle-population library is not declared in `requirements.txt` and is
> imported under a module name it no longer installs under. Until that is
> resolved (see issue #4), the Python steps in section 3 won't run from a clean
> install. Building the executables in section 1 is unaffected.

## 1. Build the models with ambuilder

The models are forked under the [AMBRS-Project](https://github.com/AMBRS-project)
organization and built by a CMake-based tool,
[ambuilder](https://github.com/AMBRS-project/ambuilder), which fetches and builds
each model and its dependencies:

* [PartMC](https://github.com/AMBRS-project/partmc) — stochastic,
  particle-resolved
* [MAM4](https://github.com/AMBRS-project/MAM_box_model) — 4-mode modal

(ambuilder can also build CARMA, for which `ambrs` has no adapter yet.)

You will need C and Fortran compilers and [CMake](https://cmake.org/). Then:

```bash
git clone https://github.com/AMBRS-project/ambuilder.git
cd ambuilder
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<prefix> -G "Unix Makefiles" [options]
cd build
make
make install
```

The executables are installed into `<prefix>/bin`. Useful options:

| Option | Meaning |
| --- | --- |
| `CMAKE_BUILD_TYPE={Debug,Release}` | build type (default `Release`) |
| `CMAKE_C_COMPILER=/path/to/cc` | C compiler |
| `CMAKE_Fortran_COMPILER=/path/to/fc` | Fortran compiler |
| `CMAKE_INSTALL_PREFIX=/path/to/install` | install location |
| `ENABLE_MOSAIC={ON,OFF}` | MOSAIC chemistry (default `OFF`) |
| `ENABLE_CAMP={ON,OFF}` | CAMP chemistry (default `OFF`) |

Two constraints from ambuilder: **only one of `ENABLE_CAMP` and `ENABLE_MOSAIC`
may be enabled**, and **PartMC does not build with the Ninja generator** — use
`-G "Unix Makefiles"` as shown.

### Which options does `ambrs` actually need?

* **`ENABLE_MOSAIC=ON` if you want condensation in PartMC.** The adapter maps
  `AerosolProcesses.condensation` onto PartMC's `do_mosaic` flag
  (`ambrs/partmc.py`), so a PartMC built without MOSAIC cannot run a scenario
  with `condensation=True`.
* **CAMP is not needed.** The adapter always writes `do_camp_chem = False`.

After `make install`, note the executable paths under `<prefix>/bin` — you pass
them to `PoolRunner` below.

## 2. Run them from `ambrs`

Executable models are driven by `ambrs.PoolRunner`, which gives each scenario its
own subdirectory under `root`, writes that scenario's input files, runs the
executable there as a subprocess (scenarios in parallel), and reads each
scenario's output back into an `ambrs.analysis.Output`.

```python
import ambrs

# ... build `ensemble` as usual (see README.md / demo.ipynb) ...

processes = ambrs.AerosolProcesses(coagulation=True)
model = ambrs.partmc.AerosolModel(processes, n_part=10000)   # or ambrs.mam4.AerosolModel(processes)

inputs = model.create_inputs(ensemble, dt=60.0, nstep=60)    # 1 h at 60 s steps

runner = ambrs.PoolRunner(
    model,
    executable='<prefix>/bin/<model-executable>',
    root='./runs',                 # must already exist
    num_processes=4,               # optional; defaults to the cpu count
    scenario_name='{index}',       # optional; '{index}' is the 1-based index
)
outputs = runner.run(inputs)       # -> list[ambrs.analysis.Output], one per input
```

Notes:

* **`root` must already exist** — `PoolRunner` raises `OSError` otherwise. It
  creates one subdirectory per scenario inside it (named by `scenario_name`), and
  warns if a scenario directory already exists (contents are overwritten).
* Each scenario directory also gets `stdout.log`, `stderr.log`, and `timer.log`
  (wall-clock seconds). **Check `stderr.log` first when a run fails.**
* `runner.run()` returning `list[Output]` requires the output-gathering support
  added in the "return Outputs from PoolRunner" change; on older revisions
  `run()` returns `None` and you must call
  `model.read_output_files(input, dir, scenario_name)` yourself.

### What each model reads and writes

| | PartMC | MAM4 |
| --- | --- | --- |
| Invocation | `<exe> <scenario>.spec` | `<exe>` (no arguments) |
| Input files | `<scenario>.spec` plus `gas_data.dat`, `gas_init.dat`, `aero_data.dat`, `aero_init_dist.dat`, `temp.dat`, `pres.dat`, `height.dat`, … | a single file named `namelist` |
| Output | `out/<scenario>_<repeat>_<step>.nc` (NetCDF) | `mam_output.nc` |
| Final state | most recent `.nc` file | last record in `mam_output.nc` |

Both models are run with the scenario directory as the working directory, so
their input files are found by relative path.

## 3. Analyze the results

Every model's `read_output_files` returns the same `ambrs.analysis.Output`, whose
`particle_population` is a particle-population object (the `PyParticle` /
`part2pop` library — see issue #4). So the analysis is identical
across models:

```python
for out in outputs:
    pop = out.particle_population
    print(out.scenario_name, pop.get_Ntot(), pop.get_tot_dry_mass())
```

Because the `Output` type is shared, PartMC and MAM4 results can be compared
against each other and against the in-process models with the same code — see
`ambrs/analysis.py` (e.g. `kl_divergence`) and the plotting helpers in
`ambrs/visualization/`.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `OSError: root path '...' does not exist!` | create the `root` directory before constructing `PoolRunner` |
| `OSError` / no NetCDF output found | the run failed — read the scenario's `stderr.log` |
| PartMC exits complaining about MOSAIC | built without `ENABLE_MOSAIC=ON` but the scenario has `condensation=True` |
| PartMC fails to build | don't use the Ninja generator; use `-G "Unix Makefiles"` |
| CMake configure error about CAMP/MOSAIC | only one of `ENABLE_CAMP` / `ENABLE_MOSAIC` may be `ON` |
