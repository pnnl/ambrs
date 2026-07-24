"""Cross-model demonstration for AMBRS.

Builds ONE shared perturbed-parameter ensemble of 4-mode scenarios and runs it
through every *in-process* AMBRS model that is installed (TOMAS-JAX and
MAM4-JAX), then produces comparison graphs you can show to others:

  1. demo_figures/size_distributions.png -- per-scenario dry size distribution
     (dN/dlnD): each model's 1-hour-evolved state vs. the initial lognormal.
  2. demo_figures/bulk_summary.png       -- total number and dry-mass
     concentration per scenario, grouped by model.

Only coagulation is enabled, so the comparison is a clean, reproducible test of
the same initial condition evolving under each model's microphysics (gas-phase
treatment, whose units differ between models, is not exercised).

The executable models (PartMC, MAM4) are NOT run here: they need binaries built
via ambuilder and are driven by ambrs.runners.PoolRunner (see the note at the
bottom of this file). Run from the repository root:

    python demo_compare_models.py

Requires the optional in-process models (tomas_jax and/or mam4_jax) plus
PyParticle. Any model that is not installed is skipped with a message.
"""
import logging
import os
import warnings
from math import log10, pi, log

import numpy as np
import matplotlib
matplotlib.use('Agg')            # write files without a display
import matplotlib.pyplot as plt
import scipy.stats as stats

import ambrs
import ambrs.aerosol as aerosol
import ambrs.gas as gas

logging.basicConfig(level=logging.WARNING)
np.random.seed(12345)            # reproducible ensemble

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# consistent per-model styling
MODEL_STYLE = {
    'TOMAS-JAX': {'color': '#1f77b4'},
    'MAM4-JAX':  {'color': '#d62728'},
}

# ---------------------------------------------------------------------------
# 1. a shared ensemble usable by every 4-mode modal model
# ---------------------------------------------------------------------------
so4 = aerosol.AerosolSpecies(name='SO4', molar_mass=97.071, density=1770,
                             hygroscopicity=0.507)
oc = aerosol.AerosolSpecies(name='OC', molar_mass=12.011, density=1000,
                            hygroscopicity=0.1)
h2so4 = gas.GasSpecies(name='H2SO4', molar_mass=98.079)

# MAM4's fixed geometric std devs, so MAM4-JAX doesn't warn about mode width
LOG10_SIGMA = [log10(1.8), log10(1.6), log10(1.8), log10(1.6)]


def _mode(name, species, number, gmd, log10_gsd):
    frac = 1.0 / len(species)
    return aerosol.AerosolModeDistribution(
        name=name, species=species, number=number, geom_mean_diam=gmd,
        log10_geom_std_dev=log10_gsd,
        mass_fractions=[stats.uniform(frac, 0) for _ in species])


SPEC = ambrs.EnsembleSpecification(
    name='cross_model_demo',
    aerosols=(so4, oc),
    gases=(h2so4,),
    size=aerosol.AerosolModalSizeDistribution(modes=[
        _mode('accumulation', [so4, oc],
              stats.loguniform(5e9, 2e10), stats.loguniform(9e-8, 1.3e-7),
              LOG10_SIGMA[0]),
        _mode('aitken', [so4],
              stats.loguniform(5e10, 2e11), stats.loguniform(2e-8, 4e-8),
              LOG10_SIGMA[1]),
        _mode('coarse', [so4],
              stats.loguniform(1e5, 1e6), stats.loguniform(1e-6, 3e-6),
              LOG10_SIGMA[2]),
        _mode('primary_carbon', [oc],
              stats.loguniform(5e9, 2e10), stats.loguniform(3e-8, 6e-8),
              LOG10_SIGMA[3]),
    ]),
    gas_concs=(stats.loguniform(1e-14, 1e-12) for _ in range(1)),  # unused (coag only)
    flux=stats.loguniform(1e-11, 1e-8),
    relative_humidity=stats.uniform(0.3, 0.4),
    temperature=stats.uniform(280, 20),
    pressure=101325,
    height=500,
)

N_SCENARIOS = 4
DT, NSTEP = 60.0, 60             # one simulated hour
ensemble = ambrs.sample(SPEC, N_SCENARIOS)

# ---------------------------------------------------------------------------
# 2. build the available in-process models
# ---------------------------------------------------------------------------
processes = ambrs.AerosolProcesses(coagulation=True)   # gas units differ, so coag only
models = {}
if ambrs.mam4_jax._MAM4_JAX_AVAILABLE:
    models['MAM4-JAX'] = ambrs.mam4_jax.AerosolModel(processes)
if ambrs.tomas_jax._TOMAS_AVAILABLE:
    models['TOMAS-JAX'] = ambrs.tomas_jax.AerosolModel(processes)

if not models:
    raise SystemExit('No in-process models installed (need tomas_jax and/or '
                     'mam4_jax). See TOMAS_JAX.md / MAM4_JAX.md.')
print(f'Running {N_SCENARIOS} scenarios through: {", ".join(models)}')

# run each model over the shared ensemble (warnings about approximate species
# mapping are expected and silenced for a clean demo)
outputs = {}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for name, model in models.items():
        inputs = model.create_inputs(ensemble, dt=DT, nstep=NSTEP)
        outputs[name] = ambrs.DirectRunner(model).run(inputs)

# ---------------------------------------------------------------------------
# 3. helpers: dN/dlnD from a population, and the analytic initial lognormal
# ---------------------------------------------------------------------------
D_MIN, D_MAX, N_BINS = 1e-9, 1e-4, 40
_EDGES = np.logspace(np.log10(D_MIN), np.log10(D_MAX), N_BINS + 1)
_CENTERS = np.sqrt(_EDGES[:-1] * _EDGES[1:])
_DLN = np.diff(np.log(_EDGES))


def dNdlnD(population):
    """Log-binned dN/dlnD [# m^-3] from a PyParticle population; integrates
(sum over bins of dN/dlnD * dlnD) to the population's total number."""
    D = np.asarray(population.get_particle_var('dry_diameter'))
    w = np.asarray(population.num_concs)
    counts, _ = np.histogram(D, bins=_EDGES, weights=w)
    return counts / _DLN


def initial_dNdlnD(scenario):
    """Analytic dN/dlnD of the scenario's initial sum-of-lognormals."""
    y = np.zeros_like(_CENTERS)
    for m in scenario.size.modes:
        lnsig = log(10.0 ** m.log10_geom_std_dev)
        y += (m.number / (np.sqrt(2 * pi) * lnsig)
              * np.exp(-(np.log(_CENTERS / m.geom_mean_diam) ** 2)
                       / (2 * lnsig ** 2)))
    return y


# ---------------------------------------------------------------------------
# 4. figure 1: per-scenario size distributions
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(N_SCENARIOS, 1, figsize=(7, 2.6 * N_SCENARIOS),
                         sharex=True)
axes = np.atleast_1d(axes)
for i in range(N_SCENARIOS):
    ax = axes[i]
    scenario = ensemble.member(i)
    ax.plot(_CENTERS * 1e9, initial_dNdlnD(scenario), color='0.6', ls='--',
            lw=1.5, label='initial (lognormal)')
    for name in models:
        ax.plot(_CENTERS * 1e9, dNdlnD(outputs[name][i].particle_population),
                lw=2, label=f'{name} (1 h)', **MODEL_STYLE[name])
    ax.set_xscale('log')
    ax.set_ylabel(r'$dN/d\ln D$  [m$^{-3}$]')
    ax.set_title(f'scenario {i + 1}', fontsize=10, loc='left')
    ax.grid(True, which='both', alpha=0.2)
    if i == 0:
        ax.legend(frameon=False, fontsize=9)
axes[-1].set_xlabel('dry diameter [nm]')
fig.suptitle('Dry size distribution after 1 h of coagulation', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.99))
f1 = os.path.join(OUT_DIR, 'size_distributions.png')
fig.savefig(f1, dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. figure 2: bulk number and dry-mass summary
# ---------------------------------------------------------------------------
fig, (axN, axM) = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(N_SCENARIOS)
width = 0.8 / max(len(models), 1)
for j, name in enumerate(models):
    Ntot = [outputs[name][i].particle_population.get_Ntot() for i in range(N_SCENARIOS)]
    Mtot = [outputs[name][i].particle_population.get_tot_dry_mass() for i in range(N_SCENARIOS)]
    off = (j - (len(models) - 1) / 2) * width
    axN.bar(x + off, Ntot, width, label=name, color=MODEL_STYLE[name]['color'])
    axM.bar(x + off, Mtot, width, label=name, color=MODEL_STYLE[name]['color'])
for ax, title, ylab in ((axN, 'Total number concentration', r'$N_{tot}$ [m$^{-3}$]'),
                        (axM, 'Total dry mass concentration', 'dry mass [kg m$^{-3}$]')):
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f's{i + 1}' for i in range(N_SCENARIOS)])
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylab)
    ax.grid(True, axis='y', alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle('Bulk aerosol metrics after 1 h (by model)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.98))
f2 = os.path.join(OUT_DIR, 'bulk_summary.png')
fig.savefig(f2, dpi=150)
plt.close(fig)

print('\nWrote:')
print(' ', f1)
print(' ', f2)

# ---------------------------------------------------------------------------
# Note on the executable models (PartMC, MAM4)
# ---------------------------------------------------------------------------
# PartMC and MAM4 are external executables (built separately via ambuilder) and
# are driven by ambrs.runners.PoolRunner rather than DirectRunner, e.g.:
#
#     runner = ambrs.PoolRunner(model, executable='/path/to/exe', root='./runs')
#     runner.run(model.create_inputs(ensemble, dt=DT, nstep=NSTEP))
#     outs = [model.read_output_files(inp, run_dir, name) for ...]
#
# Once you have those binaries, add them to `models` and read their Outputs the
# same way; dNdlnD()/get_Ntot() above work on any AMBRS Output's population.
