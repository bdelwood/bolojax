# bolojax

<!-- readme-include-start -->

[![CI status][ci-img]][ci-url] [![Documentation][doc-img]][doc-url]
[![PyPI version][pypi-img]][pypi-url] [![Python][python-img]][pypi-url]
[![License][license-img]][license-url]

[ci-img]:
  https://img.shields.io/github/actions/workflow/status/bdelwood/bolojax/ci.yaml?branch=master&style=flat-square&label=CI
[ci-url]: https://github.com/bdelwood/bolojax/actions/workflows/ci.yaml
[doc-img]: https://img.shields.io/badge/docs-bolojax-4d76ae?style=flat-square
[doc-url]: https://bdelwood.github.io/bolojax/
[pypi-img]: https://img.shields.io/pypi/v/bolojax?style=flat-square
[python-img]: https://img.shields.io/pypi/pyversions/bolojax?style=flat-square
[pypi-url]: https://pypi.org/project/bolojax/
[license-img]:
  https://img.shields.io/badge/license-BSD--3--Clause-yellow?style=flat-square
[license-url]: https://github.com/bdelwood/bolojax/blob/master/LICENSE

Bolometric sensitivity calculator for CMB instruments, built on
[JAX](https://github.com/jax-ml/jax).

bolojax models the full radiative transfer chain of a CMB telescope and computes
noise-equivalent temperature (NET), noise-equivalent power (NEP), and mapping
speed. Because the compute path is written in pure JAX, the entire forward model
is automatically differentiable, enabling gradient-based fitting, Fisher
forecasting, MCMC sampling, and potentially inverse design.

## History

This package descends from [BoloCalc](https://github.com/chill90/BoloCalc) by
Charlie Hill ([arXiv:1806.04316](https://arxiv.org/abs/1806.04316)), which was
subsequently forked and restructured by Eric Charles at
[KIPAC/bolo-calc](https://github.com/KIPAC/bolo-calc). bolojax is a fork of
bolo-calc that replaces the configuration layer with
[pydantic](https://docs.pydantic.dev/) and the numerical backend with JAX and
[equinox](https://docs.kidger.site/equinox/), making the sensitivity calculation
JIT-compiled and fully differentiable.

## What you can do

| Task                                   | Before (BoloCalc / bolo-calc)                                                                                                                                                                  | With bolojax                                                                    |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Forward modeling**                   | Supported (YAML config only)                                                                                                                                                                   | Supported, JIT-compiled                                                         |
| **Monte Carlo (posterior predictive)** | Supported (sample parameters, run many realizations)                                                                                                                                           | Supported, faster with JIT                                                      |
| **Least-squares fitting**              | Possible, but the config-file-only interface meant wrapping the entire pipeline as a subprocess and perturbing parameters via text file manipulation; Jacobians computed by finite differences | Programmatic API exposes JAX pytrees directly; exact Jacobians through autodiff |
| **Fisher analysis**                    | Possible in principle, but Jacobian accuracy limited by finite-difference step size and subprocess overhead made each evaluation expensive                                                     | Exact Fisher matrix from autodiff in a single JIT-compiled pass                 |
| **MCMC / HMC**                         | Not practical, as one could not utilize gradient-based samplers.                                                                                                                               | Enabled by autodiff                                                             |

## Installation

```bash
uv pip install "git+https://github.com/bdelwood/bolojax.git"
```

For GPU support:

```bash
 uv pip install "bolojax[gpu] @ git+https://github.com/bdelwood/bolojax.git"
```

## Model structure

An `Experiment` is the top-level container. It holds a `Universe` (sky model),
an `Instrument` (telescope hardware), and a `SimConfig` (simulation settings):

```
Experiment
├── SimConfig          # nsky_sim, ndet_sim, freq_resol
├── Universe
│   ├── Atmosphere     # HDF5 lookup table or custom text file
│   ├── Dust           # modified blackbody foreground
│   └── Synchrotron    # power-law foreground
└── Instrument
    ├── Optics         # ordered chain of optical elements
    │   ├── Mirror     # reflective element (conductivity → ohmic loss)
    │   ├── Dielectric # transmissive element (thickness, index, loss tangent)
    │   └── ApertureStop
    ├── Readout        # SQUID NEI, bolometer resistance
    └── Camera(s)
        └── Channel(s) # band center, bandwidth, detector parameters
```

The sensitivity calculation proceeds in two stages:

1. **Setup** (Python/numpy): sample sky and detector parameters, evaluate the
   optical chain, populate frequency-dependent temperatures, transmissions, and
   emissivities.
2. **Compute** (JAX):
   `compute_sensitivity(OpticsState, BoloParams) -> SensitivityResult` runs the
   full radiative transfer, noise estimation, and NET calculation as a single
   JIT-compiled, differentiable function.

`build_params(channel)` bridges the two stages, extracting JAX pytrees from a
configured `Channel`.

## Quick start

### YAML-driven

```python
import yaml
import bolojax

with open("config/myExample.yaml") as f:
    config = yaml.safe_load(f)

experiment = bolojax.Experiment(**config)
experiment.run()
experiment.instrument.print_summary()
```

Or from the command line:

```bash
bolojax -i config/myExample.yaml -o results.fits
```

### Programmatic

```python
import yaml
import equinox as eqx
import bolojax

# 1. Load base configuration
with open("config/myExample.yaml") as f:
    config = yaml.safe_load(f)
experiment = bolojax.Experiment(**config)
sim = experiment.sim_config
experiment.instrument.eval_sky(experiment.universe, sim.nsky_sim, sim.freq_resol)
experiment.instrument.eval_instrument(sim.ndet_sim, sim.freq_resol)

# 2. Extract JAX pytrees from a channel
camera = list(experiment.instrument.cameras.values())[0]
channel = list(camera.channels.values())[0]
optics, params, elem_names = bolojax.build_params(channel)

# 3. Compute
results = bolojax.compute_sensitivity(optics, params)
print(f"NET = {results.NET.squeeze():.4e} K*sqrt(s)")

# 4. Differentiate
@eqx.filter_grad
def grad_net(p):
    return bolojax.compute_sensitivity(optics, p).NET.squeeze()

g = grad_net(params)
print(f"d(NET)/d(Tc) = {g.Tc:.4e}")

# 5. Swap parameters with eqx.tree_at and recompute
new_params = eqx.tree_at(lambda p: p.Tc, params, params.Tc * 1.1)
new_results = bolojax.compute_sensitivity(optics, new_params)
```

## License

BSD 3-Clause. See
[LICENSE](https://github.com/bdelwood/bolojax/blob/master/LICENSE) for details.
