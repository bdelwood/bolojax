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
[pydantic](https://docs.pydantic.dev/) and the numerical backend with JAX,
[equinox](https://docs.kidger.site/equinox/), and
[zodiax](https://github.com/LouisDesdoigts/zodiax), making the sensitivity
calculation JIT-compiled and fully differentiable.

## What you can do

| Task                                   | Before (BoloCalc / bolo-calc)                                 | With bolojax                                      |
| -------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------- |
| **Forward modeling**                   | Supported (YAML config only)                                  | Supported, JIT-compiled                           |
| **Monte Carlo (posterior predictive)** | Supported (sample parameters, run many realizations)          | Supported, faster with JIT                        |
| **Least-squares fitting**              | Subprocess wrapper, finite-difference Jacobians               | Exact Jacobians through autodiff                  |
| **Fisher analysis**                    | Limited by finite-difference accuracy and subprocess overhead | Exact Fisher matrix in a single JIT-compiled pass |
| **MCMC / HMC**                         | Not practical (no gradients)                                  | Enabled by autodiff                               |

## Installation

```bash
uv pip install bolojax
```

For GPU support:

```bash
uv pip install 'bolojax[gpu]'
```

## Architecture

bolojax separates **configuration** (pydantic) from **computation**
(JAX/zodiax):

```
Config layer (pydantic)         Compute layer (zodiax/JAX)
───────────────────────         ──────────────────────────
ExperimentConfig                Experiment
  SimConfig                       Instrument
  Universe                          elements: {name: Element}
  InstrumentConfig                  Tc, bath_temp, psat, ...
    Optics                        fsky, obs_time, obs_effic
    CameraConfig                SensitivityResult
      ChannelConfig               NET, NEP, powers, ...
```

`ExperimentConfig.setup()` bridges the two layers, returning an `Experiment`
zodiax pytree that you compute with, differentiate through, and modify with
`.set()`.

## Quick start

### YAML-driven

```python
import bolojax

config = bolojax.ExperimentConfig.from_yaml("config/example.yaml")
experiment = config.setup()

# Compute and export
ds = experiment.to_dataset()
ds.to_netcdf("results.nc")
```

Or from the command line:

```bash
bolojax -i config/example.yaml -o results.nc
```

### Programmatic

```python
import bolojax
import equinox as eqx

# 1. Load configuration and set up the compute object
config = bolojax.ExperimentConfig.from_yaml("config/example.yaml")
experiment = config.setup()

# 2. Compute sensitivity
result = experiment.compute()
print(f"NET = {result.NET.squeeze() * 1e6:.2f} uK-rts")

# 3. Get labeled xarray output
ds = experiment.to_dataset()
ds.to_netcdf("results.nc")

# 4. Modify parameters and recompute
exp2 = experiment.set("instrument.elements.window.loss_tangent", 2.5e-4)
result2 = exp2.compute()

# 5. Differentiate (filter_grad skips non-float leaves like ndet)
@eqx.filter_grad
def grad_net(exp):
    return exp.compute().NET.squeeze()

g = grad_net(experiment)
```

### Optical element types

Each element in the optical chain has a `type` field and computes its own
emissivity and transmission from physical properties. Quantities can include
units.:

```yaml
elements:
  - forebaffle:
      temperature: "240 K"
      scatter_frac: 0.013
  - window:
      type: dielectric
      temperature: "260 K"
      thickness: "0.025 m"
      index: 1.5246
      loss_tangent: 1.38e-4
      reflection: 0.01
  - primary:
      type: mirror
      temperature: "273 K"
      conductivity: 3.6e7
  - aperture:
      type: aperture_stop
      temperature: "5.5 K"
      spillover: 0.15
```

## License

BSD 3-Clause. See
[LICENSE](https://github.com/bdelwood/bolojax/blob/master/LICENSE) for details.
