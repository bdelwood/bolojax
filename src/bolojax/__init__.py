# ruff: noqa: E402
# E402 disabled: jax.config.update must run before any JAX imports.
"""Bolometric sensitivity calculator for CMB instruments."""

import jax

jax.config.update("jax_enable_x64", True)

from bolojax._version import __version__ as __version__
from bolojax.compute import elements as elements
from bolojax.compute import noise as noise
from bolojax.compute import physics as physics
from bolojax.compute.experiment import Experiment as Experiment
from bolojax.compute.experiment import Instrument as Instrument
from bolojax.compute.sensitivity import SensitivityResult as SensitivityResult
from bolojax.compute.sensitivity import compute_sensitivity as compute_sensitivity
from bolojax.models.experiment import ExperimentConfig as ExperimentConfig
from bolojax.models.experiment import SimConfig as SimConfig
from bolojax.models.params import Var as Var
