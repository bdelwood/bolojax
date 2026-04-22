# ruff: noqa: E402
# E402 disabled: jax.config.update must run before any JAX imports.
"""Bolometric sensitivity calculator for CMB instruments."""

import jax

jax.config.update("jax_enable_x64", True)

from bolojax._version import __version__ as __version__
from bolojax.compute import elements as elements
from bolojax.compute import noise as noise
from bolojax.compute import physics as physics
from bolojax.compute.sensitivity import BoloParams as BoloParams
from bolojax.compute.sensitivity import OpticsState as OpticsState
from bolojax.compute.sensitivity import SensitivityResult as SensitivityResult
from bolojax.compute.sensitivity import compute_sensitivity as compute_sensitivity
from bolojax.io.sensitivity import Sensitivity as Sensitivity
from bolojax.io.sensitivity import build_params as build_params
from bolojax.io.tables import TableDict as TableDict
from bolojax.models import unit as unit
from bolojax.models import utils as utils
from bolojax.models.camera import Camera as Camera
from bolojax.models.camera import build_cameras as build_cameras
from bolojax.models.channel import Channel as Channel
from bolojax.models.experiment import Experiment as Experiment
from bolojax.models.experiment import SimConfig as SimConfig
from bolojax.models.instrument import Instrument as Instrument
from bolojax.models.interp import FreqInterp as FreqInterp
from bolojax.models.params import OutputField as OutputField
from bolojax.models.params import OutputHolder as OutputHolder
from bolojax.models.params import ParamHolder as ParamHolder
from bolojax.models.params import StatsSummary as StatsSummary
from bolojax.models.params import Var as Var
from bolojax.models.params import VariableHolder as VariableHolder
from bolojax.models.pdf import ChoiceDist as ChoiceDist
from bolojax.models.readout import Readout as Readout
from bolojax.models.sky import AmAtm as AmAtm
from bolojax.models.sky import AtmBackend as AtmBackend
from bolojax.models.sky import Atmosphere as Atmosphere
from bolojax.models.sky import AtmProfile as AtmProfile
from bolojax.models.sky import Dust as Dust
from bolojax.models.sky import Foreground as Foreground
from bolojax.models.sky import Synchrotron as Synchrotron
from bolojax.models.sky import Universe as Universe
