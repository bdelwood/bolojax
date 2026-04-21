# ruff: noqa: E402
# E402 disabled: jax.config.update must run before any JAX imports.
"""Bolometric sensitivity calculator for CMB instruments."""

import jax

jax.config.update("jax_enable_x64", True)

from bolojax._version import __version__ as __version__

from .compute import noise as noise
from .compute import physics as physics
from .compute.sensitivity import BoloParams as BoloParams
from .compute.sensitivity import OpticsState as OpticsState
from .compute.sensitivity import SensitivityResult as SensitivityResult
from .compute.sensitivity import compute_sensitivity as compute_sensitivity
from .io.sensitivity import Sensitivity as Sensitivity
from .io.sensitivity import build_params as build_params
from .io.tables import TableDict as TableDict
from .models import unit as unit
from .models import utils as utils
from .models.camera import Camera as Camera
from .models.camera import build_cameras as build_cameras
from .models.channel import Channel as Channel
from .models.experiment import Experiment as Experiment
from .models.experiment import SimConfig as SimConfig
from .models.instrument import Instrument as Instrument
from .models.interp import FreqInterp as FreqInterp
from .models.params import OutputField as OutputField
from .models.params import OutputHolder as OutputHolder
from .models.params import ParamHolder as ParamHolder
from .models.params import StatsSummary as StatsSummary
from .models.params import Var as Var
from .models.params import VariableHolder as VariableHolder
from .models.pdf import ChoiceDist as ChoiceDist
from .models.readout import Readout as Readout
from .models.sky import AtmModel as AtmModel
from .models.sky import Atmosphere as Atmosphere
from .models.sky import CustomAtm as CustomAtm
from .models.sky import Dust as Dust
from .models.sky import Foreground as Foreground
from .models.sky import Synchrotron as Synchrotron
from .models.sky import Universe as Universe
