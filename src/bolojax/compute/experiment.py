"""Top-level compute objects.

``Instrument`` holds the optical chain and detector parameters.
``Experiment`` adds survey parameters and provides ``.compute()``
and ``.to_dataset()`` methods.
"""

from __future__ import annotations

from collections import OrderedDict

import xarray as xr
import zodiax as zdx
from jax import Array

from bolojax.compute.sensitivity import SensitivityResult, compute_sensitivity


class Instrument(zdx.Base):
    """Optical chain + detector parameters.

    The differentiable instrument model. Each element in the optical
    chain implements ``emiss_trans(freqs)``, making the full chain
    differentiable via JAX autodiff.
    """

    # Optical chain
    freqs: Array
    bandwidth: float
    elements: OrderedDict
    corr_factors: Array

    # Bolometer thermal
    Tc: Array
    bath_temp: Array
    carrier_index: Array
    psat: Array
    psat_factor: Array
    G: Array
    Flink: Array

    # Readout
    squid_nei: Array
    bolo_R: Array
    response_factor: Array
    read_frac: Array
    optical_coupling: Array

    # Array
    ndet: int
    det_yield: Array


class Experiment(zdx.Base):
    """Top-level compute object: instrument + survey parameters.

    Use ``.compute()`` for a JAX-traceable ``SensitivityResult``,
    or ``.to_dataset()`` for a labeled ``xarray.Dataset``.

    Modify parameters with ``.set()`` and recompute::

        exp2 = experiment.set("instrument.elements.window.loss_tangent", lt)
        result = exp2.compute()
    """

    instrument: Instrument

    # Survey parameters
    fsky: Array
    obs_time: Array
    obs_effic: Array
    NET_scale: Array

    def compute(self) -> SensitivityResult:
        """Run the sensitivity computation. JAX-traceable."""
        return compute_sensitivity(self)

    def to_dataset(self) -> xr.Dataset:
        """Compute and return an xarray Dataset with labeled dims and units."""
        return self.compute().to_dataset(list(self.instrument.elements.keys()))
