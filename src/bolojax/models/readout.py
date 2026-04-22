"""Instrument readout model."""

from __future__ import annotations

from bolojax.models.base import BolojaxModel
from bolojax.models.params import Var


class Readout(BolojaxModel):
    """Instrument readout model."""

    squid_nei: Var() = None
    read_noise_frac: Var() = 0.1
    dwell_time: Var() = None
    revisit_rate: Var() = None
    nyquist_inductance: Var() = None
