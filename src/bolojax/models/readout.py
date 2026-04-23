"""Instrument readout model."""

from __future__ import annotations

from bolojax.models.base import BolojaxModel
from bolojax.models.params import Var


class Readout(BolojaxModel):
    """Instrument readout model."""

    squid_nei: Var("pA/rtHz") = None
    read_noise_frac: Var() = 0.1
    dwell_time: Var("s") = None
    revisit_rate: Var("Hz") = None
    nyquist_inductance: Var("nH") = None
