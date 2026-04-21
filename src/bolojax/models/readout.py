"""Instrument readout model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from bolojax.models.params import Var


class Readout(BaseModel):
    """Instrument readout model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    squid_nei: Var() = None
    read_noise_frac: Var() = 0.1
    dwell_time: Var() = None
    revisit_rate: Var() = None
    nyquist_inductance: Var() = None
