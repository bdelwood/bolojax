"""Class to model camera."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from bolojax.compute.noise import Noise
from bolojax.models.base import BolojaxModel
from bolojax.models.channel import ChannelConfig
from bolojax.models.params import Var
from bolojax.models.sky import Universe
from bolojax.models.utils import is_not_none

if TYPE_CHECKING:
    from bolojax.models.instrument import InstrumentConfig
    from bolojax.models.optics import OpticalElement


class CameraConfig(BolojaxModel):
    """Camera model."""

    boresite_elevation: Var() = 0.0
    pixel_elevation: Var() = None
    optical_coupling: Var() = 1.0
    f_number: Var() = 2.5
    bath_temperature: Var("K") = 0.1
    skip_optical_elements: list = Field(default_factory=list)
    chan_config: dict | None = None
    beam_model: str | dict = "bolocalc"

    # Channels built from chan_config during construction
    channels: dict[str, ChannelConfig] = Field(default_factory=dict)

    _optics: OrderedDict[str, OpticalElement] | None = PrivateAttr(default=None)
    _instrument: InstrumentConfig | None = PrivateAttr(default=None)
    _name: str | None = PrivateAttr(default=None)
    _noise_calc: Noise | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _init_derived(self) -> CameraConfig:
        # Build the noise calculator with the configured beam model
        self._noise_calc = Noise(beam_preset=self.beam_model)

        # Build channels from chan_config if channels weren't provided directly
        if not self.channels and self.chan_config:
            defaults = self.chan_config.get("default", {})
            channels = OrderedDict()
            for name, overrides in self.chan_config["elements"].items():
                channels[name] = ChannelConfig(**{**defaults, **(overrides or {})})
            self.channels = channels
        return self

    @property
    def noise_calc(self) -> Noise | None:
        """Return the noise calculator for this camera."""
        return self._noise_calc

    @property
    def optics(self) -> OrderedDict[str, OpticalElement] | None:
        return self._optics

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        self._name = value

    def set_parent(self, instrument: InstrumentConfig, name: str) -> None:
        """Pass information from the parent instrument down the food chain."""
        self._instrument = instrument
        self._name = name
        optics = OrderedDict()
        for key, val in instrument.optics.elements.items():
            if key in self.skip_optical_elements:
                continue
            optics[key] = val
        self._optics = optics
        for idx, chan in enumerate(self.channels.values()):
            chan.set_camera(self, idx)

    def sample(self, nsamples: int = 0) -> None:
        """Sample parameters in all the channels."""
        for chan in self.channels.values():
            chan.sample(nsamples)

    def eval_optical_chains(
        self, nsamples: int = 0, freq_resol: float | None = None
    ) -> None:
        """Compute the performance of the elements of the optical chain for each channel."""
        for chan in self.channels.values():
            chan.eval_optical_chain(nsamples, freq_resol)

    def eval_sky(
        self, universe: Universe, nsamples: int = 0, freq_resol: float | None = None
    ) -> None:
        """Compute parameters related to the sky that depend on the particular camera.

        If ``pixel_elevation`` is configured, each channel gets the
        atmosphere evaluated at a per-pixel elevation drawn from the
        distribution. Otherwise the instrument-level elevation is used.
        """
        elevation = None
        if (
            is_not_none(self.pixel_elevation)
            and np.isfinite(self.pixel_elevation.SI).all()
        ):
            self.pixel_elevation.sample(nsamples)
            elevation = self.pixel_elevation()
        for chan in self.channels.values():
            chan.eval_sky(universe, freq_resol, elevation=elevation)

    def eval_det_response(
        self, nsample: int = 0, freq_resol: float | None = None
    ) -> None:
        """Compute the performance of the detectors of the optical chain."""
        for chan in self.channels.values():
            chan.eval_det_response(nsample, freq_resol)

    @property
    def instrument(self) -> InstrumentConfig | None:
        """Return the parent instrument."""
        return self._instrument


def build_cameras(def_channel_config: dict, camera_config: dict) -> OrderedDict:
    """Build a set of cameras from a configuration dictionary."""
    defaults = camera_config.get("default", {})
    ret = OrderedDict()
    for key, overrides in camera_config["elements"].items():
        cam_cfg = {**defaults, **(overrides or {})}
        cam_cfg["chan_config"]["default"] = def_channel_config.copy()
        ret[key] = CameraConfig(**cam_cfg)
    return ret
