"""Class to model camera."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from bolojax.compute.noise import Noise

from .channel import Channel
from .params import Var


class Camera(BaseModel):
    """Camera model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    boresite_elevation: Var() = 0.0
    optical_coupling: Var() = 1.0
    f_number: Var() = 2.5
    bath_temperature: Var() = 0.1
    skip_optical_elements: list = Field(default_factory=list)
    chan_config: dict | None = None
    beam_model: str = "bolocalc"

    # Channels built from chan_config during construction
    channels: dict[str, Channel] = Field(default_factory=dict)

    _optics: Any = PrivateAttr(default=None)
    _instrument: Any = PrivateAttr(default=None)
    _name: str | None = PrivateAttr(default=None)
    _noise_calc: Any = PrivateAttr(default=None)

    def model_post_init(self, __context):
        # Build the noise calculator with the configured beam model
        self._noise_calc = Noise(beam_preset=self.beam_model)

        # Build channels from chan_config if channels weren't provided directly
        if not self.channels and self.chan_config:
            defaults = self.chan_config.get("default", {})
            channels = OrderedDict()
            for name, overrides in self.chan_config["elements"].items():
                channels[name] = Channel(**{**defaults, **(overrides or {})})
            self.channels = channels

    @property
    def noise_calc(self):
        """Return the noise calculator for this camera."""
        return self._noise_calc

    @property
    def optics(self):
        return self._optics

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def set_parent(self, instrument, name):
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

    def sample(self, nsamples=0):
        """Sample parameters in all the channels."""
        for chan in self.channels.values():
            chan.sample(nsamples)

    def eval_optical_chains(self, nsamples=0, freq_resol=None):
        """Compute the performance of the elements of the optical chain for each channel."""
        for chan in self.channels.values():
            chan.eval_optical_chain(nsamples, freq_resol)

    def eval_sky(self, universe, freq_resol=None):
        """Compute parameters related to the sky that depend on the particular camera."""
        for chan in self.channels.values():
            chan.eval_sky(universe, freq_resol)

    def eval_det_response(self, nsample=0, freq_resol=None):
        """Compute the performance of the detectors of the optical chain."""
        for chan in self.channels.values():
            chan.eval_det_response(nsample, freq_resol)

    @property
    def instrument(self):
        """Return the parent instrument."""
        return self._instrument


def build_cameras(def_channel_config: dict, camera_config: dict) -> OrderedDict:
    """Build a set of cameras from a configuration dictionary."""
    defaults = camera_config.get("default", {})
    ret = OrderedDict()
    for key, overrides in camera_config["elements"].items():
        cam_cfg = {**defaults, **(overrides or {})}
        cam_cfg["chan_config"]["default"] = def_channel_config.copy()
        ret[key] = Camera(**cam_cfg)
    return ret
