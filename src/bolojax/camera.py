"""Class to model camera."""

from __future__ import annotations

from collections import OrderedDict as odict
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, create_model

from .cfg import Var, expand_dict
from .channel import Channel


class Camera_Base(BaseModel):
    """Camera model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    boresite_elevation: Var() = 0.0
    optical_coupling: Var() = 1.0
    f_number: Var() = 2.5
    bath_temperature: Var() = 0.1
    skip_optical_elements: list = []
    chan_config: dict | None = None

    _channels: Any = PrivateAttr(default=None)
    _optics: Any = PrivateAttr(default=None)
    _instrument: Any = PrivateAttr(default=None)
    _name: str | None = PrivateAttr(default=None)

    def model_post_init(self, __context):
        channels = odict()
        for key, val in self.__dict__.items():
            if isinstance(val, Channel):
                channels[key] = val
        object.__setattr__(self, "_channels", channels)
        # Ensure skip_optical_elements is a fresh list
        if self.skip_optical_elements is None:
            object.__setattr__(self, "skip_optical_elements", [])

    @property
    def channels(self):
        return self._channels

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
        optics = odict()
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
        return self._instrument


def build_camera_class(name="Camera", **kwargs):
    """Build a camera from a configuration dictionary."""
    kwcopy = kwargs.copy()
    type_dicts = [{None: Channel}]
    config_dicts = [kwcopy.pop("chan_config")]
    return _build_camera_model(name, Camera_Base, config_dicts, type_dicts, **kwcopy)


def _build_camera_model(name, base_class, config_dicts, type_dicts, **kwargs):
    """Build a pydantic model class dynamically and return an instance."""
    kwcopy = kwargs.copy()
    field_definitions = {}
    for config_dict, type_dict in zip(config_dicts, type_dicts):
        expanded = expand_dict(config_dict)
        kwcopy.update(expanded)
        for field_name, field_config in expanded.items():
            cls = type_dict[field_config.pop("obj_type", None)]
            field_definitions[field_name] = (cls | None, None)
    new_class = create_model(name, __base__=base_class, **field_definitions)
    return new_class(**kwcopy)


def build_cameras(def_channel_config, camera_config):
    """Build a set of cameras from a configuration dictionary."""
    cam_full = expand_dict(camera_config)

    ret = odict()
    for key, val in cam_full.items():
        cam_config = val.copy()
        cam_config["chan_config"]["default"] = def_channel_config.copy()
        ret[key] = build_camera_class(key, **cam_config)
    return ret
