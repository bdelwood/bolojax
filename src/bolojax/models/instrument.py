"""Instrument configuration model."""

from __future__ import annotations

from collections import OrderedDict

from pydantic import PrivateAttr, model_validator

from bolojax.models.base import BolojaxModel
from bolojax.models.camera import CameraConfig, build_cameras
from bolojax.models.optics import Optics, build_optics
from bolojax.models.params import Var
from bolojax.models.readout import Readout
from bolojax.models.sky import Universe


class InstrumentConfig(BolojaxModel):
    """Instrument configuration: optics, cameras, channels, readout."""

    site: str
    sky_temp: Var("K") = None
    obs_time: Var("yr") = None
    sky_fraction: Var() = None
    NET: Var() = None

    custom_atm_file: str | None = None

    elevation: Var("deg") = None
    pwv: Var("mm") = None
    obs_effic: Var() = None

    readout: Readout
    camera_config: dict
    optics_config: dict
    channel_default: dict

    _optics: Optics | None = PrivateAttr(default=None)
    _cameras: OrderedDict[str, CameraConfig] | None = PrivateAttr(default=None)

    @property
    def optics(self) -> Optics | None:
        return self._optics

    @property
    def cameras(self) -> OrderedDict[str, CameraConfig] | None:
        return self._cameras

    @model_validator(mode="after")
    def _init_derived(self) -> InstrumentConfig:
        self._optics = build_optics(self.optics_config)
        self._cameras = build_cameras(self.channel_default, self.camera_config)
        for key, val in self._cameras.items():
            val.set_parent(self, key)
        return self

    def eval_sky(
        self, universe: Universe, nsamples: int = 0, freq_resol: float | None = None
    ) -> None:
        """Sample requested inputs and evaluate the parameters of the sky model."""
        universe.sample(nsamples)
        self.obs_effic.sample(nsamples)
        for camera in self.cameras.values():
            camera.eval_sky(universe, nsamples, freq_resol)

    def eval_instrument(
        self, nsamples: int = 0, freq_resol: float | None = None
    ) -> None:
        """Sample requested inputs and evaluate the parameters of the instrument model."""
        for camera in self.cameras.values():
            camera.sample(nsamples)
            camera.eval_optical_chains(nsamples, freq_resol)
            camera.eval_det_response(nsamples, freq_resol)
