"""Class to represent an instrument."""

from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Any

from astropy.table import vstack
from pydantic import BaseModel, ConfigDict, PrivateAttr

from bolojax.io.sensitivity import Sensitivity
from bolojax.io.tables import TableDict

from .camera import build_cameras
from .optics import build_optics
from .params import Var
from .readout import Readout


class Instrument(BaseModel):
    """Class to represent an instrument."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    site: str
    sky_temp: Var("K") = None
    obs_time: Var("yr") = None
    sky_fraction: Var() = None
    NET: Var() = None

    custom_atm_file: str | None = None

    elevation: Var() = None
    pwv: Var() = None
    obs_effic: Var() = None

    readout: Readout
    camera_config: dict
    optics_config: dict
    channel_default: dict

    _optics: Any = PrivateAttr(default=None)
    _cameras: Any = PrivateAttr(default=None)
    _tables: Any = PrivateAttr(default=None)
    _sns_dict: Any = PrivateAttr(default=None)

    @property
    def optics(self):
        return self._optics

    @property
    def cameras(self):
        return self._cameras

    @property
    def tables(self):
        return self._tables

    def model_post_init(self, __context):
        self._optics = build_optics(self.optics_config)
        self._cameras = build_cameras(self.channel_default, self.camera_config)
        for key, val in self._cameras.items():
            val.set_parent(self, key)

    def eval_sky(self, universe, nsamples=0, freq_resol=None):
        """Sample requested inputs and evaluate the parameters of the sky model."""
        universe.sample(nsamples)
        self.obs_effic.sample(nsamples)
        for camera in self.cameras.values():
            camera.eval_sky(universe, freq_resol)

    def eval_instrument(self, nsamples=0, freq_resol=None):
        """Sample requested inputs and evaluate the parameters of the instrument model."""
        for camera in self.cameras.values():
            camera.sample(nsamples)
            camera.eval_optical_chains(nsamples, freq_resol)
            camera.eval_det_response(nsamples, freq_resol)

    def eval_sensitivities(self):
        """Evaluate the sensitivities."""
        self._sns_dict = OrderedDict()
        for cam_name, camera in self.cameras.items():
            for chan_name, channel in camera.channels.items():
                full_name = f"{cam_name}_{chan_name}"
                self._sns_dict[full_name] = Sensitivity(channel)

    def make_tables(self, basename="", **kwargs):
        """Make fits tables with output values."""
        self._tables = TableDict()
        for key, val in self._sns_dict.items():
            val.make_tables(f"{basename}{key}", self._tables, **kwargs)

        if kwargs.get("save_summary", True):
            sum_keys = [key for key in self._tables.keys() if key.find("_summary") > 0]  # noqa: SIM118
            sum_table = vstack(
                [self._tables.pop_table(sum_key) for sum_key in sum_keys]
            )
            self._tables.add_datatable(f"{basename}summary", sum_table)
        if kwargs.get("save_optical", True):
            opt_keys = [key for key in self._tables.keys() if key.find("_optical") > 0]  # noqa: SIM118
            opt_table = vstack(
                [self._tables.pop_table(opt_key) for opt_key in opt_keys]
            )
            self._tables.add_datatable(f"{basename}optical", opt_table)
        return self._tables

    def write_tables(self, filename):
        """Write output fits tables."""
        if self._tables:
            self._tables.save_datatables(filename)

    def print_summary(self, stream=sys.stdout):
        """Print summary stats in human readable format."""
        for key, val in self._sns_dict.items():
            stream.write(f"{key} ---------\n")
            val.print_summary(stream)
            stream.write("---------\n")

    def print_optical_output(self, stream=sys.stdout):
        """Print summary stats in human readable format."""
        for key, val in self._sns_dict.items():
            stream.write(f"{key} ---------\n")
            val.print_optical_output(stream)
            stream.write("---------\n")

    def run(self, universe, sim_cfg, basename=""):
        """Run the analysis chain."""
        self.eval_sky(universe, sim_cfg.nsky_sim, sim_cfg.freq_resol)
        self.eval_instrument(sim_cfg.ndet_sim, sim_cfg.freq_resol)
        self.eval_sensitivities()
        save_summary = sim_cfg.save_summary
        if max(sim_cfg.nsky_sim, 1) * max(sim_cfg.ndet_sim, 1) == 1:
            save_summary = False
        self.make_tables(
            basename,
            save_summary=save_summary,
            save_sim=sim_cfg.save_sim,
            save_cfg=sim_cfg.save_optical,
        )
