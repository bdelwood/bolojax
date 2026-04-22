"""Experiment-level configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import model_validator

from bolojax.io.sensitivity import build_params
from bolojax.models.base import BolojaxModel
from bolojax.models.instrument import Instrument
from bolojax.models.sky import Universe
from bolojax.models.utils import set_config_dir


class SimConfig(BolojaxModel):
    """Simulation configuration."""

    nsky_sim: int = 0
    ndet_sim: int = 0
    save_summary: bool = True
    save_sim: bool = True
    save_optical: bool = True
    freq_resol: float | None = None
    config_dir: str = str(Path("..") / "config")

    @model_validator(mode="after")
    def _init_derived(self) -> SimConfig:
        set_config_dir(self.config_dir)
        return self


class Experiment(BolojaxModel):
    """A complete bolometer sensitivity experiment."""

    sim_config: SimConfig
    universe: Universe
    instrument: Instrument

    @model_validator(mode="after")
    def _init_derived(self) -> Experiment:
        self.universe.atmosphere.set_telescope(self.instrument)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        """Load an experiment from a YAML config file."""
        path = Path(path)
        with path.open() as f:
            config = yaml.safe_load(f)
        return cls.model_validate(config)

    def run(self) -> None:
        """Run the entire analysis."""
        self.instrument.run(self.universe, self.sim_config)

    def setup(self, camera: str | None = None, channel: str | None = None) -> tuple:
        """Run setup and extract JAX pytrees for a single channel.

        Calls ``eval_sky`` and ``eval_instrument``, then returns
        ``(OpticsState, BoloParams, elem_names)`` ready for
        :func:`~bolojax.compute.sensitivity.compute_sensitivity`.

        Args:
            camera: camera name. If ``None``, uses the first camera.
            channel: channel index. If ``None``, uses the first channel.
        """
        sim = self.sim_config
        self.instrument.eval_sky(self.universe, sim.nsky_sim, sim.freq_resol)
        self.instrument.eval_instrument(sim.ndet_sim, sim.freq_resol)

        cams = self.instrument.cameras
        cam = cams[camera] if camera else next(iter(cams.values()))
        chs = cam.channels
        ch = chs[channel] if channel is not None else next(iter(chs.values()))
        return build_params(ch)
