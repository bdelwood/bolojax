"""Experiment-level configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import model_validator

from bolojax.compute.experiment import Experiment
from bolojax.io.sensitivity import build_experiment
from bolojax.models.base import BolojaxModel
from bolojax.models.instrument import InstrumentConfig
from bolojax.models.sky import Universe
from bolojax.models.utils import set_config_dir


class SimConfig(BolojaxModel):
    """Simulation configuration."""

    nsky_sim: int = 0
    ndet_sim: int = 0
    freq_resol: float | None = None
    config_dir: str = str(Path("..") / "config")

    @model_validator(mode="after")
    def _init_derived(self) -> SimConfig:
        set_config_dir(self.config_dir)
        return self


class ExperimentConfig(BolojaxModel):
    """Configuration for a bolometer sensitivity experiment.

    Parse from YAML with ``ExperimentConfig.from_yaml(path)``, then
    call ``.setup()`` for a single-channel ``Experiment`` or
    ``.setup_all()`` for all channels.
    """

    sim_config: SimConfig
    universe: Universe
    instrument: InstrumentConfig

    @model_validator(mode="after")
    def _init_derived(self) -> ExperimentConfig:
        self.universe.atmosphere.set_telescope(self.instrument)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load config from a YAML file."""
        path = Path(path)
        with path.open() as f:
            config = yaml.safe_load(f)
        return cls.model_validate(config)

    def _eval(self) -> None:
        """Run sky and instrument evaluation (idempotent setup)."""
        sim = self.sim_config
        self.instrument.eval_sky(self.universe, sim.nsky_sim, sim.freq_resol)
        self.instrument.eval_instrument(sim.ndet_sim, sim.freq_resol)

    def setup(
        self, camera: str | None = None, channel: str | None = None
    ) -> Experiment:
        """Run setup and return an ``Experiment`` compute object.

        Args:
            camera: camera name. Defaults to the first camera.
            channel: channel name. Defaults to the first channel.
        """
        self._eval()
        cams = self.instrument.cameras
        cam = cams[camera] if camera else next(iter(cams.values()))
        chs = cam.channels
        ch = chs[channel] if channel is not None else next(iter(chs.values()))
        return build_experiment(ch)
