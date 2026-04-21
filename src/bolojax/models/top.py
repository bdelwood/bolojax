"""Top level configuration."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .instrument import Instrument
from .sky import Universe
from .utils import set_config_dir


class SimConfig(BaseModel):
    """Simulation configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    nsky_sim: int = 0
    ndet_sim: int = 0
    save_summary: bool = True
    save_sim: bool = True
    save_optical: bool = True
    freq_resol: float | None = None
    config_dir: str = str(Path("..") / "config")

    def model_post_init(self, __context):
        set_config_dir(self.config_dir)


class Top(BaseModel):
    """Top level configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    sim_config: SimConfig
    universe: Universe
    instrument: Instrument

    def model_post_init(self, __context):
        self.universe.atmosphere.set_telescope(self.instrument)

    def run(self):
        """Run the entire analysis."""
        self.instrument.run(self.universe, self.sim_config)
