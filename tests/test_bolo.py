"""Unit tests for bolojax package."""

import unittest
from pathlib import Path

import yaml

import bolojax

CONFIG_PATH = Path("config/example.yaml")


class ExampleTestCase(unittest.TestCase):
    def test_run_single(self):
        with CONFIG_PATH.open() as f:
            dd = yaml.safe_load(f)
        dd["sim_config"]["config_dir"] = "config"
        dd["sim_config"]["ndet_sim"] = 0
        dd["sim_config"]["nsky_sim"] = 0
        config = bolojax.ExperimentConfig(**dd)
        config.instrument.custom_atm_file = "Bands/atacama_atm.txt"
        experiment = config.setup()
        result = experiment.compute()
        assert result.NET.squeeze() > 0

    def test_run_multi(self):
        with CONFIG_PATH.open() as f:
            dd = yaml.safe_load(f)
        dd["sim_config"]["config_dir"] = "config"
        dd["sim_config"]["ndet_sim"] = 10
        dd["sim_config"]["nsky_sim"] = 10
        config = bolojax.ExperimentConfig(**dd)
        config.instrument.custom_atm_file = "Bands/atacama_atm.txt"
        for cam_name, camera in config.instrument.cameras.items():
            for chan_name in camera.channels:
                experiment = config.setup(cam_name, chan_name)
                result = experiment.compute()
                assert result.NET.squeeze().shape[0] > 0

    def test_to_dataset(self):
        with CONFIG_PATH.open() as f:
            dd = yaml.safe_load(f)
        dd["sim_config"]["config_dir"] = "config"
        dd["sim_config"]["ndet_sim"] = 0
        dd["sim_config"]["nsky_sim"] = 0
        config = bolojax.ExperimentConfig(**dd)
        config.instrument.custom_atm_file = "Bands/atacama_atm.txt"
        experiment = config.setup()
        ds = experiment.to_dataset()
        assert "NET" in ds
        assert "element" in ds.elem_power_to_det.dims


if __name__ == "__main__":
    unittest.main()
