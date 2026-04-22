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
        experiment = bolojax.Experiment(**dd)
        experiment.sim_config.ndet_sim = 0
        experiment.sim_config.nsky_sim = 0
        experiment.instrument.custom_atm_file = "Bands/atacama_atm.txt"
        experiment.run()
        experiment.instrument.print_summary()
        experiment.instrument.write_tables("test.fits")

    def test_run_multi(self):
        with CONFIG_PATH.open() as f:
            dd = yaml.safe_load(f)
        dd["sim_config"]["config_dir"] = "config"
        experiment = bolojax.Experiment(**dd)
        experiment.sim_config.ndet_sim = 10
        experiment.sim_config.nsky_sim = 10
        experiment.instrument.custom_atm_file = "Bands/atacama_atm.txt"
        experiment.run()
        experiment.instrument.print_summary()
        experiment.instrument.write_tables("test.fits")


if __name__ == "__main__":
    unittest.main()
