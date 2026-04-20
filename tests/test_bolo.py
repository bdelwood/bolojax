"""Unit tests for bolojax package."""

import unittest
from pathlib import Path

import yaml

import bolojax

CONFIG_PATH = Path("config/myExample.yaml")


class ExampleTestCase(unittest.TestCase):
    def test_run_single(self):
        with CONFIG_PATH.open() as f:
            dd = yaml.safe_load(f)
        dd["sim_config"]["config_dir"] = "config"
        top = bolojax.Top(**dd)
        top.sim_config.ndet_sim = 0
        top.sim_config.nsky_sim = 0
        top.instrument.custom_atm_file = "Bands/atacama_atm.txt"
        top.run()
        top.instrument.print_summary()
        top.instrument.write_tables("test.fits")

    def test_run_multi(self):
        with CONFIG_PATH.open() as f:
            dd = yaml.safe_load(f)
        dd["sim_config"]["config_dir"] = "config"
        top = bolojax.Top(**dd)
        top.sim_config.ndet_sim = 10
        top.sim_config.nsky_sim = 10
        top.instrument.custom_atm_file = "Bands/atacama_atm.txt"
        top.run()
        top.instrument.print_summary()
        top.instrument.write_tables("test.fits")


if __name__ == "__main__":
    unittest.main()
