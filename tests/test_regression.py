"""Regression tests against stored reference values."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import bolojax

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")

with (FIXTURES / "reference.json").open() as f:
    REFERENCE = json.load(f)

CONFIGS = [
    ("example.yaml", "example"),
    ("comprehensive.yaml", "comprehensive"),
]


def run_bolojax(config_name: str) -> dict[str, dict]:
    with (FIXTURES / config_name).open() as f:
        cfg = yaml.safe_load(f)
    cfg["sim_config"]["config_dir"] = CONFIG_DIR

    config = bolojax.ExperimentConfig(**cfg)
    results = {}
    # unpack same format as reference json
    for cam_name, camera in config.instrument.cameras.items():
        for chan_name in camera.channels:
            experiment = config.setup(cam_name, chan_name)
            r = experiment.compute()
            prefix = config_name.removesuffix(".yaml")
            results[f"{prefix}:{cam_name}_{chan_name}"] = {
                "net": float(r.NET.squeeze()) * 1e6,
                "net_corr": float(r.NET_corr.squeeze()) * 1e6,
                "corr_fact": float(r.corr_fact.squeeze()),
                "opt_power": float(r.opt_power.squeeze()) * 1e12,
                "effic": float(r.effic.squeeze()),
                "NEP_bolo": float(r.NEP_bolo.squeeze()) * 1e18,
                "NEP_ph": float(r.NEP_ph.squeeze()) * 1e18,
                "NEP_read": float(r.NEP_read.squeeze()) * 1e18,
            }
    return results


@pytest.fixture(params=CONFIGS, ids=[c[1] for c in CONFIGS])
def comparison(request):
    config_name, _ = request.param
    prefix = config_name.removesuffix(".yaml")
    bj = run_bolojax(config_name)
    ref = {k: v for k, v in REFERENCE.items() if k.startswith(prefix)}
    channels = sorted(set(bj) & set(ref))
    return bj, ref, channels


def assert_close(bj, ref, channels, field, rtol):
    for ch in channels:
        bj_val = bj[ch][field]
        ref_val = ref[ch][field]
        rel = abs(bj_val - ref_val) / max(abs(ref_val), 1e-30)
        assert rel < rtol, (
            f"{ch} {field}: got={bj_val:.6f} expected={ref_val:.6f} "
            f"rel={rel:.2e} > {rtol:.0e}"
        )


def test_net(comparison):
    bj, ref, ch = comparison
    assert_close(bj, ref, ch, "net", 0.001)


def test_corr_fact(comparison):
    bj, ref, ch = comparison
    assert_close(bj, ref, ch, "corr_fact", 0.001)


def test_efficiency(comparison):
    bj, ref, ch = comparison
    assert_close(bj, ref, ch, "effic", 0.001)


def test_optical_power(comparison):
    bj, ref, ch = comparison
    assert_close(bj, ref, ch, "opt_power", 0.001)


def test_nep_bolo(comparison):
    bj, ref, ch = comparison
    assert_close(bj, ref, ch, "NEP_bolo", 0.001)


def test_nep_photon(comparison):
    bj, ref, ch = comparison
    assert_close(bj, ref, ch, "NEP_ph", 0.001)
