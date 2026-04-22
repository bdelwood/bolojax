"""Command-line interface for bolojax."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr
import yaml

from bolojax import ExperimentConfig


def main() -> None:
    """Entry point for the bolojax CLI."""
    parser = argparse.ArgumentParser(description="Bolometric sensitivity calculator")
    parser.add_argument(
        "-i", "--input", default=None, required=True, help="Input configuration file"
    )
    parser.add_argument("-o", "--output", default=None, help="Output file (.nc)")

    args = parser.parse_args()

    with Path(args.input).open() as f:
        dd = yaml.safe_load(f)

    config = ExperimentConfig(**dd)

    datasets = []
    for cam_name, camera in config.instrument.cameras.items():
        for chan_name in camera.channels:
            experiment = config.setup(cam_name, chan_name)
            ds = experiment.to_dataset()
            ds = ds.expand_dims(channel=[f"{cam_name}_{chan_name}"])
            datasets.append(ds)

            # Print summary
            sys.stdout.write(f"{cam_name}_{chan_name} ---------\n")
            for var in ["effic", "opt_power", "P_sat", "NET", "NET_arr"]:
                if var in ds:
                    val = ds[var].values
                    unit = ds[var].attrs.get("units", "")
                    sys.stdout.write(f"  {var:20s}: {float(val.mean()):12.4f} {unit}\n")
            sys.stdout.write("---------\n")

    if args.output:
        xr.concat(datasets, dim="channel").to_netcdf(args.output)


if __name__ == "__main__":
    main()
