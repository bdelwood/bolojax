"""Command-line interface for bolojax."""

import argparse
from pathlib import Path

import yaml

from bolojax import Experiment


def main():
    """Entry point for the bolojax CLI."""
    parser = argparse.ArgumentParser(description="Bolometric sensitivity calculator")
    parser.add_argument(
        "-i", "--input", default=None, required=True, help="Input configuration file"
    )
    parser.add_argument("-o", "--output", default=None, help="Output file")

    args = parser.parse_args()

    with Path(args.input).open() as f:
        dd = yaml.safe_load(f)

    experiment = Experiment(**dd)
    experiment.run()
    experiment.instrument.print_summary()
    experiment.instrument.print_optical_output()

    if args.output:
        experiment.instrument.write_tables(args.output)


if __name__ == "__main__":
    main()
