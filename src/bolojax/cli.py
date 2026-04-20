"""Command-line interface for bolojax."""

import argparse

import yaml

from bolojax import Top


def main():
    """Entry point for the bolojax CLI."""
    parser = argparse.ArgumentParser(description="Bolometric sensitivity calculator")
    parser.add_argument(
        "-i", "--input", default=None, required=True, help="Input configuration file"
    )
    parser.add_argument("-o", "--output", default=None, help="Output file")

    args = parser.parse_args()

    with open(args.input) as f:
        dd = yaml.safe_load(f)

    top = Top(**dd)
    top.run()
    top.instrument.print_summary()
    top.instrument.print_optical_output()

    if args.output:
        top.instrument.write_tables(args.output)


if __name__ == "__main__":
    main()
