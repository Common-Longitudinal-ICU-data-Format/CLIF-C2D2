"""CLIF-to-C2D2 ETL CLI entry point."""

import argparse

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="CLIF-to-C2D2 ETL: Extract C2D2 common data elements from CLIF tables"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
