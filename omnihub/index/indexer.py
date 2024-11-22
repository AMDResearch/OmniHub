import argparse
import os
import pathlib

import pandas
import yaml

formats = {"csv"}

ignore_fields = [
    ("job", "omnihub-directory"),
    ("job", "model-directory"),
    ("job", "model-directory"),
    ("job", "nodes"),
    ("job", "head-node"),
    ("job", "head-address"),
    ("job", "head-port"),
]


def load_tabular_data(results_dir):
    """
    Loads tabular data from job executions in the given results directory.
    Tabular data is available for processed job executions as flat YAML files
    under the `processed-data` subdirectory. Job executions that haven't been
    processed yet are ignored.

    Tabular data combines: job information, configuration, high-level tool
    data, and results.

    Args:
    - results_dir (str): Path to OmniHub results directory.

    Returns:
    - df: A pandas dataframe with tabular data for job executions; columns
      identify the source and name of key indicators, and rows represent
      different job executions.
    """
    rows = []
    for execution in pathlib.Path(results_dir).glob("*/job.sh"):
        job_id = execution.parent.name
        processed_dir = f"{results_dir}/{job_id}/processed-data"

        tabular_data = []
        for tabular_file in pathlib.Path(processed_dir).glob("*.yaml"):
            data_source = tabular_file.stem
            with open(tabular_file, "r") as f:
                records = yaml.safe_load(f)
            df = pandas.DataFrame.from_records([records])
            df.columns = [[data_source] * len(df.columns), df.columns]
            tabular_data.append(df)

        if len(tabular_data):
            row = pandas.concat(tabular_data, axis=1)
            rows.append(row)

    df = pandas.concat(rows)
    df.drop(ignore_fields, axis=1, inplace=True)

    # Sort the first level of columns to ensure job and tools are always
    # first; remaining columns are sorted in alphabetical order.
    data_sources = set(df.columns.get_level_values(0))
    first = ["job", "tools"]
    remaining = [c for c in data_sources if c not in set(first)]
    sorted_columns = first + sorted(remaining)
    df = df.reindex(columns=df.columns.reindex(sorted_columns, level=0)[0])

    df.reset_index(drop=True, inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Index OmniHub results directory",
        prog="omnihub-index",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30
        ),
    )
    parser._optionals.title = "Help"
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--results-dir",
        help="Path to OmniHub results",
        type=str,
        required=True,
        metavar="",
    )
    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--format",
        help="Output format:\n\t{}".format("\n\t".join(sorted(formats))),
        type=str,
        default="csv",
        metavar="",
    )
    optional_group.add_argument(
        "--output",
        help="Base output file or directory name (default: index)",
        type=str,
        default="index",
        metavar="",
    )

    args = parser.parse_args()

    df = load_tabular_data(args.results_dir)

    if args.format == "csv":
        df.to_csv(f"{args.output}.csv")


if __name__ == "__main__":
    main()
