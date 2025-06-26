import argparse
import logging
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import yaml

from omnihub.process import parsers

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def process_execution(execution_dir, strict_exit_code=False):
    # Only check the exit_code if strict_exit_code is True
    if strict_exit_code:
        with open(f"{execution_dir}/job-status.yaml", "r") as f:
            data = yaml.safe_load(f)
        if data.get("exit_code") != 0:
            logging.warning(
                f"Skipped processing job in {execution_dir} because it did not complete successfully. Exit code: {data.get('exit_code')}"
            )
            return

    parser_registry = []

    processed_dir = f"{execution_dir}/processed-data"
    os.makedirs(processed_dir, exist_ok=True)

    parser_registry.append(parsers.JobConfigParser(execution_dir))
    parser_registry.append(parsers.AppLogParser(execution_dir))

    if os.path.isfile(f"{execution_dir}/job-status.yaml"):
        parser_registry.append(parsers.JobStatusParser(execution_dir))

    if os.path.isfile(f"{execution_dir}/app.yaml"):
        parser_registry.append(parsers.AppConfigParser(execution_dir))

    if os.path.isdir(f"{execution_dir}/tools/omnihub-monitor"):
        parser_registry.append(parsers.OmnihubMonitorParser(execution_dir))

    # TODO: Enable sys info parser when we are able to handle more complex
    # data. It's currently disabled because the data is hierarchical with a
    # lot of detail that doesn't translate well when flattened.
    # if os.path.isdir(f"{execution_dir}/sysinfo"):
    #     parser_registry.append(parsers.SysInfoParser(execution_dir))

    for omnistat_variant in pathlib.Path(f"{execution_dir}/tools").glob("omnistat*"):
        if omnistat_variant.is_dir():
            name = omnistat_variant.name
            parser_registry.append(parsers.OmnistatReportParser(execution_dir, name))
            parser_registry.append(parsers.OmnistatParser(execution_dir, name))
            parser_registry.append(parsers.OmnistatRangeParser(execution_dir, name))

    if os.path.isdir(f"{execution_dir}/tools/pytorch-trace"):
        parser_registry.append(parsers.PytorchTraceParser(execution_dir))

    if os.path.isdir(f"{execution_dir}/tools/rccl-info"):
        parser_registry.append(parsers.RcclInfoParser(execution_dir))

    if os.path.isdir(f"{execution_dir}/tools/rocprofv3-stats"):
        parser_registry.append(parsers.RocprofStatsParser(execution_dir))

    # Create a job performance report based on what other tools generate
    parser_registry.append(parsers.ReportCardParser(execution_dir))

    # Create a hash of the execution directory to uniquely identify groups of related executions.
    parser_registry.append(parsers.HashParser(execution_dir))

    for parser in parser_registry:
        parser.load()
        parser.parse()


def main():
    parser = argparse.ArgumentParser(
        description="Process OmniHub execution",
        prog="omnihub-process",
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
    )
    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-j",
        "--jobs",
        help="Number of parallel jobs to run (default: 1)",
        type=int,
        default=1,
    )
    optional_group.add_argument(
        "--force",
        help="Process execution results that have already been processed",
        action="store_true",
        default=False,
    )
    optional_group.add_argument(
        "--successful-executions",
        help=(
            "Process only successful executions based on their exit code. "
            "Executions with a non-zero exit code will be skipped."
        ),
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    execution_dirs = []
    for job in pathlib.Path(args.results_dir).rglob("job.sh"):
        execution_dir = job.parent
        processed_dir = f"{execution_dir}/processed-data"
        if not os.path.isdir(processed_dir) or args.force:
            execution_dirs.append(execution_dir)

    print(f"Found {len(execution_dirs)} executions to process.")

    futures = []
    # Use ProcessPoolExecutor to process each execution directory in parallel
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                process_execution, execution_dir, args.successful_executions
            )
            for execution_dir in execution_dirs
        ]
    # Ensure all executions complete
    for future in futures:
        future.result()

    # After processing, count executions where "processed-data" folder exists
    success_count = 0
    for execution_dir in execution_dirs:
        processed_dir = f"{execution_dir}/processed-data"
        if not os.path.isdir(processed_dir):
            logging.warning(f"Processed data directory not found: {processed_dir}")
        else:
            success_count += 1
    failure_count = len(execution_dirs) - success_count
    print(
        f"Processed data directories found for {success_count} executions and missing for {failure_count} executions."
    )


if __name__ == "__main__":
    main()
