import argparse
import logging
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

from omnihub.process import parsers

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def process_execution(execution_dir):
    parser_registry = []

    processed_dir = f"{execution_dir}/processed-data"
    os.makedirs(processed_dir, exist_ok=True)

    parser_registry.append(parsers.JobConfigParser(execution_dir))
    parser_registry.append(parsers.AppLogParser(execution_dir))

    if os.path.isfile(f"{execution_dir}/job-report.yaml"):
        parser_registry.append(parsers.JobReportParser(execution_dir))

    if os.path.isfile(f"{execution_dir}/app.yaml"):
        parser_registry.append(parsers.AppConfigParser(execution_dir))

    if os.path.isdir(f"{execution_dir}/tools/omnihub-monitor"):
        parser_registry.append(parsers.OmnihubMonitorParser(execution_dir))

    if os.path.isdir(f"{execution_dir}/sysinfo"):
        parser_registry.append(parsers.SysInfoParser(execution_dir))

    if os.path.isdir(f"{execution_dir}/tools/omnistat"):
        parser_registry.append(parsers.OmnistatReportParser(execution_dir))

    if os.path.isdir(f"{execution_dir}/tools/pytorch-trace"):
        parser_registry.append(parsers.PytorchTraceParser(execution_dir))

    if os.path.isdir(f"{execution_dir}/tools/rocprofv3-stats"):
        parser_registry.append(parsers.RocprofStatsParser(execution_dir))

    for parser in parser_registry:
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
        metavar="",
    )
    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-j",
        "--jobs",
        help="Number of parallel jobs to run (default: 1)",
        type=int,
        default=1,
        metavar="",
    )
    optional_group.add_argument(
        "--force",
        help="Process execution results that have already been processed",
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

    print(f"Found {len(execution_dirs)} executions to process")

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for execution_dir in execution_dirs:
            executor.submit(process_execution, execution_dir)


if __name__ == "__main__":
    main()
