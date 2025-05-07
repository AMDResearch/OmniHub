import argparse
import pathlib
import re
import subprocess
import sys
import time

import yaml

from omnihub.generate.app_config import generate_app_config
from omnihub.generate.job import default_rocm_version, generate_job, load_tool_config


def generate_sweep(
    config_dir,
    omnihub_dir,
    sweep_dir,
    template,
    cluster="hpcfund",
    partitions=None,
    num_nodes=[1],
    platform="apptainer",
    rocm_version=default_rocm_version,
    runner=None,
    tools=None,
    time_limit="1h",
    delay=60,
    dry_run=False,
):
    tools_dir = f"{config_dir}/tools"
    tool_config = load_tool_config(tools_dir)
    tool_names = set(tool_config.keys())

    cluster_file = f"{config_dir}/{cluster}.yaml"
    with open(cluster_file, "r") as f:
        cluster_info = yaml.safe_load(f)["cluster"]

    if partitions is None:
        partitions = []

    if tools is None:
        tools = []

    if len(partitions) == 0:
        default_partition = cluster_info["subsets"][0]["partition"]
        partitions.append(default_partition)

    # If no tools are provided, make sure there's at least an empty list to
    # ensure the generation loop works as expected.
    if len(tools) == 0:
        tools.append([])

    # Maintain a list of dicts for serialization, and an equivalent set of
    # tuples to check whether a particular job has been submitted.
    submitted_list = []
    submitted_set = set()
    submitted_file = f"{sweep_dir}/submitted.yaml"

    config_files = list(pathlib.Path(sweep_dir).glob("config-*.yaml"))
    num_generated = len(config_files)
    if num_generated == 0:
        _, num_generated = generate_app_config(sweep_dir, template)
        print(f"Starting a new sweep")
        print(f".. Generated configurations: {num_generated}")

    num_submissions = len(partitions) * len(num_nodes) * len(tools) * num_generated
    print(f"Number of jobs in this sweep: {num_submissions}")

    # If this is an existing sweep, load the list of submitted jobs.
    if pathlib.Path(submitted_file).exists():
        with open(submitted_file, "r") as f:
            submitted_list = yaml.safe_load(f)
            for s in submitted_list:
                submitted_set.add(
                    (s["partition"], s["num_nodes"], frozenset(s["tools"]), s["config"])
                )

        num_remaining = num_submissions - len(submitted_set)
        print(f"Restoring interrupted sweep")
        print(f".. Number of submitted jobs: {len(submitted_set)}")
        print(f".. Number of remaining jobs: {num_remaining}")

    config_files = list(pathlib.Path(sweep_dir).glob("config-*.yaml"))

    for p in partitions:
        for n in num_nodes:
            for t in tools:
                toolset = ",".join(t) if len(t) > 0 else "-"

                for c in config_files:
                    submission_tuple = (p, n, frozenset(t), c.name)
                    if submission_tuple in submitted_set:
                        continue

                    print(f"Submitting job: {p}/{n}/{toolset}/{c.name}")
                    generate_job(
                        config_dir=config_dir,
                        omnihub_dir=omnihub_dir,
                        app_config=c,
                        app_args="",
                        cluster=cluster,
                        partition=p,
                        num_nodes=n,
                        platform=platform,
                        rocm_version=rocm_version,
                        runner=runner,
                        tools=list(t),
                        time_limit=time_limit,
                        output=f"{sweep_dir}/job.sh",
                    )

                    if dry_run:
                        continue

                    result = subprocess.run(
                        ["sbatch", "--parsable", f"{sweep_dir}/job.sh"],
                        capture_output=True,
                        text=True,
                    )
                    if not result.returncode == 0:
                        print("Error submitting job")
                        sys.exit(1)

                    if not result.stdout.strip().isdigit():
                        print("Unexpected sbatch output")
                        sys.exit(1)

                    job_id = int(result.stdout.strip())
                    print(f"Submitted job ID {job_id}")

                    submission_dict = {
                        "partition": p,
                        "num_nodes": n,
                        "tools": list(t),
                        "config": c.name,
                        "job_id": job_id,
                    }
                    submitted_list.append(submission_dict)
                    submitted_set.add(submission_tuple)

                    with open(submitted_file, "w") as f:
                        yaml.dump(submitted_list, f)

                    job_path = pathlib.Path(f"{sweep_dir}/job.sh")
                    job_path.unlink()

                    time.sleep(delay)


def main():
    # Paths to YAML configuration files.
    script_path = pathlib.Path(__file__).parent.parent.parent.resolve()
    config_dir = f"{script_path}/config"
    tools_dir = f"{config_dir}/tools"

    tool_config = load_tool_config(tools_dir)
    tool_names = set(tool_config.keys())

    parser = argparse.ArgumentParser(
        description="Run OmniHub sweeps",
        prog="omnihub-sweep",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30
        ),
    )
    parser._optionals.title = "Help"
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--omnihub-dir",
        help="Path to OmniHub",
        type=str,
        required=True,
        metavar="",
    )
    required_group.add_argument(
        "--sweep-dir",
        help="Sweep directory",
        type=str,
        required=True,
        metavar="",
    )
    required_group.add_argument(
        "--template",
        help="Application configuration template",
        type=str,
        required=True,
        metavar="",
    )
    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--cluster",
        help="Name of the cluster:\n\thpcfund (default)\n\tradha",
        type=str,
        default="hpcfund",
        metavar="",
    )
    optional_group.add_argument(
        "--partitions",
        help="List of cluster partitions",
        type=str,
        nargs="+",
        default=None,
        metavar="",
    )
    optional_group.add_argument(
        "--num-nodes",
        help="List of number of nodes (default: [1])",
        type=int,
        nargs="+",
        default=[1],
        metavar="",
    )
    optional_group.add_argument(
        "--platform",
        help="Container platform to execute in:\n\tapptainer (default)\n\tdocker",
        type=str,
        required=False,
        default="apptainer",
        metavar="",
    )
    optional_group.add_argument(
        "--runner",
        help="Distributed runner. Required for multiple nodes.\n\tmanual\n\ttorchrun",
        type=str,
        required=False,
        metavar="",
    )
    optional_group.add_argument(
        "--tools",
        help="List of list of tools. Choose from:\n\t{}".format(
            "\n\t".join(sorted(tool_names))
        ),
        type=str,
        nargs="+",
        action="append",
        default=None,
        metavar="",
    )
    optional_group.add_argument(
        "--time-limit",
        help="Time limit for the SLURM job as an integer followed by a time unit (default: 1h). Examples: 120s, 30m, 5h.",
        type=str,
        default="1h",
        metavar="",
    )
    optional_group.add_argument(
        "--delay",
        help="Seconds between job submissions",
        type=int,
        default=60,
        metavar="",
    )
    optional_group.add_argument(
        "--dry-run",
        help="Generate sweep without submitting the jobs",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    args_dict = vars(args)

    generate_sweep(config_dir, **args_dict)


if __name__ == "__main__":
    main()
