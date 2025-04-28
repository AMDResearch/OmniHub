import argparse
import datetime
import os
import pathlib
import re
import sys
from string import Template

import yaml

# Omnihub main script
script = "omnihub-run"

# Sets of options supported by the generator.
platforms = {"apptainer", "docker"}
runners = {"manual", "torchrun"}

# Sets of supported GPU architectures.
gpu_mapping = {
    "mi210": "gfx90a",
    "mi250": "gfx90a",
    "mi300": "gfx942",
}

# Sets of supported ROCm versions.
rocm_versions = {"6.2.0", "6.3.1"}
default_rocm_version = "6.3.1"

seconds_per_unit = {"s": 1, "m": 60, "h": 3600}


def load_tool_config(tools_dir):
    """
    Load tool YAML configuration files.
    """
    tools = {}
    for i in pathlib.Path(tools_dir).glob("*.yaml"):
        source = i.stem
        with open(i, "r") as f:
            config = yaml.safe_load(f)
            conflicts = config.get("conflicts", [])
            for tool in config["tools"]:
                name = tool["name"]
                tool["source"] = source
                tool["conflicts"] = conflicts
                tools[name] = tool

    return tools


# Generate a SLURM job file to run Omnihub models
#
# This function uses the given arguments to generate a single SLURM job file
# and print it to stdout. The generated scripts are tailored to be executed in
# a particular cluster, and details about each supported cluster are described
# in YAML files in the configuration directory, e.g. `config/hpcfund.yaml`.
#
# The main command to be executed by the SLURM job is generated starting with
# the innermost Python script and then building around it as follows:
#  1. Python script (`base_command`) and additional arguments (`base_args`).
#  2. Prefixes to the Python script (`base_prefix`), which can include certain
#     runners like torchrun, or profilers like omniperf.
#  3. Container wrappers, to create the right execution environment for
#     container platforms like Apptainer (`apptainer_wrap`) and Docker
#     (`docker_wrap`).
#
# In addition to the main command to be executed, there are other ways to
# configure the execution, including:
#  - Commands to be executed before and after the main command: `pre_execute`
#    and `post_execute`. These can be useful to set up certain profiling tools
#    like Omnistat.
#  - Environment variables, included in `environment`. These can be used to
#    configure certain paths, like Omnitrace's output directory, and are
#    automatically reformatted for different contexts.
def generate_job(
    config_dir,
    omnihub_dir,
    app_config,
    app_args,
    num_nodes,
    partition,
    rocm_version,
    platform,
    cluster,
    runner,
    tools,
    time,
):
    template_file = f"{config_dir}/job.template"

    tools_dir = f"{config_dir}/tools"
    tool_config = load_tool_config(tools_dir)
    tool_names = set(tool_config.keys())

    # Add "omnihub-monitor" to the list of tools if not already present
    if "omnihub-monitor" not in tools:
        tools.append("omnihub-monitor")

    cluster_file = f"{config_dir}/{cluster}.yaml"

    # Lists of commands to execute.
    pre_execute = []
    execute = []
    post_execute = []

    environment = []

    base_prefix = []
    base_args = []
    base_command = (
        "{omnihub_dir}/{base_script} {base_args}"
        " --output-dir={results_dir}"
        " --app-config={omnihub_dir}/{app_config}"
        f" {app_args}"
        " > {results_dir}/logs/srun-\\$SLURM_PROCID.out"
        " 2> {results_dir}/logs/srun-\\$SLURM_PROCID.err"
    )

    manual_args = [
        "--manual-launch-ddp",
        "--master-addr=$head_host",
        "--master-port=$head_port",
        "--rank=\\$SLURM_PROCID",
        "--world-size=$SLURM_NTASKS",
    ]

    torchrun_prefix = (
        "torchrun"
        " --nnodes=\\$SLURM_JOB_NUM_NODES"
        " --nproc_per_node=$num_gpus_per_node"
        " --master_addr=$head_host"
        " --master_port=$head_port"
        " --node-rank=\\$SLURM_PROCID"
        " --log-dir {results_dir}/logs"
        " --redirect 3"
    )

    apptainer_wrap = (
        "srun apptainer run $shared_dir/apptainer/omnihub.{gpu_arch}.{rocm_version}.sif"
        ' /bin/bash -c "export ROCM_PATH=/opt/rocm; export ROCM_LIB=/opt/rocm/lib;'
        " export LD_LIBRARY_PATH=/opt/rocm/lib:\\$LD_LIBRARY_PATH;"
        " export PATH=\\$CONDA_DIR/bin:\\$PATH;"
        ' {base_command}"'
    )

    docker_wrap = (
        'srun bash -c "docker run --rm {docker_args}'
        " -v $omnihub_dir:$docker_omnihub_dir:ro"
        " -v $models_dir:$docker_models_dir:ro"
        " -v $shared_dir:$docker_shared_dir:ro"
        " -v $results_dir:$docker_results_dir"
        " -w $docker_results_dir"
        " --env-file=<(env | grep -E 'SLURM|NCCL|OMNIHUB')"
        " --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        " --device=/dev/kfd --device=/dev/dri"
        " --network=host --ipc=host --shm-size 8G"
        " docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:{gpu_arch}.{rocm_version}"
        ' bash -c \\"{base_command}; fix-host-owner $docker_results_dir\\""'
    )

    # Temporarily using these per-cluster commands for testing. These are
    # candidates to be included in the cluster-specific configuration files, but
    # it's still not clear how these should be exposed, or if we can find another
    # option that works in all clusters.
    head_host_commands = {
        "hpcfund": "$(ip -f inet addr show eth0 | awk '/inet / {print $2}' | cut -d/ -f1)",
        "radha": "$(dig +short $(hostname).amd.com)",
    }

    # STEP 1. Basic validation of command line options.

    if not os.path.isdir(config_dir):
        print(f"Unable to find configuration directory: {config_dir}")
        sys.exit(1)

    if not os.path.isfile(template_file):
        print(f"Unable to find job template: {template_file}")
        sys.exit(1)

    if not os.path.isfile(cluster_file):
        print(f"Unable to find cluster configuration: {cluster_file}")
        sys.exit(1)

    if runner and runner not in runners:
        print(f"Unsupported distributed runner: {runner}")
        sys.exit(1)

    if num_nodes > 1 and not runner:
        print(f"Multi-node execution requires a distributed runner")
        sys.exit(1)

    args_tools_names = set(tools)
    if not args_tools_names.issubset(tool_names):
        unsupported = args_tools_names - tool_names
        print(f"Unsupported tools: {', '.join(unsupported)}")
        sys.exit(1)

    args_tools_sources = set()
    for tool in tools:
        source = tool_config[tool]["source"]
        if source in args_tools_sources:
            print(f"Unable to use multiple variants of the same tool: {source}")
            sys.exit(1)
        args_tools_sources.add(source)

    args_tools_all = args_tools_names | args_tools_sources
    for tool in tools:
        conflicts = set(tool_config[tool]["conflicts"])
        intersection = conflicts & args_tools_all
        if len(intersection) > 0:
            print(f"Incompatible tool combination: {tool}, {', '.join(intersection)}")
            sys.exit(1)

    time_pattern = f"[0-9]+[{''.join(seconds_per_unit.keys())}]"
    if not re.fullmatch(time_pattern, time):
        print(f"Unexpected time limit format: {time}")
        sys.exit(1)

    time_limit_seconds = int(time[:-1]) * seconds_per_unit[time[-1]]
    time_limit_delta = datetime.timedelta(seconds=time_limit_seconds)

    # STEP 2. Cluster-related validation, configuration and node selection.

    with open(cluster_file, "r") as f:
        cluster_info = yaml.safe_load(f)["cluster"]

    container_platforms = cluster_info["container-platforms"]
    models_dir = cluster_info["models-dir"]
    shared_dir = cluster_info["shared-dir"]
    results_dir = cluster_info["results-dir"]
    partitions = {x["partition"]: x for x in cluster_info["subsets"]}

    cluster_name = cluster_info["name"]
    if not set(container_platforms).issubset(set(platforms)):
        print(
            f"Unsupported container_platforms {set(container_platforms) - set(platforms)} in {cluster_file}"
        )
        sys.exit(1)

    # If platform argument is not present, default to first container platform in the cluster.
    platform = platform if platform else container_platforms[0]
    if platform not in container_platforms:
        print(f"Unsupported platform {platform} in the cluster {cluster_name}")
        sys.exit(1)

    # If partition argument is not present, default to first partition in the cluster.
    partition_name = partition
    if not partition_name:
        partition_name = cluster_info["subsets"][0]["partition"]

    if partition_name not in partitions.keys():
        print(
            f"Unable to find partition: {partition_name}. Choose from {', '.join(partitions.keys())}"
        )
        sys.exit(1)

    # Extract the ROCm version
    rocm_version = rocm_version if rocm_version else default_rocm_version
    if rocm_version not in rocm_versions:
        print(
            f"Unsupported ROCm version: {rocm_version}. Choose from {', '.join(rocm_versions)}"
        )
        sys.exit(1)
    rocm_version = rocm_version.replace(".", "")

    max_num_nodes = partitions[partition_name]["num-nodes"]
    if num_nodes > max_num_nodes:
        print(f"Requested {num_nodes} nodes; selected subset only has {max_num_nodes}")
        sys.exit(1)

    # Default to using all GPUs in the partition.
    num_gpus_per_node = partitions[partition_name]["num-gpus"]
    num_tasks_per_node = num_gpus_per_node

    # Map the GPU type to the corresponding architecture.
    gpu_type = partitions[partition_name]["gpu"]
    if gpu_type in gpu_mapping:
        gpu_arch = gpu_mapping[gpu_type]
    else:
        print(f"Unsupported GPU type: {gpu_type}")
        sys.exit(1)

    # STEP 3. Build commands to execute and configure profilers.

    for tool in tools:
        tool_info = tool_config[tool]
        if "pre-execute" in tool_info:
            pre_execute.append("# Pre-execute commands")
            pre_execute.extend(tool_info["pre-execute"].splitlines())
        if "post-execute" in tool_info:
            post_execute.append("# Post-execute commands")
            post_execute.extend(tool_info["post-execute"].splitlines())
        if "env" in tool_info:
            environment.extend(tool_info["env"].items())
        if "args" in tool_info:
            base_args.extend(tool_info["args"])
        if "prefix" in tool_info:
            base_prefix.append(tool_info["prefix"])

    # Force num_tasks_per_node to 1 for rocprof-based tools to lower chance of
    # hangs/crashes.
    rocprof_based_tools = {"omniperf", "rocprof"}
    if len(args_tools_sources.intersection(rocprof_based_tools)) > 0:
        num_tasks_per_node = 1

    # Single node execution without a runner should be executed without distributed
    # execution as a single task.
    if not runner:
        num_tasks_per_node = 1

    if runner == "manual":
        base_args.extend(manual_args)

    if runner == "torchrun":
        num_tasks_per_node = 1
        base_prefix.append(torchrun_prefix)

    # Main set of variables as expected in the host context. We assume Apptainer
    # shares the same context as the host, so these are also used to format
    # Apptainer executions.
    variables = [
        ("omnihub_dir", f"{omnihub_dir}"),
        ("models_dir", f"{models_dir}"),
        ("shared_dir", f"{shared_dir}"),
        ("results_dir", f"{results_dir}"),
        ("app_config", f"{app_config}"),
        ("cluster", f"{cluster}"),
        ("partition", f"{partition_name}"),
        ("platform", f"{platform}"),
        ("runner", f"{runner}" if runner else ""),
        ("tools", f"{','.join(tools)}" if len(tools) else ""),
        ("num_nodes", f"{num_nodes}"),
        ("num_gpus_per_node", f"{num_gpus_per_node}"),
        ("num_tasks_per_node", f"{num_tasks_per_node}"),
        ("gpu_arch", f"{gpu_arch}"),
        ("rocm_version", f"{rocm_version}"),
    ]

    # Re-define paths that are different in Docker.
    docker_variables = [
        ("omnihub_dir", "/omnihub"),
        ("models_dir", "/models"),
        ("shared_dir", "/shared"),
        ("results_dir", "/results"),
    ]

    # For completeness, make sure all missing variables in the Docker context have
    # the same values as the equivalent variable in the host.
    docker_variables_set = {x[0] for x in docker_variables}
    for name, value in variables:
        if name not in docker_variables_set:
            docker_variables.append((name, value))

    # Build the default map of host variables to format base_prefix, base_command,
    # and and all environment variables.
    variable_map = {x[0]: x[1] for x in variables}

    # Generate host environment variables as bash exports.
    if len(environment) > 0:
        pre_execute.append("# Set environment variables")
    for variable, value in environment:
        pre_execute.append(f"export {variable}={value.format(**variable_map)}")

    # Generate environment variables for the docker container, creating a list of
    # name-value tuples that is later passed to the docker CLI.
    docker_environment = []
    if platform == "docker":
        variable_map = {x[0]: x[1] for x in docker_variables}
        for variable, value in environment:
            docker_environment.append((variable, value.format(**variable_map)))

    # Expose model and model directory to applications as environment variables.
    # The OMNIHUB prefix also ensures these are exported to Docker containers.
    omnihub_environment = [
        ("OMNIHUB_MODELS_DIR", "{models_dir}"),
    ]
    pre_execute.append("# Set OmniHub environment variables")
    for variable, value in omnihub_environment:
        pre_execute.append(f"export {variable}={value.format(**variable_map)}")

    # Additional variables that are needed to format base_command.
    variable_map["base_script"] = script
    variable_map["base_args"] = " ".join(base_args)

    if base_prefix:
        base_command = f"{' '.join(base_prefix)} {base_command}"

    base_command = base_command.format(**variable_map)

    command = base_command
    if platform == "apptainer":
        command = apptainer_wrap.format(
            base_command=base_command, gpu_arch=gpu_arch, rocm_version=rocm_version
        )
    elif platform == "docker":
        docker_args = " ".join([f"-e {x[0]}={x[1]}" for x in docker_environment])
        command = docker_wrap.format(
            base_command=base_command,
            docker_args=docker_args,
            gpu_arch=gpu_arch,
            rocm_version=rocm_version,
        )

    execute.append(command)

    # STEP 4. Generate job file.

    bash_variables = [f"{x[0]}={x[1]}" for x in variables]

    # Generate bash variables for Docker-related variables with a `docker_` prefix;
    # allows mapping volumes in the docker CLI, for example:
    #   -v $omnihub_dir:$docker_omnihub_dir`
    if platform == "docker":
        bash_variables.extend([f"docker_{x[0]}={x[1]}" for x in docker_variables])

    substitutions = {
        "job_name": "omnihub",
        "partition": partition_name,
        "num_nodes": num_nodes,
        "variables": "\n".join(bash_variables),
        "head_host_command": head_host_commands[cluster],
        "num_tasks_per_node": num_tasks_per_node,
        "num_gpus_per_node": num_gpus_per_node,
        "time_limit": time_limit_delta,
        "time_limit_seconds": time_limit_seconds,
        "pre_execute_commands": "\n".join(pre_execute),
        "execute_commands": "\n".join(execute),
        "post_execute_commands": "\n".join(post_execute),
    }

    with open(template_file, "r") as f:
        template = Template(f.read())

    print(template.safe_substitute(substitutions))


def main():
    # Paths to YAML configuration files.
    script_path = pathlib.Path(__file__).parent.parent.parent.resolve()
    config_dir = f"{script_path}/config"
    tools_dir = f"{config_dir}/tools"

    tool_config = load_tool_config(tools_dir)
    tool_names = set(tool_config.keys())

    parser = argparse.ArgumentParser(
        description="Generate a SLURM job file to run Omnihub models",
        prog="omnihub-generate-job",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30
        ),
    )
    parser._optionals.title = "Help"
    required_group = parser.add_argument_group("Required arguments")
    optional_group = parser.add_argument_group("Optional arguments")
    required_group.add_argument(
        "--omnihub-dir", help="Path to OmniHub", type=str, required=True, metavar=""
    )
    required_group.add_argument(
        "--app-config",
        help="Path to the application configuration file (relative to OmniHub directory)",
        type=str,
        required=True,
        metavar="",
    )
    optional_group.add_argument(
        "--app-args",
        help="Application-specific command line arguments",
        type=str,
        required=False,
        default="",
        metavar="",
    )
    optional_group.add_argument(
        "--num-nodes",
        help="Number of nodes (default: 1)",
        type=int,
        default=1,
        metavar="",
    )
    optional_group.add_argument(
        "--partition",
        help="Partition of the cluster",
        type=str,
        required=False,
        metavar="",
    )
    optional_group.add_argument(
        "--rocm-version",
        help="ROCm version to use (default: 6.3.1)",
        type=str,
        required=False,
        default="6.3.1",
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
        "--cluster",
        help="Name of the cluster:\n\thpcfund (default)\n\tradha",
        type=str,
        default="hpcfund",
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
        help="List of tools to choose from:\n\t{}".format(
            "\n\t".join(sorted(tool_names))
        ),
        type=str,
        nargs="+",
        default=[],
        metavar="",
    )
    optional_group.add_argument(
        "--time",
        help="Time limit for the SLURM job as an integer followed by a time unit (default: 1h). Examples: 120s, 30m, 5h.",
        type=str,
        default="1h",
        metavar="",
    )

    args = parser.parse_args()
    args_dict = vars(args)

    generate_job(config_dir, **args_dict)


if __name__ == "__main__":
    main()
