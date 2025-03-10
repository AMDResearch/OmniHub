#!/bin/bash

results_dir=$1

mkdir -p ${results_dir}/sysinfo

amd-smi version --json > ${results_dir}/sysinfo/amd_version.json
amd-smi firmware --json > ${results_dir}/sysinfo/amd_firmware.json
amd-smi static --json > ${results_dir}/sysinfo/amd_static.json
amd-smi topology --json > ${results_dir}/sysinfo/amd_topology.json
pip list --format json  | python -m json.tool --indent=2 > ${results_dir}/sysinfo/python_packages.json
