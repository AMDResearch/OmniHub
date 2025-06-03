#!/usr/bin/env python3
#
# parse.py -- Application parser for vLLM throughput benchmark

import argparse
import json
import pathlib

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("execution_dir")
args = parser.parse_args()

output_file = f"{args.execution_dir}/processed-data/app-parser.yaml"
throughput_file = pathlib.Path(f"{args.execution_dir}/throughput_results.json")


def parse_log(throughput_file, output_file):
    if not throughput_file.is_file():
        return False

    data = {}
    with open(throughput_file, "r") as f:
        json_data = json.load(f)
    data["requests_per_second"] = json_data.get("requests_per_second")
    data["tokens_per_second"] = json_data.get("tokens_per_second")

    with open(output_file, "w") as f:
        yaml.dump(data, f)

    return True


parse_log(throughput_file, output_file)
