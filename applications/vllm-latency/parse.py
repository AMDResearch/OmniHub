#!/usr/bin/env python3
#
# parse.py -- Application parser for vLLM latency benchmark

import argparse
import json
import pathlib

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("execution_dir")
args = parser.parse_args()

output_file = f"{args.execution_dir}/processed-data/app-parser.yaml"
latency_file = pathlib.Path(f"{args.execution_dir}/latency_results.json")


def parse_log(latency_file, output_file):
    if not latency_file.is_file():
        return False

    data = {}
    with open(latency_file, "r") as f:
        json_data = json.load(f)
    data["avg_latency"] = json_data.get("avg_latency")

    with open(output_file, "w") as f:
        yaml.dump(data, f)

    return True


parse_log(latency_file, output_file)
