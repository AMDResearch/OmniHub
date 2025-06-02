#!/usr/bin/env python3
#
# parse.py -- Application parser for Huggingface finetuning

import argparse
import pathlib
import re

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("execution_dir")
args = parser.parse_args()

logs_dir = f"{args.execution_dir}/logs/"
output_file = f"{args.execution_dir}/processed-data/app-parser.yaml"


def parse_log(log_file, output_file):
    if not log_file.is_file():
        return False

    with open(log_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return False

    # Manually parse key-value pairs in the log. Unfortunately the output of
    # Huggingface is not a valid JSON object nor a pickled Python dict.
    last = lines[-1].strip()
    item_pattern = r"'(.+?)':\s*(\d*\.\d+|\d+)"
    matches = re.findall(item_pattern, last)
    if not matches:
        return False

    data = {}
    for key, value in matches:
        data[key] = float(value)

    if not "train_runtime" in data.keys():
        return False

    with open(output_file, "w") as f:
        yaml.dump(data, f)

    return True


manual_logs = pathlib.Path(f"{logs_dir}/srun-0.out")
torchrun_logs = pathlib.Path(logs_dir).glob("none_*/attempt_0/0/stdout.log")
logs = [manual_logs] + list(torchrun_logs)

for log in logs:
    if parse_log(log, output_file):
        break
