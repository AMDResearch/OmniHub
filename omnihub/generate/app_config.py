import argparse
import itertools
import os
import sys

import yaml
from yaml.representer import Representer

# Definition of conflicting configurations. When specific sections (keys) are
# defined in application templates, we'll check whether the elements in the
# list (values) match the configuration. When all elements match, the
# configuration can be skipped.
match_conflicts = {
    "SFTConfig": [
        [("fp16", True), ("bf16", True)],
    ],
    "BitsAndBytesConfigDataclass": [
        [("load_in_4bit", True), ("load_in_8bit", True)],
    ],
}

# Definition of configurations where only one of the keys can be set to a non-empty value. When one of the keys is set,
# the other key must be empty. When both keys are set, the configuration can be skipped. Each section key maps to a list
# of conflict rules. Each conflict rule is a list of (key, expected_default_value) pairs.
inv_match_conflicts = {
    "SFTConfig": [
        [("fsdp", ""), ("deepspeed", "")],
    ],
}


# Recursively expand a nested dictionary with list values into a list of simple dictionaries representing each unique
# path, while minimizing extra data copies by using in-place mutation and copying only at leaf nodes.
def to_combinations(d):
    # If d is not a dictionary, return a list containing d (or each element if d is a list)
    if not isinstance(d, dict):
        return d if isinstance(d, list) else [d]

    def helper(cur, items):
        if not items:
            # Only copy at the leaf to complete one full combination.
            yield cur.copy()
            return
        key, value = items[0]
        rest = items[1:]
        if isinstance(value, list):
            for item in value:
                # Expand dict items in the list recursively.
                if isinstance(item, dict):
                    for sub in to_combinations(item):
                        cur[key] = sub
                        yield from helper(cur, rest)
                else:
                    cur[key] = item
                    yield from helper(cur, rest)
            cur.pop(key, None)
        elif isinstance(value, dict):
            for sub in to_combinations(value):
                cur[key] = sub
                yield from helper(cur, rest)
            cur.pop(key, None)
        else:
            cur[key] = value
            yield from helper(cur, rest)
            cur.pop(key, None)

    return list(helper({}, list(d.items())))


def has_conflict(conf, conflict_dict, check_fn):
    # Returns True if any section in conflict_dict violates the check_fn.
    # check_fn is a callable that compares the actual and expected value.
    for section, conflict_rules in conflict_dict.items():
        if section not in conf:
            continue
        for rule in conflict_rules:
            # If all conditions are met, we return True
            # This means that the configuration is a conflict
            if all(
                check_fn(conf[section].get(key), expected) for key, expected in rule
            ):
                return True
    return False


# Generate sets of configuration files
#
# Given a template configuration file in YAML, generate sets of experiment
# configuration files. Lists in the template are evaluated as different
# configuration values, and this function generates all possible combinations
# of such values. It supports any levels of nested dictionaries.  For
# example, the following template:
#
#   template.yaml
#     entrypoint: finetune.py
#     SFTConfig:
#       num_train_epochs:
#        - 1
#        - 2
#
# Generates these two configuration files:
#
#   config-0001.yaml
#     entrypoint: finetune.py
#     SFTConfig:
#       num_train_epochs: 1
#
#   config-0002.yaml
#     entrypoint: finetune.py
#     SFTConfig:
#       num_train_epochs: 2
#
# Assumes all lists are templated values.
def generate_app_config(output_dir, template_file):
    if not os.path.isdir(output_dir):
        print("Error: missing output directory")
        sys.exit(1)

    if not os.path.exists(template_file):
        print("Error: missing template file")
        sys.exit(1)

    with open(template_file, "r") as f:
        template = yaml.safe_load(f)

    combinations = to_combinations(template)

    num_conflicts = 0

    for i, conf in enumerate(combinations):
        # Check normal match - if a section has all values equal, that's a violation.
        if has_conflict(
            conf, match_conflicts, lambda actual, expected: actual == expected
        ):
            num_conflicts += 1
            continue

        # Check inverted match - if a section has all values not equal, that's a violation.
        # (e.g., either fsdp or deepspeed can be set, but not both)
        if has_conflict(
            conf, inv_match_conflicts, lambda actual, expected: actual != expected
        ):
            num_conflicts += 1
            continue

        conf_id = str(i).zfill(5)
        with open(f"{output_dir}/config-{conf_id}.yaml", "w") as f:
            yaml.dump(conf, f)

    num_combinations = len(combinations)
    num_generated = num_combinations - num_conflicts
    return num_combinations, num_generated
