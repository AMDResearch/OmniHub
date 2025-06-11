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


# Turns nested dictionary to pairs of key-values, where keys are tuples that
# represent a path in the original dictionary.
def to_pairs(d, parent=None):
    items = []
    for k, v in d.items():
        path = parent + [k] if parent else [k]
        if isinstance(v, dict):
            items.extend(to_pairs(v, path).items())
        else:
            items.append((tuple(path), v))
    return dict(items)


# Turns dictionary of key-values pairs in which keys are tuples to a nested
# dictionary.
def to_nested(d):
    nested = {}
    for path, v in d.items():
        current_dict = nested
        *nodes, leaf = path
        for k in nodes:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[leaf] = v
    return nested


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
# of such values. It only supports two levels of nested dictionaries.  For
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

    pairs = to_pairs(template)

    # Ensure all values are lists, including keys that are not templated and only
    # have one possible value.
    for k, v in pairs.items():
        if not isinstance(v, list):
            pairs[k] = [v]

    keys = pairs.keys()
    values = pairs.values()
    combinations = [dict(zip(keys, i)) for i in itertools.product(*values)]

    num_conflicts = 0

    for i, combination in enumerate(combinations):
        conf = to_nested(combination)

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
