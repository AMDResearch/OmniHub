import json
from dataclasses import fields

from transformers import HfArgumentParser
from trl import SFTConfig

from omnihub.run.arguments_huggingface import (
    BitsAndBytesConfigDataclass,
    DataTrainingArguments,
    LoRAArguments,
    ModelArguments,
)
from omnihub.run.arguments_vllm import (
    PromptsArguments,
    SamplingParamsArguments,
    VllmArguments,
)


# add alias to metadata (--dataclass_name.field)
def add_aliases_to_fields(dataclass_type):
    for f in fields(dataclass_type):
        aliases = [f"--{dataclass_type.__name__}.{f.name}"]
        metadata = dict(f.metadata)
        metadata["aliases"] = aliases
        f.metadata = metadata


class CustomArgumentParser(HfArgumentParser):
    def _add_dataclass_arguments(self, dataclass_type):
        add_aliases_to_fields(dataclass_type)
        super()._add_dataclass_arguments(dataclass_type)


def parse_config(config: dict, custom_args: list):
    dataclass_list = []
    config_args = []

    # Identify dataclasses found in the configuration, and convert
    # configuration into arguments.
    for class_type, class_config in config.items():
        dataclass_list.append(globals().get(class_type))
        for name, value in class_config.items():
            config_args.append(f"--{class_type}.{name}")
            if isinstance(value, list):
                config_args.extend(map(str, value))
            if isinstance(value, dict):
                config_args.append(json.dumps(value))
            else:
                config_args.append(str(value))

    args = config_args + custom_args
    parser = CustomArgumentParser(dataclass_list)
    populated_dataclasses = parser.parse_args_into_dataclasses(
        args, return_remaining_strings=True
    )

    return populated_dataclasses
