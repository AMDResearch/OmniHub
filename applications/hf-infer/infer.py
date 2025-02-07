import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field, fields, make_dataclass

import torch
from accelerate import DistributedType, PartialState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
)

import omnihub


def print_outputs(prompt, outputs):
    for output in outputs:
        generated_text = output["generated_text"]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


@dataclass
class LoRAArguments:
    lora_alpha: int = field(
        default=16, metadata={"help": "The alpha parameter for LoRA."}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "The dropout rate for LoRA."}
    )
    r: int = field(default=64, metadata={"help": "The rank parameter for LoRA."})
    bias: str = field(
        default="none", metadata={"help": "The bias configuration for LoRA."}
    )
    task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "The task type for LoRA."}
    )
    target_modules: str = field(
        default="all-linear", metadata={"help": "The target modules for LoRA."}
    )


# turn BitsandBytesConfig class variables to fields, save as dataclass, so that they can be referenced by parser
my_dict = BitsAndBytesConfig(quant_method="bitsandbytes").to_dict()
BitsAndBytesConfigDataclass = make_dataclass(
    "BitsAndBytesConfigDataclass",
    ((k, type(v), field(default=v)) for k, v in my_dict.items()),
    bases=(BitsAndBytesConfig,),
)


# Should work with python3.10
# @dataclass
# class LoRAArguments(LoraConfig):
#     init_lora_weights: bool = field(default=True)
#     layers_to_transform: int = field(default=None)
#     loftq_config: dict = field(default_factory=dict)


@dataclass
class ModelArguments:
    args: dict = field(
        default_factory=dict,
        metadata={
            "help": "dict of other args that can be passed to from_pretrained such as model_args"
        },
    )
    pretrained_model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


# add alias to metadata (--dataclass_name.field)
def add_aliases_to_fields(dataclass_type):
    for f in fields(dataclass_type):
        aliases = [f"--{dataclass_type.__name__}.{f.name}"]
        metadata = dict(f.metadata)
        metadata["aliases"] = aliases
        f.metadata = metadata


class CustomHfArgumentParser(HfArgumentParser):
    def _add_dataclass_arguments(self, dataclass_type):
        add_aliases_to_fields(dataclass_type)
        super()._add_dataclass_arguments(dataclass_type)


class Inferencer:
    def __init__(self, custom_args, config=None) -> None:
        self.parse_args(custom_args, config)

        default_model_path = self.ModelArguments.pretrained_model_name_or_path
        omnihub_model_path = os.path.join(
            os.getenv("OMNIHUB_MODELS_DIR"), default_model_path
        )

        # Check if the provided argument/config is an existing directory with
        # an without the OMNIHUB_MODELS_DIR prefix. If no directory can be
        # found, assume it's a model name to be loaded from Huggingface.
        model_path = default_model_path
        if not os.path.isdir(default_model_path) and os.path.isdir(omnihub_model_path):
            model_path = omnihub_model_path

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=(
                "auto"
                if PartialState().distributed_type is DistributedType.NO
                else {"": PartialState().local_process_index}
            ),
            **self.ModelArguments.args,
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        if hasattr(self, "LoRAArguments"):
            model = get_peft_model(model, LoraConfig(**asdict(self.LoRAArguments)))

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.pipe = pipeline(
            task="text-generation",
            model=model,
            max_length=200,
            tokenizer=tokenizer,
        )

    def parse_args(self, custom_args: list, config: dict):

        DataclassesToParse = [ModelArguments]
        config_as_args = []

        for class_type, class_config in config.items():

            # Add classes to list, should only intantiate classes if present in args
            DataclassesToParse.append(globals().get(class_type))

            for arg, value in class_config.items():

                # turn config into same format as custom _args
                config_as_args.append(f"--{class_type}.{arg}={value}")

        DataclassesToParse = list(set(DataclassesToParse))
        merged_args = config_as_args + custom_args

        parser = CustomHfArgumentParser(DataclassesToParse)

        parsed_args_as_dataclasses = parser.parse_args_into_dataclasses(
            merged_args, return_remaining_strings=True
        )
        for attribute, dataclass in zip(DataclassesToParse, parsed_args_as_dataclasses):
            setattr(self, attribute.__name__, dataclass)

        # convert BitsAndBytesConfigDataclass to hf BitsAndBytesConfig and add to ModelArguments
        if hasattr(self, "BitsAndBytesConfigDataclass"):
            self.ModelArguments.args["quantization_config"] = BitsAndBytesConfig(
                **asdict(self.BitsAndBytesConfigDataclass)
            )

    @omnihub.tools.profile()
    def run(self):
        pipe = self.pipe

        # answer an open question
        prompt = "What is a large language model? Explain it to a 10 year old."
        outputs = pipe(f"<s>[INST] {prompt} [/INST]", truncation=True)
        print_outputs(prompt, outputs)


@omnihub.entrypoint
def run(*args, **kwargs):
    Inferencer(*args, **kwargs).run()
