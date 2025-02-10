import json
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field, fields, make_dataclass

import bitsandbytes as bnb
import torch
import yaml
from accelerate import DistributedType, PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

import omnihub.run
import omnihub.tools


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_split: str = field(
        default="train", metadata={"help": "The load dataset split."}
    )
    dataset_split_test_size: str = field(
        default="train", metadata={"help": "The dataset train test split size."}
    )
    train_file: str = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: str = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_train_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    max_eval_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
        },
    )


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


class FineTuner:
    def __init__(self, custom_args: list, config: dict) -> None:
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

        dataset = self.get_dataset(self.DataTrainingArguments)

        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=self.SFTConfig,
        )

    def get_dataset(self, DataTrainingArgs: DataTrainingArguments):
        if DataTrainingArgs.dataset_name is not None:
            dataset = load_dataset(
                path=DataTrainingArgs.dataset_name,
                split=DataTrainingArgs.dataset_split,
            )
        elif (
            DataTrainingArgs.train_file is not None
            and DataTrainingArgs.validation_file is not None
        ):
            data_files = {
                "train": DataTrainingArgs.train_file,
                "validation": DataTrainingArgs.validation_file,
            }
            dataset = load_dataset("csv", data_files=data_files)
        elif DataTrainingArgs.train_file is not None:
            data_files = {"train": DataTrainingArgs.train_file}
            dataset = load_dataset("csv", data_files=data_files)
        else:
            raise ValueError(
                "You must specify either a dataset name or training/validation files."
            )

        # Optionally truncate the dataset
        if DataTrainingArguments.dataset_split_test_size is not None:
            dataset = dataset.train_test_split(test_size=0.1)["train"]
        if DataTrainingArguments.max_train_samples is not None:
            dataset["train"] = dataset["train"].select(
                range(DataTrainingArguments.max_train_samples)
            )
        if (
            DataTrainingArguments.max_eval_samples is not None
            and "validation" in dataset
        ):
            dataset["validation"] = dataset["validation"].select(
                range(DataTrainingArguments.max_eval_samples)
            )
        return dataset

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
        self.trainer.train()


@omnihub.run.entrypoint
def run(*args, **kwargs):
    FineTuner(*args, **kwargs).run()
