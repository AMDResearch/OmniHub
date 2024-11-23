import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field

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

import omnihub


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


@dataclass
class ModelArguments:
    args: dict = field(
        default_factory=dict,
        metadata={
            "help": "dict of other args that can be passed to from_pretrained such as model_args"
        },
    )
    pretrained_model_name_or_path: str = field(
        default=(
            os.getenv("OMNIHUB_MODEL_DIR")
            or os.getenv("OMNIHUB_MODEL")
            or "bert-base-uncased"
        ),
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


class FineTuner:
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

    def parse_args(self, supported_dataclass: list, custom_args: list, config: dict):
        # Default: read from args from input config
        for dataclass in supported_dataclass:
            dataclass_name = dataclass.__name__
            parser = HfArgumentParser([dataclass])

            # Flatten config; if dataclass is not in config, default values are supplied
            flat_config = {}
            if dataclass_name in config:
                flat_config = config[dataclass_name]

            (parsed_dataclass,) = parser.parse_dict(flat_config, allow_extra_keys=True)
            setattr(self, dataclass_name, parsed_dataclass)

        # turn custom_args into list of key value pairs
        args_override = [(arg.lstrip("--")).split("=") for arg in custom_args]

        for key, val in args_override:
            for dataclass in supported_dataclass:
                dataclass_name = dataclass.__name__

                # replace attribute in self.<dataclass_name>
                if hasattr(dataclass, key):
                    attr = getattr(self, dataclass_name)
                    setattr(attr, key, val)
                    setattr(self, dataclass_name, attr)
                    continue

    def __init__(self, custom_args: list, config: dict) -> None:
        supported_dataclasses = [
            ModelArguments,
            SFTConfig,
            LoRAArguments,
            DataTrainingArguments,
            BitsAndBytesConfig,
        ]
        self.parse_args(supported_dataclasses, custom_args, config)

        self.ModelArguments.args["quantization_config"] = self.BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            self.ModelArguments.pretrained_model_name_or_path,
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

        if "LoRAArguments" in config:
            peft_config = LoraConfig(**asdict(self.LoRAArguments))
            model = get_peft_model(model, peft_config)

        dataset = self.get_dataset(self.DataTrainingArguments)

        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=self.SFTConfig,
        )

    @omnihub.tools.profile()
    def run(self):
        self.trainer.train()


@omnihub.entrypoint
def run(*args, **kwargs):
    FineTuner(*args, **kwargs).run()
