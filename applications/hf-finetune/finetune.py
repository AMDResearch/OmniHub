import os
import sys
from dataclasses import asdict

import torch
from accelerate import DistributedType, PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import omnihub.run
import omnihub.tools
from omnihub.run.arguments import parse_config


class FineTuner:
    def __init__(self, custom_args: list, config: dict) -> None:
        self._parse_config(config, custom_args)

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

    def _parse_config(self, config: dict, custom_args: list):
        populated_dataclasses = parse_config(config, custom_args)

        for i in populated_dataclasses:
            setattr(self, i.__class__.__name__, i)

        # Convert BitsAndBytesConfig and add it to ModelArguments
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
