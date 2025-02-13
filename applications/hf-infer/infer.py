import os
import sys
from dataclasses import asdict

import torch
from accelerate import DistributedType, PartialState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

import omnihub.run
import omnihub.tools
from omnihub.run.arguments import parse_config


def print_outputs(prompt, outputs):
    for output in outputs:
        generated_text = output["generated_text"]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


class Inferencer:
    def __init__(self, custom_args, config=None) -> None:
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

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.pipe = pipeline(
            task="text-generation",
            model=model,
            max_length=200,
            tokenizer=tokenizer,
        )

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
        pipe = self.pipe

        # answer an open question
        prompt = "What is a large language model? Explain it to a 10 year old."
        outputs = pipe(f"<s>[INST] {prompt} [/INST]", truncation=True)
        print_outputs(prompt, outputs)


@omnihub.run.entrypoint
def run(*args, **kwargs):
    Inferencer(*args, **kwargs).run()
