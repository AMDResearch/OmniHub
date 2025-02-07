import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from accelerate import DistributedType, PartialState
from transformers import pipeline

import omnihub


def print_outputs(prompt, outputs):
    for output in outputs:
        generated_text = output["generated_text"]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


class Inferencer:
    def __init__(self, custom_args, config=None) -> None:
        parser = ArgumentParser(
            description="Inference using a Hugging Face model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "-m", "--model-dir", help="Path to the model", type=str, required=True
        )

        self.args = parser.parse_args(args=custom_args)

        if not os.path.exists(self.args.model_dir) or not os.path.isdir(
            self.args.model_dir
        ):
            print("Model path does not exist")
            parser.print_help()
            sys.exit(1)

        self.pipe = pipeline(
            task="text-generation",
            model=self.args.model_dir,
            max_length=200,
            model_kwargs={
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                "low_cpu_mem_usage": True,
            },
            device_map=(
                "auto"
                if PartialState().distributed_type is DistributedType.NO
                else {"": PartialState().local_process_index}
            ),
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
