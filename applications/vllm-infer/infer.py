import os
from dataclasses import asdict

import torch
from vllm import LLM, SamplingParams

import omnihub
from omnihub.run.arguments import parse_config


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


class Inferencer:
    def __init__(self, custom_args, config) -> None:
        self._parse_config(config, custom_args)

        default_model_path = self.VllmArguments.model
        omnihub_model_path = os.path.join(
            os.getenv("OMNIHUB_MODELS_DIR"), default_model_path
        )

        # Check if the provided argument/config is an existing directory with
        # an without the OMNIHUB_MODELS_DIR prefix. If no directory can be
        # found, assume it's a model name to be loaded from Huggingface.
        self.VllmArguments.model = default_model_path
        if not os.path.isdir(default_model_path) and os.path.isdir(omnihub_model_path):
            self.VllmArguments.model = omnihub_model_path

        self.VllmArguments.tensor_parallel_size = (
            self.VllmArguments.tensor_parallel_size
            if self.VllmArguments.tensor_parallel_size != -1
            else torch.cuda.device_count()
        )

        self.llm = LLM(
            **asdict(self.VllmArguments),
            # TODO/FIXME(aaji): TP does not work for MI250 and does not work reliably for MI300
            # pipeline_parallel_size=torch.cuda.device_count(),
            # gpu_memory_utilization=0.99,
            # max_model_len=800,
            # enforce_eager=True,
            # distributed_executor_backend="ray",
        )

    def _parse_config(self, config: dict, custom_args: list):
        populated_dataclasses = parse_config(config, custom_args)
        for i in populated_dataclasses:
            setattr(self, i.__class__.__name__, i)

    @omnihub.tools.profile()
    def run(self):
        # Create a sampling params object.
        sampling_params = SamplingParams(
            **asdict(self.SamplingParamsArguments)
            # n=1,
            # temperature=0.8,
            # top_p=0.95,
            # max_tokens=256,
            # skip_special_tokens=True,
            # ignore_eos=True,
        )

        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = self.llm.generate(self.PromptsArguments.prompts, sampling_params)
        # Print the outputs.
        print_outputs(outputs)


@omnihub.run.entrypoint
def run(*args, **kwargs):
    Inferencer(*args, **kwargs).run()
