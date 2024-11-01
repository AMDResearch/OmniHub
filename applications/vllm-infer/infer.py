import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from vllm import LLM, SamplingParams

import omnihub


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


class Inferencer:
    def __init__(self, custom_args) -> None:
        parser = ArgumentParser(
            description="Inference using a vLLM model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "-m", "--model-dir", help="Path to the model", type=str, required=True
        )
        parser.add_argument(
            "--tensor-parallel-size", help="Tensor parallel size", type=int, default=-1
        )
        self.args = parser.parse_args(args=custom_args)

        if not os.path.exists(self.args.model_dir) or not os.path.isdir(
            self.args.model_dir
        ):
            print("Model path does not exist")
            parser.print_help()
            sys.exit(1)

        self.llm = LLM(
            model=self.args.model_dir,
            # TODO/FIXME(aaji): TP does not work for MI250 and does not work reliably for MI300
            tensor_parallel_size=(
                self.args.tensor_parallel_size
                if self.args.tensor_parallel_size != -1
                else torch.cuda.device_count()
            ),
            # pipeline_parallel_size=torch.cuda.device_count(),
            # gpu_memory_utilization=0.99,
            # max_model_len=800,
            # enforce_eager=True,
            # distributed_executor_backend="ray",
        )

    @omnihub.tools.profile()
    def run(self):
        # Create a sampling params object.
        sampling_params = SamplingParams(
            n=1,
            temperature=0.8,
            top_p=0.95,
            max_tokens=256,
            skip_special_tokens=True,
            ignore_eos=True,
        )

        # summarize something
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": """Generate an approximately fifteen-word sentence
                                that describes all this data:
                                Midsummer House eatType restaurant;
                                Midsummer House food Chinese;
                                Midsummer House priceRange moderate;
                                Midsummer House customer rating 3 out of 5;
                                Midsummer House near All Bar One""",
            },
        ]
        outputs = self.llm.chat(
            conversation, sampling_params=sampling_params, use_tqdm=False
        )
        # Print the outputs.
        print_outputs(outputs)

        # answer an open question
        prompts = [
            "What is a large language model? Explain it to a 10 year old.",
        ]

        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = self.llm.generate(prompts, sampling_params)
        # Print the outputs.
        print_outputs(outputs)


@omnihub.entrypoint
def run(args):
    Inferencer(args).run()
