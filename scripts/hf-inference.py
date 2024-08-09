#!/usr/bin/env python3

import os
import sys

import torch
import transformers

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from accelerate import PartialState

from omnihub import tracer

parser = ArgumentParser(description="Run inference on an LLM model",
                    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path", help="Path to the model", required=True)
parser.add_argument("--ddp", action="store_true", help="Run with DDP")
parser.add_argument("--omnitrace", action="store_true", help="Enable omnitrace")
args = vars(parser.parse_args())

@tracer.profile(use_omnitrace=args["omnitrace"])
def run_inference(model_path: str=None, use_ddp: bool=False):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
        },
        device_map= {"": PartialState().process_index} if use_ddp else "auto",
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {"role": "user", "content": """Generate an approximately fifteen-word sentence 
                                    that describes all this data:
                                    Midsummer House eatType restaurant; 
                                    Midsummer House food Chinese; 
                                    Midsummer House priceRange moderate; 
                                    Midsummer House customer rating 3 out of 5; 
                                    Midsummer House near All Bar One"""},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.9,
        top_p=0.6,
    )

    print(outputs[0]["generated_text"][len(prompt):])

def main():
    if os.path.exists(args["path"]):
        base_model = args["path"]
    else:
        print("Path does not exist")
        parser.print_help()
        sys.exit(1)

    run_inference(base_model, args["ddp"])

    # (alternate) context manager approach
    #with tracer.profile(use_omnitrace=args["omnitrace"]):
    #    run_inference(base_model, args["ddp"])

if __name__ == "__main__":
    main()

