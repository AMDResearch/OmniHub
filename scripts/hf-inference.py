#!/usr/bin/env python3

import transformers
import torch
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from accelerate import PartialState

parser = ArgumentParser(description="Run inference on an LLM model",
                                 formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path", help="path to model")
parser.add_argument("--ddp", action="store_true", help="run with DDP")
args = vars(parser.parse_args())

if os.path.exists(args["path"]):
    base_model = args["path"]
else:
    print("path does not exist")
    sys.exit(1)
    
pipeline = transformers.pipeline(
    "text-generation",
    model=base_model,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
    device_map= {"": PartialState().process_index} if args["ddp"] else "auto",
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
