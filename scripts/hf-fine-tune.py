#!/usr/bin/env python3
import contextlib
import importlib.util
import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from accelerate import PartialState

from peft import LoraConfig
from trl import SFTTrainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description="Fine-tune an LLM model",
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path", help="path to model")
parser.add_argument("--ddp", action="store_true", help="run with DDP")
parser.add_argument("--omnitrace", action="store_true", help="enable omnitrace")
args = vars(parser.parse_args())

if os.path.exists(args["path"]):
    base_model = args["path"]
else:
    print("path does not exist")
    sys.exit(1)

# The relevant section of the code to be analyzed is wrapped around a context;
# by default it won't do anything, but it can change to enable profiling.
context = contextlib.nullcontext()

if args["omnitrace"]:
    # Attempt to add the omnitrace python module, which is installed in
    # /opt/omnitrace in the Docker image. To run outside of the Docker with
    # omnitrace installed in a different path, set PYTHONPATH accordingly.
    sys.path.insert(0, "/opt/omnitrace/lib/python/site-packages/")
    omnitrace_spec = importlib.util.find_spec("omnitrace")
    if omnitrace_spec is None:
        print("unable to find omnitrace module")
        sys.exit(1)
    from omnitrace import profile
    context = profile()

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model
new_model = os.path.split(os.path.dirname(args["path"]))[-1]+"-guanaco"
print("new model will be named: ", new_model)

#%%
dataset = load_dataset(guanaco_dataset, split="train")
dataset = dataset.train_test_split(test_size=0.1)["test"]

# %%
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# %%
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map= {"": PartialState().process_index} if args["ddp"] else "auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

#%%
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


#%%
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
#%%

training_params = TrainingArguments(
    output_dir=f'./fine-tuned-models/{new_model}',
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# %%

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

with context:
    trainer.train()
