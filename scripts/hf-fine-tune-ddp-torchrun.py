#!/usr/bin/env python3
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    Trainer,
)
from accelerate import PartialState
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training,TaskType
from trl import SFTTrainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import timedelta
import bitsandbytes as bnb
import datetime

import time
import json

def setup():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=5))
    device = torch.device(int(os.environ["LOCAL_RANK"]))    
    print('using device:', device)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

def formatting_prompts_func(example):
    print("type is: ", type(example.keys))
 
def main( args):
    
    if os.path.exists(args["path"]):
        model_name = args["path"]
    else:
        print("path does not exist")
        sys.exit(1)

    # New instruction dataset
    guanaco_dataset = "mlabonne/guanaco-llama2-1k"

    # Fine-tuned model
    new_model = os.path.split(os.path.dirname(args["path"]))[-1]+"-guanaco"
    print("new model will be named: ", new_model)

    dataset = load_dataset(guanaco_dataset, split="train")
    dataset = dataset.train_test_split(test_size=0.1)["test"]

    # compute_dtype = getattr(torch, "bfloat16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "bfloat16"),
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map= {"": int(os.environ["LOCAL_RANK"])} if args["ddp"] else "auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_params)
    print_trainable_parameters(model)

    training_params = TrainingArguments(
        output_dir=f'{args["output"]}/fine-tuned-models/{new_model}',
        num_train_epochs=15,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        # gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        optim="paged_adamw_32bit",
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
        # remove_unused_columns=False,
        report_to=None
    )

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

    trainer.train()

    # logging.set_verbosity(logging.CRITICAL)

    # Run text generation pipeline with our next model
    prompt = "What is a large language model?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune an LLM model",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", help="path to model")
    parser.add_argument("--ddp", action="store_true", help="run with DDP")
    parser.add_argument("--output", default=".", help="path to store output")
    args = vars(parser.parse_args())
    main(args)
