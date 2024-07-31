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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training,TaskType
from trl import SFTTrainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import timedelta
import bitsandbytes as bnb
import datetime

# sample run commands for 2 nodes use 1,6
# torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=radha3:29400 /host-home/omnihub/scripts/hf-fine-tune-ddp.py --ddp -p /share/ml-models/Meta-Llama-2-7B-Chat-safetensors
# torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 /host-home/omnihub/scripts/hf-fine-tune-ddp.py --ddp -p /share/ml-models/Meta-Llama-2-7B-Chat-safetensors

# NCCL_DEBUG=INFO NCCL_ENABLE_DMABUF_SUPPORT=0 NCCL_IB_DISABLE=0 NCCL_P2P_DISABLE=0 torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=radha1:29400 /host-home/omnihub/scripts/hf-fine-tune-ddp.py --ddp -p /share/ml-models/Meta-Llama-2-7B-Chat-safetensors
# NCCL_DEBUG=INFO NCCL_ENABLE_DMABUF_SUPPORT=0 NCCL_IB_DISABLE=0 NCCL_P2P_DISABLE=0 torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 /host-home/omnihub/scripts/hf-fine-tune-ddp.py --ddp -p /share/ml-models/Meta-Llama-2-7B-Chat-safetensors

def setup(args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://{}:{}".format(args["master_addr"], args["master_port"]),
        # init_method='env://',
        rank=int(args["local_rank"]),
        world_size=int(args["world_size"]),
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))    
    print('using device:',torch.cuda.current_device())

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
#     return f"""### Instruction:
# Use the Input below to create an instruction, which could have been used to generate the input using an LLM.
 
# ### Input:
# {example['train']}
 
# ### Response:
# {example['response']}
# """

def main( args):
    
    if os.path.exists(args["path"]):
        model_name = args["path"]
    else:
        print("path does not exist")
        sys.exit(1)

    if args["ddp"]:
        setup(args) 
    # %%

    # New instruction dataset
    guanaco_dataset = "mlabonne/guanaco-llama2-1k"

    # Fine-tuned model
    new_model = os.path.split(os.path.dirname(args["path"]))[-1]+"-guanaco"
    print("new model will be named: ", new_model)

    #%%
    dataset = load_dataset(guanaco_dataset, split="train")
    dataset = dataset.train_test_split(test_size=0.1)["test"]

    # %%
    # compute_dtype = getattr(torch, "bfloat16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "bfloat16"),
        bnb_4bit_use_double_quant=True,
    )

    # %%
    print("making model")
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
    

    #%%
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    #%%
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
    print("ddp local rank is: ",os.environ['LOCAL_RANK'])
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=True)

    print_trainable_parameters(model)
    #%%

    training_params = TrainingArguments(
        output_dir=f'{args["output"]}/fine-tuned-models/{new_model}',
        num_train_epochs=4,
        per_device_train_batch_size=2,
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
        report_to="tensorboard"
    )

    # %%

    trainer = SFTTrainer(
        model=model.module,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        # dataset_kwargs={"skip_prepare_dataset":True},
        # formatting_func=formatting_prompts_func,
        packing=False,
    )


    trainer.train()
    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)

    # Run text generation pipeline with our next model
    prompt = "What is a large language model?"
    pipe = pipeline(task="text-generation", model=model.module, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])
    dist.destroy_process_group()

if __name__ == "__main__":
    print("herere")
    parser = ArgumentParser(description="Fine-tune an LLM model",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", help="path to model")
    parser.add_argument("--ddp", action="store_true", help="run with DDP")
    parser.add_argument("--output", default=".", help="path to store output")
    parser.add_argument("--local_rank",  help="local_rank")
    parser.add_argument("--world_size",  help="world_size")
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=str, required=True)
    args = vars(parser.parse_args())
    print(args)
    num_gpus_per_node = torch.cuda.device_count()
    print ("num_gpus_per_node = " + str(num_gpus_per_node), flush=True)

    local_rank = int(args["local_rank"]) % int(num_gpus_per_node) 
    os.environ['WORLD_SIZE'] = str(args["world_size"])
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = str(args["master_addr"])
    os.environ['MASTER_PORT'] = str(args["master_port"])
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    print(args)
    print("world:",args["world_size"])
    print("global_rank", args["local_rank"])
    main(args)
