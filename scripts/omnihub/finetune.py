import os
import sys

import torch
import bitsandbytes as bnb

from torch.nn.parallel import DistributedDataParallel

from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)

from trl import SFTTrainer

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

def setup_finetune(args, manual_runner=False):
    if not os.path.isdir(args.model_dir):
        print("Failed to find model path")
        sys.exit(1)

    if not os.path.isdir(args.output_dir):
        print("Failed to find output path")
        sys.exit(1)

    model_path = args.model_dir

    # New instruction dataset
    guanaco_dataset = "mlabonne/guanaco-llama2-1k"

    orig_model_name = os.path.split(os.path.dirname(args.model_dir))[-1]
    tuned_model_name = f"{orig_model_name}-guanaco"
    print(f"Fine-tuned model: {tuned_model_name}")

    train_dataset = load_dataset(guanaco_dataset, split="train")
    train_dataset = train_dataset.train_test_split(test_size=0.1)["test"]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "bfloat16"),
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map={"": int(os.environ["LOCAL_RANK"])} if args.ddp else "auto",
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    if args.manual_runner:
        model = DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=True)

    print_trainable_parameters(model)

    train_args = TrainingArguments(
        output_dir=f'{args.output_dir}/fine-tuned-models/{tuned_model_name}',
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
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
        report_to=None
    )

    return model, train_dataset, peft_config, tokenizer, train_args

def run_finetune(model, train_dataset, peft_config, tokenizer, train_args):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=train_args,
        packing=False,
    )
    trainer.train()

def run_inference(model, tokenizer):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    prompt = "What is a large language model?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])
