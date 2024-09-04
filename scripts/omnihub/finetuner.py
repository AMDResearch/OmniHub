import os
import sys

import torch
import bitsandbytes as bnb

from datasets import load_dataset
from accelerate import PartialState, DistributedType

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from trl import SFTTrainer


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)


class FineTuner:
    def __init__(self, args, manual_ddp: bool = False):
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

        train_args = TrainingArguments(
            output_dir=f"{args.output_dir}/fine-tuned-models/{tuned_model_name}",
            num_train_epochs=4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            ddp_find_unused_parameters=False,  # FIXME(aaji): option conflicts with torch DDP arg for manual launcher
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
            report_to=None,  # "tensorboard"
            gradient_checkpointing_kwargs={'use_reentrant':False},
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            # attn_implementation="flash_attention_2",
            device_map=(
                "auto"
                if PartialState().distributed_type is DistributedType.NO
                else {"": PartialState().local_process_index}
            ),
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=find_all_linear_names(model),
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

        self.trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=None,
            args=train_args,
            packing=False,
        )

    def run(self):
        self.trainer.train()
