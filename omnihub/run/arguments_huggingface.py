from dataclasses import dataclass, field, fields, make_dataclass

from transformers import BitsAndBytesConfig


@dataclass
class ModelArguments:
    args: dict = field(
        default_factory=dict,
        metadata={
            "help": "dict of other args that can be passed to from_pretrained such as model_args"
        },
    )
    pretrained_model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class LoRAArguments:
    lora_alpha: int = field(
        default=16, metadata={"help": "The alpha parameter for LoRA."}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "The dropout rate for LoRA."}
    )
    r: int = field(default=64, metadata={"help": "The rank parameter for LoRA."})
    bias: str = field(
        default="none", metadata={"help": "The bias configuration for LoRA."}
    )
    task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "The task type for LoRA."}
    )
    target_modules: str = field(
        default="all-linear", metadata={"help": "The target modules for LoRA."}
    )


# Turn BitsandBytesConfig class variables to fields, save as dataclass, so that they can be referenced by parser
bnb_config_dict = BitsAndBytesConfig(quant_method="bitsandbytes").to_dict()
BitsAndBytesConfigDataclass = make_dataclass(
    "BitsAndBytesConfigDataclass",
    ((k, type(v), field(default=v)) for k, v in bnb_config_dict.items()),
    bases=(BitsAndBytesConfig,),
)


# Should work with python3.10
# @dataclass
# class LoRAArguments(LoraConfig):
#     init_lora_weights: bool = field(default=True)
#     layers_to_transform: int = field(default=None)
#     loftq_config: dict = field(default_factory=dict)


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_split: str = field(
        default="train", metadata={"help": "The load dataset split."}
    )
    dataset_split_test_size: str = field(
        default="train", metadata={"help": "The dataset train test split size."}
    )
    train_file: str = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: str = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_train_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    max_eval_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
        },
    )
