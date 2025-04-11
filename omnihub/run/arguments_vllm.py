from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class VllmArguments:
    model: str = field(metadata={"help": "Path to the model or model name"})
    tokenizer: Optional[str] = field(
        default=None, metadata={"help": "Path to the tokenizer or tokenizer name"}
    )
    tokenizer_mode: str = field(
        default="auto", metadata={"help": "Mode for the tokenizer"}
    )
    skip_tokenizer_init: bool = field(
        default=False, metadata={"help": "Whether to skip tokenizer initialization"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Whether to trust remote code"}
    )
    # allowed_local_media_path: str = field(
    #     default="",
    #     metadata={"help": "Allowed local media path"}
    # )
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of tensor parallelism"}
    )
    dtype: str = field(default="auto", metadata={"help": "Data type"})
    quantization: Optional[str] = field(
        default=None, metadata={"help": "Quantization method"}
    )
    revision: Optional[str] = field(default=None, metadata={"help": "Model revision"})
    tokenizer_revision: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer revision"}
    )
    seed: int = field(default=0, metadata={"help": "Random seed"})
    gpu_memory_utilization: float = field(
        default=0.9, metadata={"help": "GPU memory utilization"}
    )
    swap_space: float = field(default=4, metadata={"help": "Swap space"})
    cpu_offload_gb: float = field(default=0, metadata={"help": "CPU offload in GB"})
    enforce_eager: Optional[bool] = field(
        default=None, metadata={"help": "Whether to enforce eager execution"}
    )
    max_seq_len_to_capture: int = field(
        default=8192, metadata={"help": "Maximum sequence length to capture"}
    )
    disable_custom_all_reduce: bool = field(
        default=False, metadata={"help": "Whether to disable custom all-reduce"}
    )
    disable_async_output_proc: bool = field(
        default=False,
        metadata={"help": "Whether to disable asynchronous output processing"},
    )
    # hf_overrides: Optional[Dict[str, Any]] = field(
    #     default=None,
    #     metadata={"help": "Hugging Face overrides"}
    # )
    # mm_processor_kwargs: Optional[Dict[str, Any]] = field(
    #     default=None,
    #     metadata={"help": "Multi-modal processor kwargs"}
    # )
    # task: str = field(
    #     default="auto",
    #     metadata={"help": "Task option"}
    # )
    # override_pooler_config: Optional[Dict[str, Any]] = field(
    #     default=None,
    #     metadata={"help": "Override pooler config"}
    # )
    # compilation_config: Optional[Union[int, Dict[str, Any]]] = field(
    #     default=None,
    #     metadata={"help": "Compilation config"}
    # )


@dataclass
class SamplingParamsArguments:
    n: int = field(
        default=1,
        metadata={"help": "Number of output sequences to return for the given prompt."},
    )
    best_of: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens."
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens."
        },
    )
    top_k: int = field(
        default=-1,
        metadata={
            "help": "Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens."
        },
    )
    min_p: float = field(
        default=0.0,
        metadata={
            "help": "Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this."
        },
    )
    ppl_measurement: bool = field(
        default=False,
        metadata={
            "help": "Measure perplexity towards the deterministic string instead of probabilistic regressing."
        },
    )
    # seed: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "Random seed to use for the generation."}
    # )
    stop: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of strings that stop the generation when they are generated. The returned output will not contain the stop strings."
        },
    )
    stop_token_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens."
        },
    )
    include_stop_str_in_output: bool = field(
        default=False,
        metadata={
            "help": "Whether to include the stop strings in output text. Defaults to False."
        },
    )
    ignore_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the EOS token and continue generating tokens after the EOS token is generated."
        },
    )
    max_tokens: int = field(
        default=16,
        metadata={"help": "Maximum number of tokens to generate per output sequence."},
    )
    min_tokens: int = field(
        default=0,
        metadata={
            "help": "Minimum number of tokens to generate per output sequence before EOS or stop_token_ids can be generated."
        },
    )
    logprobs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of log probabilities to return per output token. When set to None, no probability is returned. If set to a non-None value, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens. Note that the implementation follows the OpenAI API: The API will always return the log probability of the sampled token, so there may be up to `logprobs+1` elements in the response."
        },
    )
    prompt_logprobs: Optional[int] = field(
        default=None,
        metadata={"help": "Number of log probabilities to return per prompt token."},
    )
    detokenize: bool = field(
        default=True,
        metadata={"help": "Whether to detokenize the output. Defaults to True."},
    )
    skip_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to skip special tokens in the output."},
    )
    spaces_between_special_tokens: bool = field(
        default=True,
        metadata={
            "help": "Whether to add spaces between special tokens in the output. Defaults to True."
        },
    )
    logits_processors: Optional[List[Callable[[Any], Any]]] = field(
        default=None,
        metadata={
            "help": "List of functions that modify logits based on previously generated tokens, and optionally prompt tokens as a first argument."
        },
    )
    truncate_prompt_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set to an integer k, will use only the last k tokens from the prompt (i.e., left truncation). Defaults to None (i.e., no truncation)."
        },
    )
    bad_words: Optional[List[str]] = field(
        default=None, metadata={"help": "Disallowed words for generation."}
    )
    logit_bias: Optional[Dict[int, float]] = field(
        default=None,
        metadata={"help": "Logit biases for specified token IDs."},
    )
    allowed_token_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Restricts generation to only these token IDs."},
    )
    # output_kind: "RequestOutputKind" = field(
    #     default_factory=lambda: RequestOutputKind.CUMULATIVE,
    #     metadata={"help": "Specifies how the output is returned."},
    # )
    # guided_decoding: Optional["GuidedDecodingParams"] = field(
    #     default=None,
    #     metadata={"help": "Parameters for guided decoding."},
    # )


@dataclass
class PromptsArguments:
    prompts: List[str] = field(
        default_factory=list,
        metadata={"help": "List of prompts to generate text from."},
    )
