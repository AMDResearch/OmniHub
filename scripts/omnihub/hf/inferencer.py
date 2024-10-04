import torch
from accelerate import DistributedType, PartialState
from transformers import pipeline


def print_outputs(prompt, outputs):
    for output in outputs:
        generated_text = output["generated_text"]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


class Inferencer:
    def __init__(self, args):
        self.pipe = pipeline(
            task="text-generation",
            model=args.model_dir,
            max_length=200,
            model_kwargs={
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                "low_cpu_mem_usage": True,
            },
            device_map=(
                "auto"
                if PartialState().distributed_type is DistributedType.NO
                else {"": PartialState().local_process_index}
            ),
        )

    def run(self):
        pipe = self.pipe

        # summarize something
        messages = [
            {"role": "system", "content": "You are a helpful assistant!"},
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

        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = pipe(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.9,
            top_p=0.6,
        )

        print_outputs(prompt, outputs)

        # answer an open question
        prompt = "What is a large language model? Explain it to a 10 year old."
        outputs = pipe(f"<s>[INST] {prompt} [/INST]", truncation=True)
        print_outputs(prompt, outputs)
