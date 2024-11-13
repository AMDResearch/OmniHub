# Model Organization
All ML models are stored on the following clusters and the corresponding locations.

- `radha:/shared/projs/omnihub/ml-models`
- `hpcfund:/work1/models/amd`

The models are organized as a flat structure with an approximate naming convention like `<org>-<model>-<params>-<task>-<format>`.

## Use case: Llama
For example, Llama2 models that are fine tuned for the "chat" task are organized into the following directories.
```
Meta-Llama-2-13B-Chat-safetensors
Meta-Llama-2-70B-Chat-safetensors
Meta-Llama-2-7B-Chat-safetensors
```

Similarly, Llama3 models, some of which are fine tuned for the "instruct" task and some of which are not fine tuned, are organized into the following directories.
```
Meta-Llama-3-8B-Instruct-safetensors
Meta-Llama-3-70B-Instruct-safetensors

Meta-Llama-3-8B-safetensors
Meta-Llama-3-70B-safetensors
```

## HuggingFace and Safetensors
Model directories with the `-safetensors` suffix indicate that the models have been converted to the HuggingFace-compatible "Safetensors" format.
The safetensors models can only be used by HuggingFace libraries (e.g., `transformers`)
Most of the model directories also contain the original `.pth` model as well, which may be used by native PT libraries.
