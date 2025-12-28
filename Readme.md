# Generating tokens faster using predictions from multiple token positions

This repository shares how to unlock the existing parallel decoding ability of autoregressive large language models (LLMs).
We call this algorithm "MultiDecode".
Without any modification to the architecture, training, or hardware of the LLM, use cases involving multiple content blocks (such as RAG)
or multiple completion paths (such as beam search) can achieve almost linear speedup.
MultiDecode leverages custom RoPE position values and custom attention masks 
to simultaneously and efficiently generate exact next token predictions for multiple independent token positions, using a single, shared KV cache.
Support for these custom position and mask arguments already exists in many libraries, including the Hugging Face Transformers library and vLLM.

This repo contains explanations, examples, and sample code showing how to use the MultiDecode paradigm for different use cases. 
A [YouTube video explanation of MultiDecode](https://youtu.be/9ld43ZYKzeI) is also available: \
[<img src="assets/images/video1.png?raw=true" width="400">](https://youtu.be/9ld43ZYKzeI)

## Motivating example

Consider a scenario where the manager of the technical support department wants to analyze support call transcripts.
There are 10,000 transcripts, on average several thousand tokens long.
The manager finalizes 8 yes/no questions they want an LLM to answer about each call.
Standard decoding will require 80,000 inference steps.
The wall clock time of these steps can be reduced by doing inference in batches and storing KV cache prefixes,
but it will always sum to 80,000 inference steps.
With MultiDecode, all 8 of these questions can be answered simultaneously for each document, in only 10,000 total inference steps,
each of which requires approximately the same amount of time as a standard decoding inference step 
(because decoding is I/O bound, not compute bound).

This 8x reduction in compute cost and time can be achieved with any model without any changes or fine tuning.

## Documentation

- **[MultiDecode](docs/multidecode.md)** - Background on autoregressive LLMs and self-attention, plus detailed explanation of the MultiDecode algorithm and how it uses graph structures, RoPE embeddings, and attention masks

### Use Cases

- **[Beam Search](docs/use-cases-beam-search.md)** - Accelerate beam search by parallelizing multiple candidate sequences
- **[Multiple Questions](docs/use-cases-multiple-questions.md)** - Answer multiple questions about the same context simultaneously
- **[Writing in the Margins (RAG)](docs/use-cases-writing-in-margins.md)** - Generate multiple annotations and responses for RAG applications
- **[Multiple Completions](docs/use-cases-multiple-completions.md)** - Parallel reasoning traces, sampling strategies, and multiple completion paths
