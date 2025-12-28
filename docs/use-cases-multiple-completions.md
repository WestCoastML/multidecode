# Use Case: Multiple Completions

MultiDecode enables efficient generation of multiple completion paths, parallel reasoning traces, and parallel sampling strategies.

## Parallel Reasoning Traces

When you need to explore multiple reasoning paths simultaneously, MultiDecode allows you to generate them in parallel. Each reasoning trace can branch from a common context and be processed together, sharing the KV cache for the shared prefix.

## Parallel Sampling Strategies

MultiDecode can be used with parallel sampling strategies such as Entropix, where multiple samples are needed from the same context. Rather than sampling sequentially, all samples can be generated in a single forward pass, dramatically reducing the time needed for strategies that require multiple samples.

## General Multiple Completions

For any use case where you need to generate multiple completions from the same starting point, MultiDecode provides significant speedup by:
- Sharing the KV cache for the common prefix
- Generating all completions in parallel using custom position_ids and attention_mask arguments
- Reducing the total number of forward passes needed

This makes MultiDecode ideal for applications that need to explore multiple generation paths, compare different completions, or generate diverse outputs from the same input.

