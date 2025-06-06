# Generating tokens faster using predictions from multiple token positions

This repository shares how to unlock the existing parallel decoding ability of autoregressive large language models (LLMs).
We call this algorithm "MultiDecode".
Without any modification to the architecture, training, or hardware of the LLM, use cases involving multiple content blocks (such as RAG)
or multiple completion paths (such as beam search) can achieve almost linear speedup.
MultiDecode leverages custom RoPE position values and custom attention masks 
to simultaneously and efficiently generate exact next token predictions for multiple independent token positions, using a single, shared KV cache.
Support for these custom position and mask arguments already exists in many libraries, including the Hugging Face Transformers library.

This repo contains explanations, examples, and sample code showing how to use the MultiDecode paradigm for different use cases. 
A [YouTube video explanation of MultiDecode](https://youtu.be/9ld43ZYKzeI) is also available.

## Introduction

Beam search is a popular decoding algorithm for text generation that explores multiple candidate sequences (branchs) to find the most likely output. However, traditional beam search can be computationally expensive, especially when generating long sequences or using a large number of beams.

MultiDecode is a generalization of beam search that can be applied to many use case, such as 
- beamsearch 
- multiple questions 
- writing in the margins
- parallel reasoning
- parallel sampling (i.e. entropix)
- predicting users 

MultiDecode is an optimized decoding algorithm that improves efficiency by processing multiple generative sequences simultaneously.  It does this by using the position_ids and attention_mask arguments to simultaneously predict the next token for all branches.  This is more efficient because the context sequence (prior to the branching) is only loaded once and is share amoungst all branchs. It is also faster because multiple tokens are generated on each forward pass of the model.
