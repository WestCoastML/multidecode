# Use Case: Writing in the Margins (RAG)

MultiDecode is particularly effective for RAG (Retrieval-Augmented Generation) use cases that involve "writing in the margins" - generating multiple annotations, summaries, or responses for different parts of a document simultaneously.

When processing retrieved documents in RAG applications, you often need to generate multiple pieces of content based on the same context. For example:
- Generating summaries for different sections
- Creating annotations for multiple passages
- Producing answers to different questions about the same retrieved content
- Generating multiple perspectives or analyses

MultiDecode allows you to generate all of these simultaneously, sharing the same KV cache for the common context, resulting in significant speedup compared to sequential generation.

The context sequence (the retrieved document) is only loaded once and is shared amongst all the generation tasks, making this approach highly efficient.

