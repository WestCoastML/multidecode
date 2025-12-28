# Use Case: Beam Search

Beam search is a popular decoding algorithm for text generation that explores multiple candidate sequences (branches) to find the most likely output. However, traditional beam search can be computationally expensive, especially when generating long sequences or using a large number of beams.

MultiDecode can make beam search dramatically faster by parallelizing multiple branches of the search at almost identical cost to standard decoding of only a single token.

MultiDecode is an optimized decoding algorithm that improves efficiency by processing multiple generative sequences simultaneously.
It does this by using the position_ids and attention_mask arguments to simultaneously predict the next token for all branches.
This is more efficient because the context sequence (prior to the branching) is only loaded once and is shared amongst all branches.
It is also faster because multiple tokens are generated on each forward pass of the model.

