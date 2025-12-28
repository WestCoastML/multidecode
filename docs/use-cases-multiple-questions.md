# Use Case: Answering Multiple Questions

MultiDecode excels when you need to answer multiple questions about the same context or document.

## Example

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

