import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from multidecode.mdecode import MultiDecodeLLM
import torch

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)
# if torch.backends.mps.is_available():
#     print("Using MPS backend")
# model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the MultiDecodeLLM class
mdllm = MultiDecodeLLM(model=model, tokenizer=tokenizer)

# Define a long context and a single prompt
context = "Once upon a time, in a small village, there lived a wise old man. He was known for his stories that captivated the hearts of everyone. Every evening, children would gather around him to listen to his tales."
prompt = "Who is the protagonist of the story?"

delimiter = "."

# TODO: Make example much longer
# TODO: List of subchunks instead of delimiter
# TODO: Concatenate all answer branches + original user context and prompt

# Measure setup time
start_setup_time = time.time()
mask, positions, branch_locations, input_ids = mdllm.setup_writing_in_margins(context, prompt, delimiter, verbose=True)
end_setup_time = time.time()
print(f"Setup time: {end_setup_time - start_setup_time:.2f} seconds")

# Measure generation time
start_gen_time = time.time()
output = mdllm.generate(
    model=model,
    input_ids=input_ids,
    mask=mask,
    gen_len=20,
    positions=positions,
    n_branch=len(branch_locations),
    branch_locations=branch_locations,
    greedy=True
)
end_gen_time = time.time()
print(f"Generation time: {end_gen_time - start_gen_time:.2f} seconds")

# Print the results
mdllm.print_results(output)
