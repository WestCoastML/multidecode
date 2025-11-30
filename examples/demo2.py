## Example script demonstrating the use of MultiDecodeLLM one run for multiple prompts without shared context

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from multidecode.mdecode import MultiDecodeLLM
from multidecode.auth import hf_login

# Load the model and tokenizer
hf_login()
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the MultiDecodeLLM class
mdllm = MultiDecodeLLM(model=model, tokenizer=tokenizer)

# Define prompts
prompts = ["Once upon a time", "In a galaxy far, far away"]

# Measure setup time
start_setup_time = time.time()
mask, positions, branch_locations, input_ids = mdllm.setup_multi_prompt_one_run(prompts, verbose=True)
end_setup_time = time.time()
print(f"Setup time: {end_setup_time - start_setup_time:.2f} seconds")

# Measure generation time
start_gen_time = time.time()
output = mdllm.generate(
    model=model,
    input_ids=input_ids,
    positions=positions,
    mask=mask,
    branch_locations=branch_locations,
    gen_len=10,
    greedy=False
)
end_gen_time = time.time()
print(f"Generation time: {end_gen_time - start_gen_time:.2f} seconds")

# Print the results
mdllm.print_results(output)
