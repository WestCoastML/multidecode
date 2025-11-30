## Example script demonstrating the use of MultiDecodeLLM for one prompt with multiple runs

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from multidecode.mdecode import MultiDecodeLLM
from multidecode.auth import hf_login

# Load the model and tokenizer
hf_login()
model_name="meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the MultiDecodeLLM class
mdllm = MultiDecodeLLM(model=model, tokenizer=tokenizer)

# Define a prompt
prompt = "Once upon a time"

# Measure setup time
start_setup_time = time.time()
mask, input_ids = mdllm.setup_one_prompt_n_runs(prompt, verbose=True)
end_setup_time = time.time()
print(f"Setup time: {end_setup_time - start_setup_time:.2f} seconds")

# Measure generation time
start_gen_time = time.time()
output = mdllm.generate(
    model=model,
    input_ids=input_ids,
    mask=mask,
    gen_len=10,
    n_branch=5,
    greedy=False
)
end_gen_time = time.time()
print(f"Generation time: {end_gen_time - start_gen_time:.2f} seconds")

# Print the results
mdllm.print_results(output)