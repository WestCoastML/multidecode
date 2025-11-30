## Example script demonstrating multi-prompt generation with shared context

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
# if torch.backends.mps.is_available():
#     print("Using MPS backend")
# model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the MultiDecodeLLM class
mdllm = MultiDecodeLLM(model=model, tokenizer=tokenizer)

# Define prompts
prompts = ["What color is the bike?", "What color is the grass?"]
context = "The house is red. The grass is green. The bike is purple. "


# Measure setup time
start_setup_time = time.time()
mask, positions, branch_locations, input_ids = mdllm.setup_multi_prompt_one_run(prompts, context=context, verbose=True)
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
    greedy=True
)
end_gen_time = time.time()
print(f"Generation time: {end_gen_time - start_gen_time:.2f} seconds")

# Print the results
mdllm.print_results(output)
