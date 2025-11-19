import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from multidecode.mdecode import MultiDecodeLLM
import torch

# Load the model and tokenizer
model_name="meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)
if torch.backends.mps.is_available():
    print("Using MPS backend")
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the MultiDecodeLLM class
mdllm = MultiDecodeLLM(model=model, tokenizer=tokenizer)

# Define a prompt
prompt = "Once upon a time"

# Measure setup time
start_setup_time = time.time()
mask = mdllm.setup_one_prompt_n_runs(prompt, verbose=True)
end_setup_time = time.time()
print(f"Setup time: {end_setup_time - start_setup_time:.2f} seconds")

# Measure generation time
start_gen_time = time.time()
output = mdllm.generate(
    model=model,
    input_ids=tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device),
    mask=mask,
    gen_len=10,
    n_branch=5,
    greedy=False
)
end_gen_time = time.time()
print(f"Generation time: {end_gen_time - start_gen_time:.2f} seconds")

# Print the results
mdllm.print_results(output)