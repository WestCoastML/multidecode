
import copy
import time
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
import transformers
import torch.nn.functional as F
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import multidecode.llm_helpers as llm_helpers

class MultiDecodeLLM:
    """Helper class that prepares parameters and does text generation with an arbitrary LLM."""

    def __init__(self, llm, tokenizer=None):
        """
        llm: an instance of the model or client (e.g., Hugging Face model)
        tokenizer: optional tokenizer if the backend requires one
        """
        self.llm = llm
        self.model_route = None
        self.tokenizer = tokenizer

        # TO DO: Error checking to ensure the LLM has the necessary forward() function that accepts 4D masks and position_id

    # TO DO: Helper functions to create masks and position ID lists for different use cases

    # Case 1: one prompt, n runs
    # TODO: set tokens_to_add to -1 to generate until EOS
    def setup_one_prompt_n_runs(self, prompt: str, n_branch=1, tokens_to_add=10):
        ...

    # Case 2: multi prompt, one run
    def setup_multi_prompt_one_run(self, prompts: list, tokens_to_add=10):
        ...

    def generate(self, prompt: str, n_branch=1, tokens_to_add=10, steer=None):
        """
        An optimized multi decoding algorithm for text generation.

        Args:
            steer: A function that guides the text generation by selecting tokens.
            n_branch: The number of sequences to generate.
            tokens_to_add: The number of tokens to add to each branch.

        Returns:
            A list of lists, where each sublist contains the generated token IDs for a branch.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        batchsize, context_len = input_ids.shape
        global_t0 = time.time()
        
        assert self.llm is not None, 'Model is required argument'
        if steer is None:
            # default steer is just a greedy
            def __steer(branchs, logits, output):
                next_token_probs = F.softmax(logits / 0.7, dim=-1)
                tokens = torch.argmax(next_token_probs, dim=-1)
                return tokens
            steer = __steer

        # Each branch is a list of token positions
        branchs = [[[] for _ in range(n_branch)] for _ in range(batchsize)]
        pkv = None
        position_ids = torch.full((batchsize, n_branch), fill_value=context_len + 1, dtype=torch.int).to(self.llm.device)

        # Check if it's a LLaMA-family model
        if hasattr(self.llm.config, "model_type") and "llama" in self.llm.config.model_type.lower():
            self.model_route = 'llama'
        else:
            raise ValueError(f'Model type "{self.llm.config.model_type}" not supported yet')


        if self.model_route == 'llama':
            # Call helper function to create position IDs and full mask
            position_ids, full_mask = llm_helpers.LLaMa_create_position_ids(
                batchsize, n_branch, context_len, tokens_to_add, llm_device=self.llm.device, n=0
            )

        for n in range(tokens_to_add):
            _, seq_len = input_ids.shape
            with torch.no_grad():
                if n == 0:  # For the first token, run the model on the context
                    output = self.llm.forward(input_ids=input_ids, past_key_values=pkv, use_cache=True)
                    logits = output.logits[:, -1:, :].expand(-1, n_branch, -1)
                    pkv = output.past_key_values
                else:  # For subsequent tokens, use the cache and attention mask
                    # Call helper function to create dynamic mask for current step
                    if self.model_route == 'llama':
                        mask = llm_helpers.LLaMa_create_masks(
                            batchsize, n_branch, context_len, seq_len, branchs, self.llm.device, n
                        )

                    output = self.llm.forward(input_ids=input_ids[:, -n_branch:], position_ids=position_ids, attention_mask=mask, past_key_values=pkv, use_cache=True)
                    logits = output.logits  # Get logits for the next token
                    pkv = output.past_key_values   # Update past key/values
                    position_ids += 1

                # external function to select tokens and steer the branches
                tokens = steer(branchs, logits, output)

                # add the selected tokens to the sequence
                input_ids = torch.cat([input_ids, tokens], dim=-1)

                # record location of branch
                for idx in range(batchsize):
                    for branch in range(n_branch):
                        branchs[idx][branch].append(n * n_branch + branch)

        generated_texts = []
        for idx in range(batchsize):
            generated_texts.append([])
            for branch in range(n_branch):
                ids = input_ids[idx, torch.tensor(branchs[idx][branch]) + context_len].to('cpu')
                generated_text = self.tokenizer.decode(ids, skip_special_tokens=True)
                generated_texts[idx].append(generated_text)

        return generated_texts
