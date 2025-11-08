
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
from multidecode.visualization_helpers import print_args, print_results, print_mask, print_full

class MultiDecodeLLM:
    """Helper class that prepares parameters and does text generation with an arbitrary LLM."""

    def __init__(self, model, tokenizer=None):
        """
        llm: an instance of the model or client (e.g., Hugging Face model)
        tokenizer: optional tokenizer if the backend requires one
        """
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

        self.model_route = None
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id=tokenizer.eos_token_id


        # TO DO: Error checking to ensure the LLM has the necessary forward() function that accepts 4D masks and position_id

    # TO DO: Helper functions to create masks and position ID lists for different use cases

    def lut_attn(self, n):
        ''''
        Returns a lower triangle array with dimensions and values suitable for an attention mask
        dimension: [1,1,n,n]
        values: 0 in lower triangle and diagonal
                -inf in upper triangle
        '''
        return torch.where(torch.tril(torch.ones(n,n)) == 1, torch.tensor(0.0), torch.tensor(float('-inf'))).unsqueeze(0).unsqueeze(0)

    # Case 1: one prompt, n runs
    def setup_one_prompt_n_runs(self, prompt: str, verbose=False):
        input_ids=self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.model.device)
        mask = self.lut_attn(input_ids.shape[1])
        if verbose:
            print_args(input_ids,mask=mask)
        return mask
        

    # Case 2: multi prompt, one run
    def setup_multi_prompt_one_run(self, prompts: list, context=None, verbose=False):
        context_ids = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.model.device) if context is not None else None
        input_ids = []
        question_lens = []
        for prompt in prompts:
            encoded_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.model.device)
            input_ids.append(encoded_prompt)
            question_lens.append(encoded_prompt.shape[1])
        input_ids = torch.cat(input_ids, dim=-1)
        if context_ids is not None:
            input_ids = torch.cat([context_ids, input_ids], dim=-1)
        context_len = context_ids.shape[1]
        total_question_len = sum(question_lens)

        mask = self.lut_attn(input_ids.shape[1])
        mask[:, :, context_len:total_question_len + context_len, context_len:total_question_len + context_len] = float('-inf')


        positions = torch.cat([torch.arange(context_len + q_len) for q_len in question_lens]).unsqueeze(0)
        branch_locations = [context_len + sum(question_lens[:i]) - 1 for i in range(1, len(question_lens) + 1)]
        if verbose:
            print_args(input_ids, mask=mask, positions=positions, branch_locations=branch_locations)
        return mask, positions, branch_locations

    def generate(self, model, input_ids,positions=None,mask=None,gen_len=10,n_branch=2,greedy=False,branch_locations=None,past_key_values=None):
        """
        Implements the parallel generation of tokens using the multidecode technique.

        This function generates tokens in parallel by branching at specified positions in the input sequence.
        It uses a model's forward pass to compute logits and generate tokens iteratively, either greedily or
        through sampling. The generated tokens are accumulated and returned along with other relevant data.

        Args:
            model (torch.nn.Module): The model used for token generation. Must support `forward` with caching.
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, ctx_len).
            positions (torch.Tensor, optional): Position encodings for the input tokens. If None, defaults to
                sequential positions [0, 1, ..., ctx_len-1]. Shape must match `input_ids`.
            mask (torch.Tensor): Attention mask of shape (batch_size, ctx_len, ctx_len). Controls which tokens
                the model attends to during prefill.
            gen_len (int, optional): Number of tokens to generate. Defaults to 10.
            n_branch (int, optional): Number of parallel branches for token generation. Defaults to 2.
            greedy (bool, optional): If True, selects the most probable token at each step. If False, samples
                tokens based on probabilities. Defaults to False.
            branch_locations (list, optional): List of positions where branches start. If None, defaults to
                the end of the input context.

        Returns:
            dict: A dictionary containing:
                - 'branch_ids' (torch.Tensor): Generated token IDs for each branch, reshaped to (n_branch, batch_size).
                - 'mask' (torch.Tensor): Final attention mask after generation.
                - 'output_ids' (torch.Tensor): All generated token IDs concatenated sequentially.
                - 'input_ids' (torch.Tensor): Original input token IDs.
                - 'positions' (torch.Tensor): Position encodings for all tokens, including generated ones.

        Raises:
            AssertionError: If `positions` shape does not match `input_ids` shape.

        Example:
            output = mdgen(
                model=my_model,
                input_ids=torch.tensor([[1, 2, 3]]),
                mask=torch.ones(1, 3, 3),
                gen_len=5,
                n_branch=2,
                greedy=True
            )
            print(output['branch_ids'])
        """

        past_len=0 if past_key_values is None else past_key_values.get_seq_length()

        if positions is not None:
            assert input_ids.shape == positions.shape,"positions.shape must match input_ids.shape"
        #assert mask.shape[2]==input_ids.shape[1],"length of attn mask must match input length"


        batch_size,ctx_len=input_ids.shape
        
        # every cycle we add n_branch more tokens, so the end of the 4D attention mask is a diagonal. Create here and reuse below
        gen_mask = torch.where(torch.eye(n_branch) == 1, torch.tensor(0.0), torch.tensor(float('-inf'))).unsqueeze(0).unsqueeze(0).to(model.device)

        # position information of the initial context input_ids. If None assume 0..ctx_len
        if positions is None:
            positions=torch.arange(ctx_len,dtype=torch.int).unsqueeze(0)
        positions=positions.to(model.device)
        position_history=copy.copy(positions)


        # if branch location is not specified, assume all branches start at the end of the context
        if branch_locations is None:
            branch_locations=[ctx_len-1]*n_branch
            
        assert all(bl>past_len for bl in branch_locations),"Branches must start with new input_ids, not from past_key_values."

        # the position encoding of the first generated token is just after the branch location position encoding
        tmp=[int(positions[0,x]) for x in branch_locations] 
        gen_positions=torch.tensor(tmp).unsqueeze(0).to(model.device)


        # we will accumulate the generated tokens into output_ids
        output_ids=torch.empty((batch_size,0),dtype=torch.int).to(model.device)

        # move remaining tensors to model.device
        mask=mask.to(model.device)
        input_ids=input_ids.to(model.device)
        initial_length=input_ids.shape[1]
        pkv=past_key_values

        with torch.no_grad():
            # first step is to prefill the context and generate pkv
            output=model.forward(input_ids=input_ids[:,past_len:],position_ids=positions[:,past_len:] ,attention_mask=mask, use_cache=True,past_key_values=pkv)
            pkv = output.past_key_values

            # get logits from the locations where the branches fork
            branch_locations=torch.tensor(branch_locations,dtype=torch.int)
            
            # branch locations are relative to full input sequence,
            # so we subtrack the pkv length 
            logits=output['logits'][:,branch_locations-past_len,:]
            mask = mask[:,:,branch_locations-past_len,:]

            for i in range(gen_len):
                # select tokens, greedy or not
                next_token_probs = F.softmax(logits / 0.7, dim=-1)
                if greedy:
                    tokens = torch.argmax(next_token_probs,dim=-1)
                else:
                    samples = torch.multinomial(next_token_probs.view(-1,next_token_probs.shape[-1]), num_samples=1, replacement=True).view(batch_size,n_branch)
                    tokens = samples.squeeze(-1)

                # save the generated tokens
                output_ids=torch.cat([output_ids,tokens],dim=-1)
                mask=torch.cat([mask,gen_mask],dim=-1)

                # Generate n_branch new tokens.
                output=model.forward(input_ids=tokens,position_ids=gen_positions ,attention_mask=mask, past_key_values=pkv, use_cache=True)
                logits=output['logits']
                pkv = output['past_key_values'] 

                # increment the position information for the next token
                gen_positions+=1

                position_history=torch.cat([position_history,gen_positions],dim=-1)

        # restruture the results to have n_branch sequences
        branch_ids=output_ids.view(-1,n_branch,1).permute(2,1,0).squeeze(-1)
        full_ids=torch.cat([input_ids,output_ids],dim=-1)
        return {'branch_ids':branch_ids,'mask':mask,'output_ids':output_ids,'input_ids':full_ids,
                'n_branch':n_branch,'initial_length':initial_length,'positions':position_history,'past_key_values':pkv}
    
    def print_results(self, output):
        print()
        print("Results")
        print("raw")
        print(f"{''.join(self.tokenizer.batch_decode(output['output_ids'], skip_special_tokens=True))}")
        print()
        print("Reformated")
        print_full(output)
        print()
        print(f"positions {output['positions']}")
        print()
        print_mask(output['mask'])
        print()