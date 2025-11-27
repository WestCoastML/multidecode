
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
from multidecode.visualization_helpers import print_args, print_mask, print_full

class MultiDecodeLLM:
    """Helper class that prepares parameters and does text generation with an arbitrary LLM."""

    def __init__(self, model, tokenizer=None):
        """
        llm: an instance of the model or client (e.g., Hugging Face model)
        tokenizer: optional tokenizer if the backend requires one
        """
        if torch.cuda.is_available():
            print("Using CUDA backend")
            self.model = model.to("cuda")
        elif torch.backends.mps.is_available():
            print("Using MPS backend")
            self.model = model.to("mps")
        else:
            print("Using CPU backend")
            self.model = model.to("cpu")

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
        return mask, input_ids
    
    # TODO: Case 1: should still return input_ids for consistency
        

    # Case 2: multi prompt, one run
    def setup_multi_prompt_one_run(self, prompts: list, context=None, verbose=False):
        context_ids = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.model.device) if context is not None else None
        context_len = context_ids.shape[1] if context_ids is not None else 0
        input_ids = []
        question_lens = []
        for prompt in prompts:
            encoded_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.model.device)
            input_ids.append(encoded_prompt)
            question_lens.append(encoded_prompt.shape[1])
        input_ids = torch.cat(input_ids, dim=-1)
        if context_ids is not None:
            input_ids = torch.cat([context_ids, input_ids], dim=-1)
        total_question_len = sum(question_lens)

        mask = self.lut_attn(input_ids.shape[1])
        for prompt_idx in range(len(prompts)):
            start_q = context_len + sum(question_lens[:prompt_idx])
            end_q = start_q + question_lens[prompt_idx]
            # Block attention from this question to other questions
            mask[:, :, start_q:end_q, context_len:start_q] = float('-inf')

        ctx_positions = torch.arange(context_len)
        prpt_positions = torch.cat([torch.arange(context_len, context_len + q_len) for q_len in question_lens])
        positions = torch.cat([ctx_positions, prpt_positions]).unsqueeze(0).to(self.model.device)
        branch_locations = [context_len + sum(question_lens[:i]) - 1 for i in range(1, len(question_lens)+1)]
        if verbose:
            print_args(input_ids, mask=mask, positions=positions, branch_locations=branch_locations)
        return mask, positions, branch_locations, input_ids
    
    # Case 3: Writing in the margins
    def setup_writing_in_margins(self, context: str, prompt: str, delimiter: str, verbose=False):
        subcontexts = context.split(delimiter)
        if subcontexts[-1].strip() == "":
            subcontexts = subcontexts[:-1]
        input_strings = [subcontext + delimiter + " " + prompt for subcontext in subcontexts]
        mask, positions, branch_locations, input_ids = self.setup_multi_prompt_one_run(input_strings, verbose=verbose)
        # for subcontext in subcontexts:
        #     subcontext_ids = self.tokenizer(subcontext, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.model.device)
        #     prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.model.device)
        #     combined_ids = torch.cat([subcontext_ids, prompt_ids], dim=-1)
        #     batch_size, ctx_len = combined_ids.shape
        #     total_len = ctx_len

        #     mask = self.lut_attn(total_len)
        #     mask[:, :, :ctx_len, :ctx_len] = float(0.0)

        #     position = torch.arange(total_len).unsqueeze(0).to(self.model.device)

        #     input_ids.append(combined_ids)
        #     masks.append(mask)
        #     positions.append(position)

        # # Combine input_ids, masks, and positions
        # input_ids = torch.cat(input_ids, dim=-1)
        # masks = torch.cat(masks, dim=-1)
        # positions = torch.cat(positions, dim=-1)

        # # Create combined mask to hide questions from each other's context
        # context_lens = [ids.shape[1] for ids in input_ids.split(prompt_ids.shape[1], dim=-1)]
        # question_lens = [prompt_ids.shape[1]] * len(subcontexts)

        # total_context_len = sum(context_lens)
        # for i in range(len(subcontexts)):
        #     start_i = sum(context_lens[:i]) + sum(question_lens[:i])
        #     end_i = start_i + question_lens[i]
        #     masks[:, :, start_i:end_i, :total_context_len] = float('-inf')
        #     masks[:, :, :total_context_len, start_i:end_i] = float('-inf')

        # branch_locations = [sum(context_lens) + sum(question_lens[:i]) - 1 for i in range(1, len(question_lens) + 1)]

        # if verbose:
        #     print_args(input_ids, masks, positions)

        return mask, positions, branch_locations, input_ids

    # Case 4: Beam search

    def generate(self, model, input_ids,positions=None,mask=None,gen_len=-1,n_branch=2,greedy=False,branch_locations=None,past_key_values=None):
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

            # Determine if we need to generate until EOS
            ## Functionality to generate until EOS token, however current model weights does not trigger an EOS
            ## Possibly due to incorrect fine-tuning or not being chat format
            generate_until_eos = (gen_len == -1)
            if generate_until_eos:
                max_iterations = 30  # Safety limit to prevent infinite loops
                branches_complete = torch.zeros(n_branch, dtype=torch.bool).to(model.device)
            else:
                max_iterations = gen_len
            
            iteration = 0
            while iteration < max_iterations:
                # Check if all branches are complete (only if generating until EOS)
                if generate_until_eos and branches_complete.all():
                    break
                
                # select tokens, greedy or not
                next_token_probs = F.softmax(logits / 0.7, dim=-1)
                if greedy:
                    tokens = torch.argmax(next_token_probs,dim=-1)
                else:
                    samples = torch.multinomial(next_token_probs.view(-1,next_token_probs.shape[-1]), num_samples=1, replacement=True).view(batch_size,n_branch)
                    tokens = samples.squeeze(-1)

                # Mark branches as complete if they generated EOS token
                if generate_until_eos:
                    branches_complete |= (tokens[0] == self.tokenizer.eos_token_id)

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
                
                iteration += 1


        # restruture the results to have n_branch sequences
        branch_ids=output_ids.view(-1,n_branch,1).permute(2,1,0).squeeze(-1)
        full_ids=torch.cat([input_ids,output_ids],dim=-1)
        return {'branch_ids':branch_ids,'mask':mask,'output_ids':output_ids,'input_ids':full_ids,
                'n_branch':n_branch,'initial_length':initial_length,'positions':position_history,'past_key_values':pkv}
    
    # def select_branch(self, output, selected_branch):
    #     """
    #     Selects a specific branch from the output of the `mdgen` function.

    #     This function extracts the input IDs, position IDs, attention mask, and past key values
    #     corresponding to the specified branch index from the output of the `mdgen` function.

    #     Args:
    #         output (dict): The output dictionary from the `mdgen` function. It should contain:
    #             - 'branch_ids' (torch.Tensor): Generated token IDs for each branch.
    #             - 'positions' (torch.Tensor): Position encodings for each branch.
    #             - 'mask' (torch.Tensor): Attention mask for each branch.
    #             - 'past_key_values' (optional): Cached key-value pairs for efficient decoding.
    #         branch_index (int): The index of the branch to select.

    #     Returns:
    #         tuple: A tuple containing:
    #             - input2_ids (torch.Tensor): The token IDs for the selected branch.
    #             - position2_ids (torch.Tensor): The position encodings for the selected branch.
    #             - mask2 (torch.Tensor): The attention mask for the selected branch.
    #             - pkv (optional): The past key-value pairs for the selected branch, if available.

    #     Raises:
    #         IndexError: If the specified branch index is out of range.

    #     Example:
    #         input2_ids, position2_ids, mask2, pkv = select_branch(output, branch_index=1)
    #     """
    #     pkv=output['past_key_values']
    #     o_positions=output['positions']
    #     o_mask=output['mask']
    #     o_input_ids=output['input_ids']
    #     o_initial_len=output['initial_length']
    #     o_input_ids_len=o_input_ids.shape[1]

    #     input_indexes=torch.cat([torch.arange(o_initial_len),torch.arange(o_initial_len+selected_branch,o_input_ids_len+1,n_branch,dtype=torch.int)],dim=-1)

    #     input2_ids=o_input_ids[:,input_indexes]
    #     position2_ids=o_positions[:,input_indexes]
    #     selected_len=(o_input_ids.shape[1]- o_initial_len)//n_branch
    #     mask2=o_mask[:,:,selected_branch,input_indexes[:o_initial_len]].repeat([1,1,selected_len,1])
    #     mask2=torch.cat([mask2,self.lut_attn(selected_len).to(self.model.device)],dim=-1)

    #     pkv.crop(o_initial_len)

    #     print(f"{input2_ids.shape=} {position2_ids.shape=} {mask2.shape=} {pkv.get_seq_length()=}")
    #     return input2_ids,position2_ids, mask2, pkv
    
    def print_results(self, output):
        print()
        print("Results")
        print("raw")
        print(f"{''.join(self.tokenizer.batch_decode(output['output_ids'], skip_special_tokens=True))}")
        print()
        print("Reformated")
        print_full(output, self.tokenizer)
        print()
        print(f"positions {output['positions']}")
        print()
        print_mask(output['mask'])
        print()