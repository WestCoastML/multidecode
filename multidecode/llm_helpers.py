import torch

def LLaMa_create_position_ids(batchsize, n_branch, context_len, tokens_to_add, llm_device, n):
    # Initialize position IDs
    position_ids = torch.full((batchsize, n_branch), fill_value=context_len + 1, dtype=torch.int).to(llm_device)

    # Create full mask
    part1 = torch.full((batchsize, 1, n_branch, context_len + n_branch * tokens_to_add), fill_value=float('-inf')).to(llm_device)
    part2 = torch.where(torch.eye(n_branch) == 1, torch.tensor(0.0), torch.tensor(float('-inf')))\
        .unsqueeze(0).unsqueeze(0).expand(batchsize, 1, -1, -1).repeat(1, 1, 1, tokens_to_add).to(llm_device)
    full_mask = torch.cat([part1, part2], dim=-1)

    return position_ids, full_mask

def LLaMa_create_masks(batchsize, n_branch, context_len, seq_len, branchs, llm_device, n):
    # Create dynamic mask for current step
    mask = torch.full((batchsize, 1, n_branch, seq_len), fill_value=float('-inf')).to(llm_device)
    mask[:, :, :, :context_len] = float(0)  # Allow attention to context
    for idx in range(batchsize):
        for branch in range(n_branch):
            for pos in range(n):
                mask[idx, :, branch, context_len + branchs[idx][branch][pos]] = float(0.0)
    
    return mask