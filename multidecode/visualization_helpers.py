import torch
from transformers import AutoTokenizer

#Helpful utilities
def print_branches(branch_ids, tokenizer):
    branch_ids=branch_ids.cpu()
    for sidx,branches in enumerate(branch_ids):
        for bidx,branch_ids in enumerate(branches):
            ids=branch_ids
            print(f"{sidx}.{bidx}: {''.join(tokenizer.batch_decode(ids, skip_special_tokens=True))}")

def print_mask(mask):
    for i in range(mask.shape[2]):
        for j in range(mask.shape[3]):
            print('*' if mask[0,0,i,j]==0. else '.',end="")
        print()
    for j in range(mask.shape[3]):
        print(f'{j%10}',end="")
    print()

def print_full(output, tokenizer):
    full_ids=torch.cat([output['input_ids'],output['output_ids']],dim=-1)
    # print(f"{full_ids=}")
    # print(''.join(tokenizer.batch_decode(full_ids, skip_special_tokens=True)))
    mask=output['mask']
    for b in range(mask.shape[2]):
        branch_full_ids=[]
        for p in range(mask.shape[3]):
            if mask[0,0,b,p] == 0.0:
                branch_full_ids.append(int(full_ids[0,p]))
        print(f"***{b}:{''.join(tokenizer.batch_decode(branch_full_ids, skip_special_tokens=True))}")
        

def print_args(input_ids=None,positions=None,mask=None,branch_locations=None):
    print()
    print("Arguments:")
    print(f"{input_ids.shape=}")
    if positions is not None:
        print(f"{positions=}")
    if branch_locations is not None:
        print(f"{branch_locations=}")
    print()
    if mask is not None:
        print_mask(mask)
    print()

## moved to class to expose it to users
# def print_results(output):
#     print()
#     print("Results")
#     print("raw")
#     print(f"{''.join(tokenizer.batch_decode(output['output_ids'], skip_special_tokens=True))}")
#     print()
#     print("Reformated")
#     print_full(output)
#     print()
#     print(f"positions {output['positions']}")
#     print()
#     print_mask(output['mask'])
#     print()
    
def strmask(*args):
    n_branch=len(args)
    seq_len=len(args[0])
    ret=torch.full((n_branch,seq_len),fill_value=float('-inf'))

    for b,arg in enumerate(args):
        for i,v in enumerate(arg):
            if v=='1' or v=='*':
                ret[b,i]=0
    return ret.unsqueeze(0).unsqueeze(0)