import torch
def blockwise_causal_mask(L: int, block: int = 64, device=None):
    idx = torch.arange(L, device=device)
    i = idx.view(L,1); j = idx.view(1,L)
    return (j <= i)
