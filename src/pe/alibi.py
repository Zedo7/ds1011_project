import math, torch

def alibi_slopes(n_heads: int):
    m = 2 ** torch.arange(0, math.ceil(math.log2(n_heads))).float()
    return (1.0 / m)[:n_heads]

def alibi_bias(L: int, n_heads: int, device=None):
    i = torch.arange(L, device=device).view(1,1,L,1)
    j = torch.arange(L, device=device).view(1,1,1,L)
    dist = j - i
    slopes = alibi_slopes(n_heads).to(device).view(1,n_heads,1,1)
    return -slopes * torch.relu(dist.float())
