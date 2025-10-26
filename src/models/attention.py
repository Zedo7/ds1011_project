import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- ALiBi helpers (correct length for any n_heads) ----
def _slopes_power_of_2(n: int, device=None):
    start = 2 ** (-2 ** -(math.log2(n) - 3))
    ratio = start
    vals = [start * (ratio ** i) for i in range(n)]
    return torch.tensor(vals, dtype=torch.float32, device=device)

def alibi_slopes(n_heads: int, device=None):
    if n_heads & (n_heads - 1) == 0:
        return _slopes_power_of_2(n_heads, device=device)
    closest = 1 << int(math.floor(math.log2(n_heads)))
    return torch.cat([
        _slopes_power_of_2(closest, device=device),
        alibi_slopes(2 * closest, device=device)[::2][: n_heads - closest]
    ], dim=0)

def alibi_bias(L: int, n_heads: int, device=None):
    i = torch.arange(L, device=device).view(1,1,L,1)
    j = torch.arange(L, device=device).view(1,1,1,L)
    dist = j - i
    slopes = alibi_slopes(n_heads, device=device).view(1, n_heads, 1, 1)
    return -slopes * torch.relu(dist.float())  # (1,H,L,L)

# ---- Periodic attention bias (season-aware) ----
def periodic_attention_bias(L: int, periods: Tuple[int,...], lambdas: Tuple[float,...], device=None):
    if not periods or not lambdas: return None
    i = torch.arange(L, device=device).view(1,1,L,1)
    j = torch.arange(L, device=device).view(1,1,1,L)
    d = (j - i).abs().float()
    bias = torch.zeros(1,1,L,L, device=device)
    for P, lam in zip(periods, lambdas):
        if P is None or P <= 1 or lam == 0: 
            continue
        bias = bias + float(lam) * torch.cos(2*math.pi * d / float(P))
    return bias  # (1,1,L,L)

# ---- RoPE helpers ----
def rope_angles(L: int, d_head: int, base: float = 10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(L, device=device).float().unsqueeze(1)
    return t * inv.unsqueeze(0)  # (L, d/2)

def rope_angles_multi(L: int, d_head: int, head_bases: List[float], device=None):
    H = len(head_bases)
    idx = torch.arange(0, d_head, 2, device=device).float().view(1, -1)
    bases = torch.tensor(head_bases, device=device, dtype=torch.float32).view(H,1)
    inv = 1.0 / (bases ** (idx / d_head))
    t = torch.arange(L, device=device).float().view(1,L,1)
    return t * inv.unsqueeze(1)  # (H,L,d/2)

def apply_rope_(x: torch.Tensor, theta: torch.Tensor):
    x1, x2 = x[...,0::2], x[...,1::2]
    c, s = theta.cos(), theta.sin()
    while c.dim() < x1.dim():
        c = c.unsqueeze(0); s = s.unsqueeze(0)
    xr1 = x1*c - x2*s
    xr2 = x1*s + x2*c
    return torch.stack([xr1, xr2], dim=-1).flatten(-2)

# ---- XPOS helpers ----
def xpos_scale(L: int, d_head: int, gamma: float=1e-4, device=None):
    pos = torch.arange(L, device=device).float().unsqueeze(1)
    s = torch.exp(pos * gamma).repeat(1, d_head//2)
    return s

def apply_xpos_(q: torch.Tensor, k: torch.Tensor, theta: torch.Tensor, scale: torch.Tensor):
    def rot(x, th, sc):
        x1, x2 = x[...,0::2], x[...,1::2]
        c, s = th.cos(), th.sin()
        while c.dim() < x1.dim():
            c = c.unsqueeze(0); s = s.unsqueeze(0); sc = sc.unsqueeze(0)
        xr1 = (x1*c - x2*s) * sc
        xr2 = (x1*s + x2*c) * sc
        return torch.stack([xr1, xr2], dim=-1).flatten(-2)
    return rot(q, theta, scale), rot(k, theta, 1.0/scale)

# ---- Custom MHA with optional PE and biases ----
class PEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float=0.1,
                 pe_type: str="none", rope_base: float=10000.0,
                 xpos_gamma: float=1e-4, head_bases: Optional[List[float]]=None,
                 use_alibi: bool=False, periodic_periods: Optional[Tuple[int,...]]=None,
                 periodic_lambdas: Optional[Tuple[float,...]]=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.pe_type = pe_type.lower()
        self.rope_base = rope_base
        self.xpos_gamma = xpos_gamma
        self.head_bases = head_bases
        self.use_alibi = use_alibi
        self.periodic_periods = periodic_periods or tuple()
        self.periodic_lambdas = periodic_lambdas or tuple()

    def forward(self, x: torch.Tensor, causal: bool=True):
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.Wq(x).view(B, L, H, Dh).transpose(1,2)  # (B,H,L,Dh)
        k = self.Wk(x).view(B, L, H, Dh).transpose(1,2)
        v = self.Wv(x).view(B, L, H, Dh).transpose(1,2)

        device = x.device

        # ----- positional encodings applied to q/k -----
        if self.pe_type in ("rope","xpos"):
            if self.head_bases is not None and len(self.head_bases)==H:
                theta = rope_angles_multi(L, Dh, self.head_bases, device=device)  # (H,L,D/2)
            else:
                theta = rope_angles(L, Dh, base=self.rope_base, device=device)    # (L,D/2)

        if self.pe_type == "rope":
            q = apply_rope_(q, theta)
            k = apply_rope_(k, theta)
        elif self.pe_type == "xpos":
            scale = xpos_scale(L, Dh, gamma=self.xpos_gamma, device=device)
            q, k = apply_xpos_(q, k, theta, scale)

        # ----- attention logits -----
        scale = 1.0 / math.sqrt(Dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,L,L)

        # fast causal mask (upper triangular -inf)
        if causal:
            attn = attn + torch.full((L, L), float("-inf"), device=device).triu(1)

        # ALiBi
        if self.use_alibi:
            attn = attn + alibi_bias(L, H, device=device)

        # periodic bias
        if len(self.periodic_periods) and len(self.periodic_lambdas):
            pb = periodic_attention_bias(L, self.periodic_periods, self.periodic_lambdas, device=device)
            if pb is not None:
                attn = attn + pb  # (1,1,L,L)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,L,Dh)
        out = out.transpose(1,2).contiguous().view(B, L, D)
        return self.out(out), attn
