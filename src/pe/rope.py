import torch

def rope_angles(L: int, d: int, base: float = 10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))
    t = torch.arange(L, device=device).float().unsqueeze(1)
    return t * inv_freq.unsqueeze(0)  # (L, d/2)

def apply_rope(x: torch.Tensor, theta: torch.Tensor):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = theta.cos(), theta.sin()
    while c.dim() < x1.dim():
        c = c.unsqueeze(0); s = s.unsqueeze(0)
    xr1 = x1 * c - x2 * s
    xr2 = x1 * s + x2 * c
    return torch.stack([xr1, xr2], dim=-1).flatten(-2)

def rope_angles_multi(L: int, d_head: int, head_bases, device=None):
    H = len(head_bases)
    idx = torch.arange(0, d_head, 2, device=device).float().view(1, -1)
    bases = torch.tensor(head_bases, device=device, dtype=torch.float32).view(H, 1)
    inv = 1.0 / (bases ** (idx / d_head))
    t = torch.arange(L, device=device).float().view(1, L, 1)
    return t * inv.unsqueeze(1)  # (H, L, d/2)
