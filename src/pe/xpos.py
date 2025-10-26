import torch

def xpos_scale(L: int, d: int, gamma: float = 1e-4, device=None):
    pos = torch.arange(L, device=device).float().unsqueeze(1)
    s = torch.exp(pos * gamma)
    return s.repeat(1, d // 2)

def apply_xpos(q: torch.Tensor, k: torch.Tensor, theta: torch.Tensor, scale: torch.Tensor):
    def rot(x, th, sc):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        c, s = th.cos(), th.sin()
        while c.dim() < x1.dim():
            c = c.unsqueeze(0); s = s.unsqueeze(0); sc = sc.unsqueeze(0)
        xr1 = (x1 * c - x2 * s) * sc
        xr2 = (x1 * s + x2 * c) * sc
        return torch.stack([xr1, xr2], dim=-1).flatten(-2)
    return rot(q, theta, scale), rot(k, theta, 1.0/scale)
