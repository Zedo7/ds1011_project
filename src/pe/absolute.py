import math, torch
import torch.nn as nn

def sinusoidal_positions(L: int, d_model: int, device=None):
    pe = torch.zeros(L, d_model, device=device)
    position = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class AbsolutePE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16384, learned: bool = False):
        super().__init__()
        if learned:
            self.table = nn.Embedding(max_len, d_model)
        else:
            self.register_buffer("table", sinusoidal_positions(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor, start: int = 0):
        if isinstance(getattr(self, "table", None), nn.Embedding):
            idx = torch.arange(start, start + x.size(1), device=x.device)
            return x + self.table(idx)
        return x + self.table[start:start + x.size(1), :].to(x.device)
