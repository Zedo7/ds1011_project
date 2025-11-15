import torch
import torch.nn as nn
from src.models.attention import PEAttention
from src.pe.absolute import AbsolutePE

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=4, mlp_ratio=4.0, dropout=0.1,
                 pe_type="none", rope_base=10000.0, xpos_gamma=1e-4, head_bases=None,
                 use_alibi=False, periodic_periods=(), periodic_lambdas=()):
        super().__init__()
        self.attn = PEAttention(d_model, n_heads, dropout=dropout,
                                pe_type=pe_type, rope_base=rope_base, xpos_gamma=xpos_gamma,
                                head_bases=head_bases, use_alibi=use_alibi,
                                periodic_periods=periodic_periods, periodic_lambdas=periodic_lambdas)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model*mlp_ratio), d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h,_ = self.attn(self.ln1(x), causal=True)
        x = x + self.drop(h)
        h = self.ffn(self.ln2(x))
        x = x + self.drop(h)
        return x

class TSTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=4, dropout=0.1,
                 pe="none", rope_base=10000.0, xpos_gamma=1e-4, head_bases=None,
                 use_alibi=False, periodic_periods=(), periodic_lambdas=(),
                 in_features=1, out_features=1, horizon=1):
        super().__init__()
        self.emb = nn.Linear(in_features, d_model)
        self.abs_pe = AbsolutePE(d_model) if pe == "absolute" else None
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout=dropout,
                         pe_type=pe if pe in ("rope","xpos") else "none",
                         rope_base=rope_base, xpos_gamma=xpos_gamma,
                         head_bases=head_bases, use_alibi=use_alibi,
                         periodic_periods=periodic_periods, periodic_lambdas=periodic_lambdas)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        # horizon decoder: map last token to H*out_features
        self.horizon = int(horizon)
        self.head = nn.Linear(d_model, self.horizon*out_features)
        self.pe = pe

    def forward(self, x):
        # x: (B,L,in_features)
        h = self.emb(x)
        if self.abs_pe is not None:
            h = self.abs_pe(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        last = h[:, -1, :]                  # (B,D)
        y = self.head(last)                 # (B, H*out_features)
        return y.view(x.size(0), self.horizon, -1)  # (B,H,out_features)
