Write-Host "== Converting module notebooks to .py and cleaning up ==" -ForegroundColor Cyan

# Ensure package inits
$pkgs = @("src","src\data","src\models","src\pe","src\utils","src\tests")
foreach ($p in $pkgs) {
  New-Item -ItemType Directory -Force -Path $p | Out-Null
  New-Item -ItemType File -Force -Path (Join-Path $p "__init__.py") | Out-Null
}
Write-Host "  • Package __init__.py files created."

# Helper to write files
function Write-File($path, $content) {
  $dir = Split-Path $path
  if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  $content | Set-Content -Path $path -Encoding UTF8
  Write-Host "  • Wrote $path"
}

# --- PE modules ---
Write-File "src\pe\absolute.py" @"
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
"@

Write-File "src\pe\rope.py" @"
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
"@

Write-File "src\pe\alibi.py" @"
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
"@

Write-File "src\pe\xpos.py" @"
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
"@

Write-File "src\pe\bcm.py" @"
import torch
def blockwise_causal_mask(L: int, block: int = 64, device=None):
    idx = torch.arange(L, device=device)
    i = idx.view(L,1); j = idx.view(1,L)
    return (j <= i)
"@

# --- Utils ---
Write-File "src\utils\metrics.py" @"
import numpy as np
def mae(y,yhat):  return float(np.mean(np.abs(yhat-y)))
def mse(y,yhat):  return float(np.mean((yhat-y)**2))
def rmse(y,yhat): return float(np.sqrt(mse(y,yhat)))
def smape(y,yhat,eps=1e-6):
    denom=(np.abs(y)+np.abs(yhat)+eps)/2.0
    return float(np.mean(np.abs(yhat-y)/denom)*100.0)
def mase(y,yhat):
    q=float(np.mean(np.abs(y[1:]-y[:-1]))+1e-6)
    return float(np.mean(np.abs(yhat-y))/q)
def les(err_eval, err_train): return float(err_eval)/(float(err_train)+1e-12)
"@

Write-File "src\utils\attention_viz.py" "# Plot helpers placeholder"

# --- Data ---
Write-File "src\data\timeseries_pile.py" @"
from pathlib import Path
import numpy as np, pandas as pd
class TimeSeriesPile:
    def __init__(self, root='./data', dataset='electricity', normalize='zscore_per_series'):
        self.root=Path(root); self.dataset=dataset; self.normalize=normalize
    def load(self):
        return pd.DataFrame({'y': np.random.randn(5000)}, index=pd.RangeIndex(5000))
    def make_windows(self, arr, L=256, H=48, stride=1):
        X,Y=[],[]
        T=arr.shape[0]
        for s in range(0, T-L-H+1, stride):
            X.append(arr[s:s+L]); Y.append(arr[s+L:s+L+H])
        import numpy as np
        return np.stack(X), np.stack(Y)
"@

# --- Model ---
Write-File "src\models\ts_transformer.py" @"
import torch, torch.nn as nn
from src.pe.absolute import AbsolutePE
class TSTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=6, dropout=0.1, pe='absolute', in_features=1, out_features=1):
        super().__init__()
        self.emb = nn.Linear(in_features, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, out_features)
        self.pe_type = pe
        self.pe = AbsolutePE(d_model) if pe=='absolute' else None
    def forward(self, x):
        h = self.emb(x)
        if self.pe is not None: h = self.pe(h)
        h = self.encoder(h)
        return self.head(h[:, -1:, :])
"@

# --- Train/Eval drivers ---
Write-File "train.py" @"
import argparse, time, torch, torch.nn as nn, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.data.timeseries_pile import TimeSeriesPile
from src.models.ts_transformer import TSTransformer
from src.utils.metrics import mae

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=256); p.add_argument('--H', type=int, default=48)
    p.add_argument('--epochs', type=int, default=2); p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4); args=p.parse_args()

    ds=TimeSeriesPile(); df=ds.load()
    arr=df['y'].values.astype('float32').reshape(-1,1)
    X,Y=ds.make_windows(arr, L=args.L, H=args.H)
    Xtr,Ytr=X[:-100],Y[:-100]; Xva,Yva=X[-100:],Y[-100:]
    tr=DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), batch_size=args.batch_size, shuffle=True)
    va=DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)), batch_size=args.batch_size)

    model=TSTransformer(); opt=torch.optim.AdamW(model.parameters(), lr=args.lr); loss_fn=nn.MSELoss()
    for ep in range(1, args.epochs+1):
        t0=time.time()
        for xb,yb in tr:
            pred=model(xb); loss=loss_fn(pred.squeeze(-1), yb[:, :1, 0]); opt.zero_grad(); loss.backward(); opt.step()
        dt=time.time()-t0
        xb,yb=next(iter(va)); with torch.no_grad(): 
            pv=model(xb).squeeze(-1).numpy(); yv=yb[:, :1, 0].numpy()
        print(f"epoch {ep:02d} | loss {loss.item():.4f} | val MAE {mae(yv,pv):.4f} | {dt:.1f}s")
    torch.save(model.state_dict(), 'checkpoints/ts_transformer.pt'); print('Saved -> checkpoints/ts_transformer.pt')

if __name__=='__main__': main()
"@

Write-File "eval.py" @"
import argparse, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from src.data.timeseries_pile import TimeSeriesPile
from src.models.ts_transformer import TSTransformer
from src.utils.metrics import mae, rmse, smape, mase
def main():
    p=argparse.ArgumentParser(); p.add_argument('--L_eval', type=int, default=512); p.add_argument('--H', type=int, default=48)
    p.add_argument('--ckpt', default='checkpoints/ts_transformer.pt'); args=p.parse_args()
    ds=TimeSeriesPile(); df=ds.load(); arr=df['y'].values.astype('float32').reshape(-1,1)
    X,Y=ds.make_windows(arr, L=args.L_eval, H=args.H)
    dl=DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=128)
    m=TSTransformer(); m.load_state_dict(torch.load(args.ckpt, map_location='cpu')); m.eval()
    P,T=[],[]
    with torch.no_grad():
        for xb,yb in dl:
            P.append(m(xb).squeeze(-1).numpy()); T.append(yb[:, :1, 0].numpy())
    yhat=np.concatenate(P,0); y=np.concatenate(T,0)
    print("MAE", mae(y,yhat), "RMSE", rmse(y,yhat), "sMAPE", smape(y,yhat), "MASE", mase(y,yhat))
if __name__=='__main__': main()
"@

# Remove notebook versions of modules (keep EDA notebooks)
$nb = @(
  "src\data\timeseries_pile.ipynb","src\models\ts_transformer.ipynb",
  "src\pe\absolute.ipynb","src\pe\alibi.ipynb","src\pe\bcm.ipynb","src\pe\rope.ipynb","src\pe\xpos.ipynb",
  "src\utils\metrics.ipynb","src\utils\attention_viz.ipynb"
)
foreach ($f in $nb) { if (Test-Path $f) { Remove-Item -Force $f; Write-Host "  • Removed $f" } }

Write-Host "== Done. Try: python train.py --epochs 1 ==" -ForegroundColor Green
