import argparse, time, csv, os
from datetime import datetime
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data.timeseries_pile import TimeSeriesPile
from src.models.ts_transformer import TSTransformer
from src.utils.metrics import mae

def count_params_m(model): 
    return sum(p.numel() for p in model.parameters()) / 1e6

def denorm_horizon(pred, targ, sids, means, stds, H):
    # pred,targ: (B,H) or (N*H,); sids: (B,)
    pred = np.asarray(pred).reshape(-1)
    targ = np.asarray(targ).reshape(-1)
    sids = np.asarray(sids).astype("int64").reshape(-1)
    srep = np.repeat(sids, H)
    m = means[srep]; s = stds[srep]
    return pred * s + m, targ * s + m

def main():
    p=argparse.ArgumentParser()
    # data
    p.add_argument('--dataset', default='electricity', choices=TimeSeriesPile.available_datasets())
    p.add_argument('--L', type=int, default=256); p.add_argument('--H', type=int, default=24)
    p.add_argument('--stride', type=int, default=30); p.add_argument('--max_series', type=int, default=8)
    p.add_argument('--normalize', default='zscore', choices=['none','zscore'])
    # train
    p.add_argument('--epochs', type=int, default=5); p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    # PE
    p.add_argument('--pe', default='none', choices=['none','absolute','rope','xpos','alibi'])
    p.add_argument('--rope_base', type=float, default=10000.0); p.add_argument('--xpos_gamma', type=float, default=1e-4)
    p.add_argument('--use_periodic_bias', action='store_true')
    p.add_argument('--periodic_lambdas', type=str, default='0.3,0.2')
    p.add_argument('--multi_base', type=str, default='')
    # logging
    p.add_argument('--log_csv', default='reports/tables/results.csv')
    args=p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if (args.device=='auto' and torch.cuda.is_available()) else (args.device if args.device!='auto' else 'cpu')
    print(f"[device] {device}")

    ds=TimeSeriesPile()
    df=ds.load(args.dataset)
    periods,_ = ds.infer_periods(df, topk_fft=1)
    if args.use_periodic_bias:
        pb_periods = tuple(periods[:2])
        pb_lambdas = tuple(float(x) for x in args.periodic_lambdas.split(',')[:len(pb_periods)])
    else:
        pb_periods, pb_lambdas = (), ()

    print(f"[data] {args.dataset} df shape = {df.shape}")
    t0=time.time()
    X, Y, meta = ds.make_windows_from_df(
        df, L=args.L, H=args.H, stride=args.stride, max_series=args.max_series,
        normalize=args.normalize, return_meta=True
    )
    sids = meta["series_ids"]; means = meta["means"]; stds = meta["stds"]
    print(f"[windows] L={args.L} H={args.H} stride={args.stride} max_series={args.max_series}")
    print(f"[windows] X={X.shape} Y={Y.shape} ({time.time()-t0:.1f}s) | normalize={args.normalize}")

    # split
    n_hold = min(2000, max(200, len(X)//10))
    Xtr, Ytr, S_tr = X[:-n_hold], Y[:-n_hold], sids[:-n_hold]
    Xva, Yva, S_va = X[-n_hold:], Y[-n_hold:], sids[-n_hold:]

    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr), torch.from_numpy(S_tr)),
                    batch_size=args.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva), torch.from_numpy(S_va)),
                    batch_size=args.batch_size)

    head_bases = [float(x) for x in args.multi_base.split(',')] if args.multi_base else None
    model = TSTransformer(
        d_model=256, n_heads=8, n_layers=4, dropout=0.1,
        pe=args.pe, rope_base=args.rope_base, xpos_gamma=args.xpos_gamma, head_bases=head_bases,
        use_alibi=(args.pe=='alibi'), periodic_periods=pb_periods, periodic_lambdas=pb_lambdas,
        in_features=1, out_features=1, horizon=args.H
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epoch_times=[]; last_val_mae_norm=None; last_val_mae_raw=None

    for ep in range(1, args.epochs+1):
        et0=time.time()
        model.train()
        for xb, yb, sb in tr:
            xb=xb.to(device); yb=yb.to(device)                 # yb: (B,H,1)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb).squeeze(-1)                   # (B,H)
                targ = yb[...,0]                               # (B,H)
                loss = loss_fn(pred, targ)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        epoch_times.append(time.time()-et0)

        # quick val on one batch
        model.eval()
        with torch.no_grad():
            xb, yb, sb = next(iter(va))
            pv = model(xb.to(device)).squeeze(-1).detach().cpu().numpy()   # (B,H)
            yv = yb[...,0].detach().cpu().numpy()                           # (B,H)
            sids_b = sb.numpy()
            last_val_mae_norm = mae(yv.reshape(-1), pv.reshape(-1))
            p_raw, y_raw = denorm_horizon(pv, yv, sids_b, means, stds, H=args.H)
            last_val_mae_raw = mae(y_raw, p_raw)
        print(f"epoch {ep:02d} | {args.dataset} | pe={args.pe} | val MAE (raw over H) {last_val_mae_raw:.4f} | (z) {last_val_mae_norm:.4f} | {epoch_times[-1]:.1f}s")

    ckpt=f'checkpoints/ts_transformer_{args.dataset}_{args.pe}.pt'
    torch.save(model.state_dict(), ckpt)
    print('Saved ->', ckpt)

    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    header = ['timestamp','dataset','pe','normalize','L','H','stride','max_series','epochs','seed',
              'val_mae_raw','val_mae_z','avg_epoch_s','device','params_m','ckpt']
    row = [datetime.utcnow().isoformat(), args.dataset, args.pe, args.normalize, args.L, args.H, args.stride,
           args.max_series, args.epochs, args.seed, round(last_val_mae_raw,6), round(last_val_mae_norm,6),
           round(float(np.mean(epoch_times)),3), device, round(count_params_m(model),3), ckpt]
    write_header = not os.path.exists(args.log_csv)
    with open(args.log_csv,'a',newline='') as f:
        w=csv.writer(f); 
        if write_header: w.writerow(header)
        w.writerow(row)

if __name__=='__main__': 
    main()
