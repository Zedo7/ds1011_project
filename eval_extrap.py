import argparse, csv, os
from datetime import datetime
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from src.data.timeseries_pile import TimeSeriesPile
from src.models.ts_transformer import TSTransformer
from src.utils.metrics import mae

def make_windows(ds, df, L, H, max_series, normalize):
    X, Y, meta = ds.make_windows_from_df(
        df, L=L, H=H, stride=1, max_series=max_series,
        normalize=normalize, return_meta=True
    )
    return X, Y, meta

def denorm_horizon(pred, targ, sids, means, stds, H):
    pred = np.asarray(pred).reshape(-1)
    targ = np.asarray(targ).reshape(-1)
    sids = np.asarray(sids).astype("int64").reshape(-1)
    srep = np.repeat(sids, H)
    m = means[srep]; s = stds[srep]
    return pred * s + m, targ * s + m

def eval_at_L(model, ds, df, L, H, max_series=8, batch_size=256, device='cpu', normalize='zscore'):
    X, Y, meta = make_windows(ds, df, L, H, max_series, normalize)
    sids, means, stds = meta["series_ids"], meta["means"], meta["stds"]
    dl=DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(sids)), batch_size=batch_size)
    P,T=[],[]
    model.eval()
    with torch.no_grad():
        for xb,yb,sb in dl:
            pv = model(xb.to(device)).squeeze(-1).detach().cpu().numpy()  # (B,H)
            yv = yb[...,0].numpy()                                        # (B,H)
            p_raw, y_raw = denorm_horizon(pv, yv, sb.numpy(), means, stds, H=H)
            P.append(p_raw); T.append(y_raw)
    yhat=np.concatenate(P,0).reshape(-1); y=np.concatenate(T,0).reshape(-1)
    return mae(y,yhat)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', default='electricity', choices=TimeSeriesPile.available_datasets())
    p.add_argument('--pe', default='none', choices=['none','absolute','rope','xpos','alibi'])
    p.add_argument('--ckpt', default='')
    p.add_argument('--L_train', type=int, default=256)
    p.add_argument('--H', type=int, default=24)
    p.add_argument('--eval_multipliers', type=str, default='1,2,4')
    p.add_argument('--max_series', type=int, default=8)
    p.add_argument('--normalize', default='zscore', choices=['none','zscore'])
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    p.add_argument('--log_csv', default='reports/tables/extrapolation.csv')
    args=p.parse_args()

    device = 'cuda' if (args.device=='auto' and torch.cuda.is_available()) else (args.device if args.device!='auto' else 'cpu')

    ds=TimeSeriesPile(); df=ds.load(args.dataset)
    m=TSTransformer(horizon=args.H).to(device)
    state = torch.load(args.ckpt if args.ckpt else f'checkpoints/ts_transformer_{args.dataset}_{args.pe}.pt',
                       map_location=device, weights_only=True)
    m.load_state_dict(state)

    mults=[int(x) for x in args.eval_multipliers.split(',')]
    results=[]; mae_base=None
    for k in mults:
        L_eval = args.L_train * k
        e = eval_at_L(m, ds, df, L=L_eval, H=args.H, max_series=args.max_series, device=device, normalize=args.normalize)
        if mae_base is None: mae_base = e
        les = float(e)/float(mae_base+1e-12)
        results.append((L_eval, e, les))
        print(f"L_eval={L_eval} | MAE={e:.4f} | LES={les:.3f}")

    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    write_header = not os.path.exists(args.log_csv)
    with open(args.log_csv,'a',newline='') as f:
        w=csv.writer(f)
        if write_header: w.writerow(['timestamp','dataset','pe','ckpt','normalize','L_train','H','L_eval','MAE','LES'])
        for (L_eval, e, les) in results:
            w.writerow([datetime.utcnow().isoformat(), args.dataset, args.pe,
                        args.ckpt or f'checkpoints/ts_transformer_{args.dataset}_{args.pe}.pt',
                        args.normalize, args.L_train, args.H, L_eval, round(e,6), round(les,6)])

if __name__=='__main__': 
    main()
