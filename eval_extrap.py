import argparse, csv, os
from datetime import datetime
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from src.data.timeseries_pile import TimeSeriesPile
from src.models.ts_transformer import TSTransformer
from src.utils.metrics import mae, mase, smape, rmse
from tqdm import tqdm


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
    
    # Validate indices
    if len(sids) > 0 and sids.max() >= len(means):
        raise ValueError(f"Invalid series ID {sids.max()}, only {len(means)} series")
    
    srep = np.repeat(sids, H)
    m = means[srep]; s = stds[srep]
    return pred * s + m, targ * s + m

def compute_seasonal_naive_mae(y_test, seasonality=24):
    """
    Compute seasonal naive forecast error on TEST DATA ONLY.
    
    Args:
        y_test: Ground truth test values (1D array)
        seasonality: Seasonal period
    
    Returns:
        MAE of seasonal naive forecast
    """
    if len(y_test) <= seasonality:
        # Fallback to random walk
        return np.mean(np.abs(np.diff(y_test)))
    
    # Seasonal naive: y_pred[t] = y_true[t - seasonality]
    # For horizon H forecast, we need to compare like-for-like
    # If evaluating H-step ahead, naive should also be H-step
    
    # Simple approach: compare adjacent seasonal lags
    errors = []
    for i in range(seasonality, len(y_test)):
        errors.append(abs(y_test[i] - y_test[i - seasonality]))
    
    return float(np.mean(errors))

def eval_at_L(model, ds, df, L, H, max_series=8, batch_size=256, 
              device='cpu', normalize='zscore', seasonality=24):
    """
    Evaluate model at given context length L.
    
    Returns:
        dict with keys: 'mae', 'rmse', 'smape', 'mase'
    """
    X, Y, meta = make_windows(ds, df, L, H, max_series, normalize)
    sids, means, stds = meta["series_ids"], meta["means"], meta["stds"]
    
    dl = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(sids)), 
        batch_size=batch_size
    )
    
    P, T = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, sb in tqdm(dl, desc=f"Eval L={L}", leave=False):
            pv = model(xb.to(device)).squeeze(-1).detach().cpu().numpy()  # (B,H)
            yv = yb[..., 0].numpy()                                        # (B,H)
            p_raw, y_raw = denorm_horizon(pv, yv, sb.numpy(), means, stds, H=H)
            P.append(p_raw)
            T.append(y_raw)
    
    yhat = np.concatenate(P, 0).reshape(-1)
    y = np.concatenate(T, 0).reshape(-1)
    
    # Compute metrics
    mae_val = mae(y, yhat)
    rmse_val = rmse(y, yhat)
    smape_val = smape(y, yhat)
    
    # CORRECTED MASE: compute naive baseline on SAME test data
    naive_mae = compute_seasonal_naive_mae(y, seasonality)
    mase_val = mae_val / naive_mae if naive_mae > 0 else float('inf')
    
    return {
        'mae': mae_val,
        'rmse': rmse_val,
        'smape': smape_val,
        'mase': mase_val,
        'naive_mae': naive_mae  # For debugging
    }

def main():
    p = argparse.ArgumentParser()
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
    args = p.parse_args()

    device = 'cuda' if (args.device=='auto' and torch.cuda.is_available()) else (
        args.device if args.device!='auto' else 'cpu'
    )

    ds = TimeSeriesPile()
    df = ds.load(args.dataset)
    
    # Infer seasonality for MASE
    periods, (minutes, _) = ds.infer_periods(df, topk_fft=0)
    seasonality = periods[0] if periods else 24  # Use first detected period
    
    m = TSTransformer(horizon=args.H).to(device)
    ckpt_path = args.ckpt if args.ckpt else f'checkpoints/ts_transformer_{args.dataset}_{args.pe}.pt'
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    m.load_state_dict(state)

    mults = [int(x) for x in args.eval_multipliers.split(',')]
    results = []
    baseline_metrics = None
    
    for k in mults:
        L_eval = args.L_train * k
        
        metrics = eval_at_L(
            m, ds, df, L=L_eval, H=args.H, 
            max_series=args.max_series, device=device, 
            normalize=args.normalize, seasonality=seasonality
        )
        
        if baseline_metrics is None:
            baseline_metrics = metrics
        
        # Compute LES
        les = metrics['mae'] / (baseline_metrics['mae'] + 1e-12)
        
        results.append((L_eval, metrics, les))
        
        print(f"L_eval={L_eval:4d} | MAE={metrics['mae']:7.2f} | "
              f"RMSE={metrics['rmse']:7.2f} | sMAPE={metrics['smape']:5.1f}% | "
              f"MASE={metrics['mase']:5.2f} | LES={les:.3f}")

    # Write to CSV
    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    write_header = not os.path.exists(args.log_csv)
    
    with open(args.log_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                'timestamp', 'dataset', 'pe', 'ckpt', 'normalize', 
                'L_train', 'H', 'seasonality', 'L_eval', 
                'MAE', 'RMSE', 'sMAPE', 'MASE', 'LES'
            ])
        
        for (L_eval, metrics, les) in results:
            w.writerow([
                datetime.utcnow().isoformat(),
                args.dataset,
                args.pe,
                ckpt_path,
                args.normalize,
                args.L_train,
                args.H,
                seasonality,
                L_eval,
                round(metrics['mae'], 4),
                round(metrics['rmse'], 4),
                round(metrics['smape'], 2),
                round(metrics['mase'], 4),
                round(les, 4)
            ])

if __name__ == '__main__': 
    main()
