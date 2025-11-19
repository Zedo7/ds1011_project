import argparse, time, csv, os
from datetime import datetime
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from src.data.timeseries_pile import TimeSeriesPile
from src.models.ts_transformer import TSTransformer
from src.utils.metrics import mae
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


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

    # Create directories FIRST
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('reports/tables', exist_ok=True)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = 'cuda' if (args.device=='auto' and torch.cuda.is_available()) else (
        args.device if args.device!='auto' else 'cpu'
    )
    
    # Setup logging to file AND console
    log_file = f'logs/train_{args.dataset}_{args.pe}.log'
    file_handler = logging.FileHandler(log_file, mode='w')  # 'w' = overwrite each run
    console_handler = logging.StreamHandler()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    logger = logging.getLogger(__name__)
    
    # LOAD DATA
    ds = TimeSeriesPile()
    df = ds.load(args.dataset)
    
    # Detect periods for periodic bias
    periods, _ = ds.infer_periods(df, topk_fft=1)
    logger.info(f"Detected periods: {periods}")
    
    if args.use_periodic_bias:
        pb_periods = tuple(periods[:2])
        pb_lambdas = tuple(float(x) for x in args.periodic_lambdas.split(',')[:len(pb_periods)])
        logger.info(f"Periodic bias enabled: periods={pb_periods}, lambdas={pb_lambdas}")
    else:
        pb_periods, pb_lambdas = (), ()
        logger.info("Periodic bias disabled")

    # Create windows
    logger.info(f"Creating windows: L={args.L}, H={args.H}, stride={args.stride}, max_series={args.max_series}")
    t0 = time.time()
    X, Y, meta = ds.make_windows_from_df(
        df, L=args.L, H=args.H, stride=args.stride, max_series=args.max_series,
        normalize=args.normalize, return_meta=True
    )
    sids = meta["series_ids"]
    means = meta["means"]
    stds = meta["stds"]
    logger.info(f"Windows created: X={X.shape}, Y={Y.shape} in {time.time()-t0:.1f}s")
    # split
    n_hold = min(2000, max(200, len(X)//10))
    Xtr, Ytr, S_tr = X[:-n_hold], Y[:-n_hold], sids[:-n_hold]
    Xva, Yva, S_va = X[-n_hold:], Y[-n_hold:], sids[-n_hold:]

    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr), torch.from_numpy(S_tr)),
                    batch_size=args.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva), torch.from_numpy(S_va)),
                    batch_size=args.batch_size)

    # Create model
    head_bases = [float(x) for x in args.multi_base.split(',')] if args.multi_base else None
    model = TSTransformer(
        d_model=256, n_heads=8, n_layers=4, dropout=0.1,
        pe=args.pe, rope_base=args.rope_base, xpos_gamma=args.xpos_gamma, head_bases=head_bases,
        use_alibi=(args.pe=='alibi'), periodic_periods=pb_periods, periodic_lambdas=pb_lambdas,
        in_features=1, out_features=1, horizon=args.H
    ).to(device)
    
    # NOW we can log model info
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset}, shape={df.shape}")
    logger.info(f"PE: {args.pe}, periodic_bias={args.use_periodic_bias}")
    logger.info(f"Model params: {count_params_m(model):.2f}M parameters")   

# Optimizer and loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6
    )
    
    # Early stopping
    best_val_mae = float('inf')
    patience = 5
    epochs_without_improvement = 0
    best_model_state = None

    epoch_times = []
    last_val_mae_norm = None
    last_val_mae_raw = None

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for ep in range(1, args.epochs + 1):
        et0 = time.time()
        
        # ============ TRAINING ============
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for xb, yb, sb in tqdm(tr, desc=f"Epoch {ep}/{args.epochs}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb).squeeze(-1)              # (B, H)
                targ = yb[..., 0]                         # (B, H)
                loss = loss_fn(pred, targ)
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(opt)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        epoch_time = time.time() - et0
        epoch_times.append(epoch_time)
        
        # ============ VALIDATION (FULL) ============
        model.eval()
        val_preds = []
        val_targets = []
        val_sids = []
        
        with torch.no_grad():
            for xb, yb, sb in tqdm(va, desc="Validation", leave=False):
                xb = xb.to(device)
                pv = model(xb).squeeze(-1).detach().cpu().numpy()   # (B, H)
                yv = yb[..., 0].detach().cpu().numpy()              # (B, H)
                
                val_preds.append(pv)
                val_targets.append(yv)
                val_sids.append(sb.numpy())
        
        # Concatenate all batches
        val_preds = np.concatenate(val_preds, axis=0)       # (N, H)
        val_targets = np.concatenate(val_targets, axis=0)   # (N, H)
        val_sids = np.concatenate(val_sids, axis=0)         # (N,)
        
        # Compute metrics on z-scored data
        last_val_mae_norm = mae(val_targets.reshape(-1), val_preds.reshape(-1))
        
        # Denormalize and compute raw MAE
        pred_raw, targ_raw = denorm_horizon(
            val_preds, val_targets, val_sids, means, stds, H=args.H
        )
        last_val_mae_raw = mae(targ_raw, pred_raw)
        
        logger.info(
            f"Epoch {ep:02d}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_MAE={last_val_mae_raw:.2f} (raw) | "
            f"{last_val_mae_norm:.4f} (z) | "
            f"time={epoch_time:.1f}s"
        )
        
        # Learning rate scheduling
        scheduler.step(last_val_mae_raw)
        
        # Early stopping
        if last_val_mae_raw < best_val_mae:
            best_val_mae = last_val_mae_raw
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            logger.info(f"  ✓ New best validation MAE: {best_val_mae:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"  → No improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {ep} epochs")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model (val MAE = {best_val_mae:.4f})")
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)

    ckpt_tag = getattr(args, "ckpt_tag", "")
    ckpt = f"checkpoints/ts_transformer_{args.dataset}_{args.pe}" + (f"_{ckpt_tag}" if ckpt_tag else "") + ".pt"
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



