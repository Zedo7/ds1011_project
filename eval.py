import argparse, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from src.data.timeseries_pile import TimeSeriesPile
from src.models.ts_transformer import TSTransformer
from src.utils.metrics import mae, rmse, smape, mase

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', default='electricity', choices=TimeSeriesPile.available_datasets())
    p.add_argument('--L_eval', type=int, default=512)
    p.add_argument('--H', type=int, default=48)
    p.add_argument('--ckpt', default='')
    args=p.parse_args()

    ds=TimeSeriesPile()
    df=ds.load(args.dataset)
    X,Y = ds.make_windows_from_df(df, L=args.L_eval, H=args.H, stride=1, max_series=8)
    dl=DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=128)

    if not args.ckpt:
        args.ckpt = f'checkpoints/ts_transformer_{args.dataset}_none.pt'
    m=TSTransformer(); m.load_state_dict(torch.load(args.ckpt, map_location='cpu')); m.eval()

    P,T = [],[]
    with torch.no_grad():
        for xb,yb in dl:
            P.append(m(xb).squeeze(-1).numpy()); T.append(yb[:, :1, 0].numpy())
    yhat=np.concatenate(P,0); y=np.concatenate(T,0)
    print("MAE", mae(y,yhat), "RMSE", rmse(y,yhat), "sMAPE", smape(y,yhat), "MASE", mase(y,yhat))

if __name__=='__main__': main()
