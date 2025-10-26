from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

AUTOFORMER_FILES = {
    "electricity":    "electricity.csv",
    "traffic":        "traffic.csv",
    "exchange_rate":  "exchange_rate.csv",
    "weather":        "weather.csv",
}

def _infer_sampling_steps(index: pd.DatetimeIndex):
    """Return sampling step in minutes (approx) and a human tag."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None, None
    deltas = (index[1:] - index[:-1]).to_series().dt.total_seconds().values
    sec = float(np.median(deltas))
    if sec <= 0:
        return None, None
    minutes = sec / 60.0
    if abs(minutes - 60)   < 1:  tag = "hourly"
    elif abs(minutes - 1440) < 60: tag = "daily"
    elif abs(minutes - 10)  < 1:  tag = "10min"
    elif abs(minutes - 15)  < 1:  tag = "15min"
    elif abs(minutes - 30)  < 1:  tag = "30min"
    else:                          tag = f"{minutes:.0f}min"
    return minutes, tag

def _is_business_daily(index: pd.DatetimeIndex) -> bool:
    if not isinstance(index, pd.DatetimeIndex):
        return False
    counts = pd.Series(index.weekday).value_counts(normalize=True)
    wknd_frac = counts.get(5, 0.0) + counts.get(6, 0.0)
    return wknd_frac < 0.02

def _default_periods_from_minutes(minutes: float, index: pd.DatetimeIndex | None):
    if minutes is None or minutes <= 0:
        return (24, 168)
    if abs(minutes - 1440) < 60:
        if index is not None and _is_business_daily(index):
            return (5, 22, 260)
        else:
            return (7, 30, 365)
    daily_steps = max(2, int(round(1440.0 / minutes)))
    weekly_steps = daily_steps * 7
    return (daily_steps, weekly_steps)

def _fft_suggest_periods(series: np.ndarray, max_lag: int, topk: int = 2):
    x = series.astype("float32")
    x = (x - x.mean()) / (x.std() + 1e-6)
    n = min(len(x), max_lag*4)
    if n < 8:
        return tuple()
    X = np.fft.rfft(x[:n]); power = (X * np.conj(X)).real
    freqs = np.fft.rfftfreq(n); power[0] = 0
    idx = np.argsort(power)[-topk:][::-1]
    out=[]
    for i in idx:
        if freqs[i] > 0:
            p = int(round(1.0 / freqs[i]))
            if 2 <= p <= max_lag and p not in out:
                out.append(p)
    return tuple(out)

class TimeSeriesPile:
    def __init__(self, root: str = "./data", cache_subdir: str = "hf_cache"):
        self.root = Path(root)
        self.cache_dir = Path(root) / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def available_datasets():
        return tuple(AUTOFORMER_FILES.keys())

    def load(self, name: str = "electricity") -> pd.DataFrame:
        if name not in AUTOFORMER_FILES:
            raise KeyError(f"Dataset '{name}' not in {list(AUTOFORMER_FILES.keys())}")
        fname = AUTOFORMER_FILES[name]
        repo_dir = Path(snapshot_download(
            repo_id="AutonLab/Timeseries-PILE",
            repo_type="dataset",
            allow_patterns=[f"forecasting/autoformer/{fname}"],
            cache_dir=str(self.cache_dir)
        ))
        fpath = repo_dir / "forecasting" / "autoformer" / fname
        df = pd.read_csv(fpath)

        ts_col = None
        for cand in ("date", "Datetime", "timestamp", "time"):
            if cand in df.columns:
                ts_col = cand; break
        if ts_col:
            df = df.rename(columns={ts_col: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.set_index("timestamp")

        df = df.select_dtypes(include=["number"]).astype("float32")
        return df

    def infer_periods(self, df: pd.DataFrame, topk_fft: int = 0):
        minutes, tag = None, None
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 3:
            minutes, tag = _infer_sampling_steps(df.index)
        base = _default_periods_from_minutes(minutes, df.index if isinstance(df.index, pd.DatetimeIndex) else None)
        if topk_fft > 0 and len(df.columns) > 0:
            col = df.columns[0]
            max_lag = max(base) if len(base) else min(1000, len(df)//4)
            suggested = _fft_suggest_periods(df[col].values, max_lag=max_lag, topk=topk_fft)
            merged = list(base) + [p for p in suggested if p not in base]
        else:
            merged = list(base)
        cleaned = []
        for p in merged:
            if p >= 2 and p not in cleaned:
                cleaned.append(p)
        return tuple(cleaned), (minutes, tag)

    def make_windows_from_df(self, df: pd.DataFrame, L=256, H=48, stride=1,
                             cols=None, max_series=None, normalize="none", return_meta=False):
        """
        Window each selected column as a separate series and stack.
        normalize: "none" or "zscore" (per-series)
        return_meta: if True, returns (X, Y, meta) where meta={"series_ids","means","stds"}
        """
        if cols is None:
            cols = list(df.columns)
        if max_series is not None:
            cols = cols[:max_series]

        # per-series stats
        means = np.zeros(len(cols), dtype="float32")
        stds  = np.ones(len(cols), dtype="float32")

        Xs, Ys, sids = [], [], []
        wid = 0
        for si, c in enumerate(cols):
            series = df[c].to_numpy(dtype="float32")
            m = float(np.nanmean(series))
            s = float(np.nanstd(series) + 1e-6)
            means[si] = m; stds[si] = s
            if normalize == "zscore":
                series = (series - m) / s

            # windows
            T = len(series)
            for start in range(0, T - L - H + 1, stride):
                Xs.append(series[start:start+L].reshape(L,1))
                Ys.append(series[start+L:start+L+H].reshape(H,1))
                sids.append(si)
                wid += 1

        X = np.stack(Xs).astype("float32")
        Y = np.stack(Ys).astype("float32")
        if return_meta:
            meta = {"series_ids": np.array(sids, dtype="int32"),
                    "means": means,
                    "stds": stds,
                    "cols": cols}
            return X, Y, meta
        return X, Y
