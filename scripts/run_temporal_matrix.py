import argparse, subprocess, sys, os, yaml, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# ensure repo root on sys.path for "import src.*"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def sh(cmd):
    print(">>", " ".join(cmd)); return subprocess.call(cmd)

def hours_to_steps(hours, minutes_per_step):
    return max(2, int(round((hours*60.0)/minutes_per_step)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(ROOT/"configs/temporal_matrix.yaml"))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--normalize", default=None)
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dfl = cfg.get("defaults", {})
    pe_list = cfg["pe_list"]
    pb_on_pe = set(cfg.get("pb_on_pe", []))
    eval_mults = dfl.get("eval_multipliers", [1,2])

    from src.data.timeseries_pile import TimeSeriesPile
    ds_loader = TimeSeriesPile()

    for ds_name, ds_cfg in cfg["datasets"].items():
        # infer sampling
        df = ds_loader.load(ds_name)
        _, (mins, tag) = ds_loader.infer_periods(df, topk_fft=0)
        max_series = int(ds_cfg.get("max_series", 16))
        stride_steps = hours_to_steps(int(ds_cfg.get("stride_h", 30)), mins)

        for Lh in ds_cfg["input_windows_h"]:
            L = hours_to_steps(int(Lh), mins)
            for Hh in ds_cfg["horizons_h"]:
                H = hours_to_steps(int(Hh), mins)

                for pe in pe_list:
                    use_pb = (pe in pb_on_pe)
                    epochs = int(args.epochs or dfl.get("epochs", 3))
                    device = args.device or dfl.get("device", "cuda")
                    normalize = args.normalize or dfl.get("normalize", "zscore")

                    # default ckpt name written by train.py
                    ckpt_default = ROOT / f"checkpoints/ts_transformer_{ds_name}_{pe}.pt"
                    # unique name we want to keep
                    ckpt_unique  = ROOT / f"checkpoints/ts_transformer_{ds_name}_{pe}_L{L}_H{H}.pt"

                    if args.skip_existing and ckpt_unique.exists():
                        print(f"SKIP exist: {ckpt_unique}")
                    else:
                        cmd = [
                            sys.executable, str(ROOT/"train.py"),
                            "--dataset", ds_name, "--pe", pe,
                            "--epochs", str(epochs),
                            "--L", str(L), "--H", str(H),
                            "--stride", str(stride_steps),
                            "--max_series", str(max_series),
                            "--normalize", normalize,
                            "--batch_size", str(dfl.get("batch_size", 64)),
                            "--lr", str(dfl.get("lr", 3e-4)),
                            "--device", device,
                        ]
                        if use_pb:
                            cmd += ["--use_periodic_bias", "--periodic_lambdas",
                                    ",".join(map(str, dfl.get("pb_lambdas", [0.2,0.1])))]
                        rc = sh(cmd)
                        if rc != 0:
                            print(f"[ERROR] train failed: {ds_name} {pe} L={L} H={H}", file=sys.stderr)
                            continue

                        # rename/move default ckpt to unique name
                        if ckpt_default.exists():
                            try:
                                shutil.copy2(ckpt_default, ckpt_unique)
                            except Exception as e:
                                print(f"[WARN] could not copy ckpt: {e}", file=sys.stderr)
                        else:
                            print(f"[WARN] default ckpt not found: {ckpt_default}", file=sys.stderr)

                    # eval (LES)
                    cmd_eval = [
                        sys.executable, str(ROOT/"eval_extrap.py"),
                        "--dataset", ds_name, "--pe", pe,
                        "--ckpt", str(ckpt_unique if ckpt_unique.exists() else ckpt_default),
                        "--L_train", str(L), "--H", str(H),
                        "--eval_multipliers", ",".join(map(str, eval_mults)),
                        "--max_series", str(max_series),
                        "--normalize", normalize, "--device", device
                    ]
                    sh(cmd_eval)

    print("Done.")
if __name__ == "__main__":
    main()
