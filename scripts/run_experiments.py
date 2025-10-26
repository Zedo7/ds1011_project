import argparse, subprocess, sys, os, time, yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def run(cmd, dry=False):
    print(">>", " ".join(cmd))
    if dry: return 0
    return subprocess.call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(ROOT/"configs/experiments.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    ap.add_argument("--skip-existing", action="store_true", help="skip if checkpoint already exists")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dfl = cfg.get("defaults", {})
    datasets_cfg = cfg["datasets"]
    selected = cfg.get("selection", {}).get("datasets", list(datasets_cfg.keys()))
    seeds = cfg.get("selection", {}).get("seeds", [dfl.get("seed", 42)])
    epochs_override = cfg.get("selection", {}).get("epochs", None)

    pe_list = cfg["pe_list"]
    extrap = cfg.get("extrapolation", {"enabled": True, "multipliers":[1,2]})

    results_csv = dfl.get("results_csv", "reports/tables/results.csv")
    extrap_csv  = dfl.get("extrap_csv",  "reports/tables/extrapolation.csv")

    for ds_name in selected:
        ds = datasets_cfg[ds_name]
        L = int(ds["L"]); H = int(ds["H"]); stride=int(ds["stride"]); maxs=int(ds["max_series"])

        for pe in pe_list:
            pe_name = pe["name"]
            use_pb = pe.get("use_periodic_bias", False)
            pb_lams = pe.get("periodic_lambdas", [])

            for seed in seeds:
                epochs = int(epochs_override if epochs_override is not None else dfl.get("epochs", 3))
                cmd = [
                    sys.executable, str(ROOT/"train.py"),
                    "--dataset", ds_name,
                    "--pe", pe_name,
                    "--epochs", str(epochs),
                    "--L", str(L), "--H", str(H),
                    "--stride", str(stride),
                    "--max_series", str(maxs),
                    "--normalize", dfl.get("normalize","zscore"),
                    "--batch_size", str(dfl.get("batch_size", 64)),
                    "--lr", str(dfl.get("lr", 3e-4)),
                    "--seed", str(seed),
                    "--device", dfl.get("device","cuda"),
                    "--log_csv", results_csv
                ]
                if pe_name in ("rope","xpos"):  # alibi handled inside train via --pe alibi
                    # no extra args needed unless you want multi-base RoPE
                    pass
                if use_pb and pe_name in ("rope","none","absolute","xpos","alibi"):
                    cmd += ["--use_periodic_bias", "--periodic_lambdas", ",".join(map(str,pb_lams))]

                # checkpoint path matches train.py naming
                ckpt = ROOT / f"checkpoints/ts_transformer_{ds_name}_{pe_name}.pt"
                # if multi-seed, avoid overwrite by appending seed
                if len(seeds) > 1:
                    ckpt_seeded = ROOT / f"checkpoints/ts_transformer_{ds_name}_{pe_name}_seed{seed}.pt"
                else:
                    ckpt_seeded = ckpt

                # train (skip if desired and exists)
                if args.skip_existing and ckpt_seeded.exists():
                    print(f"SKIP (exists): {ckpt_seeded}")
                else:
                    rc = run(cmd, dry=args.dry_run)
                    if rc != 0:
                        print(f"[ERROR] training failed for {ds_name} / {pe_name} / seed {seed}", file=sys.stderr)
                        continue
                    # rename to seed-specific if needed
                    if len(seeds) > 1 and ckpt.exists():
                        ckpt.replace(ckpt_seeded)

                # extrapolation
                if extrap.get("enabled", True):
                    mults = extrap.get("multipliers", [1,2])
                    cmd_eval = [
                        sys.executable, str(ROOT/"eval_extrap.py"),
                        "--dataset", ds_name,
                        "--pe", pe_name,
                        "--ckpt", str(ckpt_seeded),
                        "--L_train", str(L),
                        "--H", str(H),
                        "--eval_multipliers", ",".join(map(str, mults)),
                        "--max_series", str(maxs),
                        "--normalize", dfl.get("normalize","zscore"),
                        "--device", dfl.get("device","cuda"),
                        "--log_csv", extrap_csv
                    ]
                    rc = run(cmd_eval, dry=args.dry_run)
                    if rc != 0:
                        print(f"[ERROR] eval_extrap failed for {ds_name} / {pe_name} / seed {seed}", file=sys.stderr)

    print("Done.")

if __name__ == "__main__":
    main()
