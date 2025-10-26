# DS1011 — Positional Encodings for Time‑Series

This repo compares positional encodings (**None, Absolute, RoPE, XPOS, ALiBi**) for multivariate **time‑series forecasting** on four subsets of Timeseries‑PILE: `electricity`, `traffic`, `exchange_rate`, and `weather`.

**Model:** Transformer **encoder** + **H‑step decoder** (predicts the next **H** steps at once from the final context token). Experiments are GPU‑first, config‑driven, and logged to CSV.

---

## What’s implemented

* **Positional encodings:** `none`, `absolute`, `rope`, `xpos`, `alibi` (+ optional **periodic seasonal bias**, e.g., daily/weekly).
* **Normalization:** per‑series **z‑score** at training time; we report **raw** and **z‑space** MAE.
* **Datasets:** pulled on demand from Hugging Face (Autoformer CSVs) and cached under `data/hf_cache/` (git‑ignored).
* **Training:** CUDA/AMP ready; reproducible seeds; checkpoints saved under `checkpoints/`.
* **Evaluation:**

  * Validation **H‑step MAE** (raw & z).
  * **LES (Length Extrapolation Score)** = `MAE(L_eval)/MAE(L_train)`; ≈1 → length‑neutral, <1 → helps, >1 → hurts when context grows.
* **Plots:** simple bar/curve figures to `reports/figures/`.

---

## Environment

```bash
# Activate your conda env (created earlier)
conda activate ds1011project

# If PyTorch isn’t CUDA‑enabled in this env, install the CUDA build (choose one):
# conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.4
# or (pip route)
# pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Python deps
pip install -r requirements.txt
```

**Verify GPU:**

```bash
python - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda runtime:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

---

## Quick start (single run)

Train **RoPE** on **Electricity** with context **L=512**, horizon **H=96**, z‑score normalization, and periodic bias (GPU):

```bash
python train.py \
  --dataset electricity --pe rope --epochs 3 \
  --L 512 --H 96 --stride 30 --max_series 16 \
  --normalize zscore --device cuda \
  --use_periodic_bias --periodic_lambdas 0.2,0.1
```

Extrapolate to longer context (compute **LES** at 512→1024):

```bash
python eval_extrap.py \
  --dataset electricity --pe rope \
  --L_train 512 --H 96 --eval_multipliers 1,2 \
  --max_series 16 --normalize zscore --device cuda
```

**Outputs** append to:

* `reports/tables/results.csv`
* `reports/tables/extrapolation.csv`

---

## Config‑driven sweeps

Describe your grid in `configs/experiments.yaml`, then run:

```bash
# Full grid
python scripts/run_experiments.py --cfg configs/experiments.yaml

# Resume later and skip already‑trained checkpoints
python scripts/run_experiments.py --cfg configs/experiments.yaml --skip-existing
```

**Tip:** To force retraining at new `L/H/normalize`, move or delete existing `.pt` files in `checkpoints/` (or use the cleanup commands below).

---

## Plots

```bash
python plots/create_figures.py
# Figures saved under reports/figures/
```

---

## Start fresh (clear artifacts)

Tables and figures are recreated automatically on the next run.

```bash
# tables
rm -f reports/tables/results.csv reports/tables/extrapolation.csv

# figures
rm -rf reports/figures/*

# checkpoints (delete all)
rm -f checkpoints/*.pt
```

**Windows/PowerShell equivalents:**

```powershell
if (Test-Path .\reports\tables\results.csv)       { Remove-Item .\reports\tables\results.csv -Force }
if (Test-Path .\reports\tables\extrapolation.csv) { Remove-Item .\reports\tables\extrapolation.csv -Force }
if (Test-Path .\reports\figures) { Remove-Item .\reports\figures\* -Recurse -Force }
Get-ChildItem .\checkpoints -Filter *.pt | Remove-Item -Force
```

---

## Notes & guidance

* On **Electricity**, LES ≈ 1.0 when going **512→1024** for **H=96** often means 512 steps already capture the weekly/monthly signal. To stress length‑generalization:

  * Train at **L=256**, then evaluate at **512/1024**; or
  * Increase **H** (e.g., **H=168**).
* Keep track of **seed**, commit hash, and `configs/experiments.yaml` for reproducibility.
* Periodic bias (`--use_periodic_bias`) is most relevant for **hourly/10‑min** datasets.

---

## Repository layout (key files)

```
configs/                  # experiments.yaml (grid)
scripts/run_experiments.py
src/                      # data/ models/ pe/ utils/
train.py                  # H-step training (raw & z MAE)
eval_extrap.py            # length extrapolation (LES)
reports/tables/           # results.csv, extrapolation.csv
reports/figures/          # generated plots
checkpoints/              # model weights (.pt)
```

---

## Git: push your changes

```bash
git status
git add -A
git commit -m "Docs & experiments: update README, configs, and scripts"
# Push to the current branch
branch=$(git rev-parse --abbrev-ref HEAD)
git push origin "$branch"
```

**PowerShell version:**

```powershell
git status
git add -A
git commit -m "Docs & experiments: update README, configs, and scripts" 2>$null
$branch = git rev-parse --abbrev-ref HEAD
git push origin $branch
```

---

## Troubleshooting GPU (Windows)

* `nvidia-smi` should show your GPU & driver.
* PyTorch must report `cuda available: True` and a CUDA runtime (e.g., 12.4).
* If not, install the CUDA build of PyTorch into the `ds1011project` env (see *Environment* above).

---

## License

Add a license here (e.g., MIT) if you plan to release the code.
