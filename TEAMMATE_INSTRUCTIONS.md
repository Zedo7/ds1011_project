# Electricity Experiments - Complete Setup & Run Instructions
# =============================================================

## Part 1: Environment Setup

### Step 1: Clone Repository
```bash
# Clone the repo
git clone <YOUR_REPO_URL>
cd ds1011_project

# Make sure you're on the right branch
git checkout main
git pull origin main
```

### Step 2: Install Anaconda/Miniconda
If you don't have conda installed:

**Windows:**
- Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Install and restart terminal

**Linux/Mac:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Step 3: Create Conda Environment
```bash
# Create environment with Python 3.10
conda create -n ds1011project python=3.10 -y

# Activate it
conda activate ds1011project
```

### Step 4: Install PyTorch with CUDA
**Check your CUDA version first:**
```bash
nvidia-smi
# Look for "CUDA Version: X.X" in top right
```

**Then install PyTorch:**

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (if no GPU):**
```bash
pip install torch torchvision torchaudio
```

**Verify PyTorch + CUDA:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Expected output:**
```
PyTorch: 2.x.x+cu118
CUDA available: True
CUDA version: 11.8
```

### Step 5: Install Required Packages
```bash
# Core ML packages
pip install numpy pandas scipy scikit-learn

# Visualization
pip install matplotlib seaborn

# Data handling
pip install pyyaml tqdm

# Additional utilities (if needed)
pip install jupyter ipython
```

**OR if there's a requirements.txt:**
```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation
```bash
# Test all imports
python -c "
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

print('✓ All packages imported successfully!')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ Pandas version: {pd.__version__}')
"
```

### Step 7: Create Required Directories
```bash
# Create directories if they don't exist
mkdir -p checkpoints
mkdir -p reports/tables
mkdir -p reports/figures
mkdir -p reports/logs
mkdir -p data
```

---

## Part 2: Running Electricity Experiments

### Overview
You need to complete the missing electricity experiments:
- **Seed 43:** 9 missing configs (~36 minutes)
- **Seed 44:** All 30 configs (~2 hours)
- **Total:** 39 experiments, ~2.5 hours

### Quick Start (Automatic)

**On Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File complete_electricity.ps1
```

**On Linux/Mac:**
```bash
# Convert PowerShell script to bash or run manually (see below)
```

The script will:
1. Check which experiments are missing from `results.csv`
2. Only run what's needed (seed 43 + seed 44)
3. Save results to `reports/tables/results.csv` and `reports/tables/extrapolation.csv`

### Manual Execution (If Script Fails)

#### Step 1: Check What's Missing
```bash
python -c "
import pandas as pd

try:
    df = pd.read_csv('reports/tables/results.csv')
    elec = df[df['dataset']=='electricity']
    
    print('Current Status:')
    print(f'  Seed 42: {len(elec[elec[\"seed\"]==42])}/30 configs')
    print(f'  Seed 43: {len(elec[elec[\"seed\"]==43])}/30 configs')
    print(f'  Seed 44: {len(elec[elec[\"seed\"]==44])}/30 configs')
    
    # Find missing seed 43 configs
    seed43 = elec[elec['seed']==43]
    all_configs = []
    for L in [168, 336]:
        for H in [24, 168, 336]:
            for pe in ['none', 'absolute', 'rope', 'xpos', 'alibi']:
                all_configs.append((L, H, pe))
    
    completed = set(zip(seed43['L'], seed43['H'], seed43['pe']))
    missing_43 = [c for c in all_configs if c not in completed]
    
    print(f'\nSeed 43 missing {len(missing_43)} configs:')
    for L, H, pe in missing_43:
        print(f'  L={L}, H={H}, pe={pe}')
    
    print(f'\nSeed 44 missing all 30 configs')
    
except FileNotFoundError:
    print('results.csv not found - starting fresh')
    print('Need to run all 90 experiments (seeds 42, 43, 44)')
"
```

#### Step 2: Run Missing Seed 43 Experiments
For each missing config from above, run:

```bash
# Example: If missing rope L=168 H=336
python train.py \
    --dataset electricity \
    --pe rope \
    --L 168 \
    --H 336 \
    --seed 43 \
    --epochs 3 \
    --batch_size 64 \
    --max_series 32 \
    --stride 120 \
    --normalize zscore \
    --device cuda \
    --lr 0.0003

# Save checkpoint with seed
cp checkpoints/ts_transformer_electricity_rope.pt \
   checkpoints/ts_transformer_electricity_rope_L168_H336_seed43.pt

# Evaluate
python eval_extrap.py \
    --dataset electricity \
    --pe rope \
    --L_train 168 \
    --H 336 \
    --eval_multipliers 1,2 \
    --max_series 32 \
    --normalize zscore \
    --device cuda \
    --ckpt checkpoints/ts_transformer_electricity_rope_L168_H336_seed43.pt
```

#### Step 3: Run All Seed 44 Experiments

**On Linux/Mac (Bash):**
```bash
for L in 168 336; do
    for H in 24 168 336; do
        for PE in none absolute rope xpos alibi; do
            echo "Running: seed=44, PE=$PE, L=$L, H=$H"
            
            # Train
            python train.py \
                --dataset electricity \
                --pe $PE \
                --L $L \
                --H $H \
                --seed 44 \
                --epochs 3 \
                --batch_size 64 \
                --max_series 32 \
                --stride 120 \
                --normalize zscore \
                --device cuda \
                --lr 0.0003
            
            # Save checkpoint
            cp checkpoints/ts_transformer_electricity_${PE}.pt \
               checkpoints/ts_transformer_electricity_${PE}_L${L}_H${H}_seed44.pt
            
            # Evaluate
            python eval_extrap.py \
                --dataset electricity \
                --pe $PE \
                --L_train $L \
                --H $H \
                --eval_multipliers 1,2 \
                --max_series 32 \
                --normalize zscore \
                --device cuda \
                --ckpt checkpoints/ts_transformer_electricity_${PE}_L${L}_H${H}_seed44.pt
        done
    done
done
```

**On Windows (PowerShell):**
```powershell
foreach ($L in @(168, 336)) {
    foreach ($H in @(24, 168, 336)) {
        foreach ($PE in @("none", "absolute", "rope", "xpos", "alibi")) {
            Write-Host "Running: seed=44, PE=$PE, L=$L, H=$H"
            
            # Train
            python train.py --dataset electricity --pe $PE --L $L --H $H --seed 44 --epochs 3 --batch_size 64 --max_series 32 --stride 120 --normalize zscore --device cuda --lr 0.0003
            
            # Save checkpoint
            Copy-Item "checkpoints/ts_transformer_electricity_${PE}.pt" "checkpoints/ts_transformer_electricity_${PE}_L${L}_H${H}_seed44.pt"
            
            # Evaluate
            python eval_extrap.py --dataset electricity --pe $PE --L_train $L --H $H --eval_multipliers 1,2 --max_series 32 --normalize zscore --device cuda --ckpt "checkpoints/ts_transformer_electricity_${PE}_L${L}_H${H}_seed44.pt"
        }
    }
}
```

---

## Part 3: Monitoring Progress

### Check Progress Anytime
```bash
python -c "
import pandas as pd
df = pd.read_csv('reports/tables/results.csv')
print('Electricity Experiments:')
print(f'  Seed 42: {len(df[df[\"seed\"]==42])}')
print(f'  Seed 43: {len(df[df[\"seed\"]==43])}')
print(f'  Seed 44: {len(df[df[\"seed\"]==44])}')
print(f'  Total: {len(df[df[\"dataset\"]==\"electricity\"])}')
"
```

**Expected when done:**
```
Electricity Experiments:
  Seed 42: 30
  Seed 43: 30
  Seed 44: 30
  Total: 90
```

### Watch GPU Usage
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

### Estimate Time Remaining
Based on average ~4 minutes per experiment:
- 1 experiment: ~4 minutes
- 10 experiments: ~40 minutes
- 30 experiments: ~2 hours
- 39 experiments: ~2.5 hours

---

## Part 4: Sending Results Back

### What to Send

**Critical Files (Must Send):**
1. `reports/tables/results.csv` - Training results
2. `reports/tables/extrapolation.csv` - Evaluation results

**Optional (Nice to Have):**
3. `checkpoints/*.pt` - Model checkpoints (large files, only if needed)

### Option A: Via GitHub (Recommended)
```bash
git add reports/tables/results.csv
git add reports/tables/extrapolation.csv
git commit -m "Add electricity seed 43 + 44 results"
git push origin main
```

### Option B: Via File Sharing
Upload to Google Drive/Dropbox and share link:
- `results.csv` (~100 KB)
- `extrapolation.csv` (~50 KB)

### Option C: Via Email
Attach the 2 CSV files (both small, <1 MB total)

---

## Part 5: Verification

Before sending results, verify completeness:

```bash
python -c "
import pandas as pd

# Load results
results_df = pd.read_csv('reports/tables/results.csv')
extrap_df = pd.read_csv('reports/tables/extrapolation.csv')

# Check electricity
elec_results = results_df[results_df['dataset']=='electricity']
elec_extrap = extrap_df[extrap_df['dataset']=='electricity']

print('='*60)
print('VERIFICATION REPORT')
print('='*60)

print('\nTraining Results (results.csv):')
for seed in [42, 43, 44]:
    count = len(elec_results[elec_results['seed']==seed])
    status = '✓' if count == 30 else '✗'
    print(f'  {status} Seed {seed}: {count}/30 configs')

print('\nEvaluation Results (extrapolation.csv):')
total_evals = len(elec_extrap)
expected_evals = 90 * 2  # 90 configs × 2 eval lengths (L_train and 2×L_train)
print(f'  Total evaluations: {total_evals} (expected ~{expected_evals})')

# Check MASE values are reasonable
mase_values = elec_extrap['MASE'].values
avg_mase = mase_values.mean()
print(f'\nMASE Quality Check:')
if 0.5 < avg_mase < 1.5:
    print(f'  ✓ Average MASE: {avg_mase:.3f} (looks good!)')
else:
    print(f'  ⚠ Average MASE: {avg_mase:.3f} (might have issues)')

# Summary
all_complete = all([
    len(elec_results[elec_results['seed']==42]) == 30,
    len(elec_results[elec_results['seed']==43]) == 30,
    len(elec_results[elec_results['seed']==44]) == 30
])

print('\n' + '='*60)
if all_complete:
    print('✓ ALL EXPERIMENTS COMPLETE! Ready to send results.')
else:
    print('✗ Some experiments missing. Check above for details.')
print('='*60)
"
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size
```bash
python train.py ... --batch_size 32  # instead of 64
```

### Issue: "ModuleNotFoundError"
**Solution:** Install missing package
```bash
pip install <package_name>
```

### Issue: Training Hangs
**Solution:** 
1. Check GPU: `nvidia-smi`
2. Kill process: `Ctrl+C`
3. Restart from that config

### Issue: MASE Values Look Wrong (>5)
**Solution:** Check that `eval_extrap.py` has the corrected MASE calculation
```bash
grep -A 10 "compute_seasonal_naive_mae" eval_extrap.py
```

Should compute naive baseline on TEST data only, not entire dataset.

### Issue: Script Not Found
**Solution:** Make sure you're in project root directory
```bash
pwd  # Should show: .../ds1011_project
ls   # Should show: train.py, eval_extrap.py, etc.
```

---

## Expected Timeline

- **Environment Setup:** 10-15 minutes (first time only)
- **Seed 43 (9 configs):** ~35 minutes
- **Seed 44 (30 configs):** ~2 hours
- **Total:** ~2.5 hours

---

## Questions?
Contact via Slack/Discord/Email

Thanks for helping! 🚀

---

## Quick Reference Commands

```bash
# Activate environment
conda activate ds1011project

# Check status
python -c "import pandas as pd; df=pd.read_csv('reports/tables/results.csv'); print(df[df['dataset']=='electricity'].groupby('seed').size())"

# Run experiments (Windows)
powershell -ExecutionPolicy Bypass -File complete_electricity.ps1

# Push results
git add reports/tables/*.csv
git commit -m "Add electricity results"
git push origin main
```