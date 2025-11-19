# Positional Encodings for Time-Series Extrapolation

Investigating whether positional encoding methods from NLP (RoPE, ALiBi, xPos) improve long-horizon time-series forecasting.

## Setup
```bash
# Install dependencies
pip install torch numpy pandas huggingface-hub pyyaml scipy scikit-learn tqdm matplotlib

# Create directories
mkdir -p data checkpoints reports/tables logs
```

## Quick Start

### Single Experiment
```bash
# Train RoPE on electricity (7 days → 1 day forecast)
python train.py \
    --dataset electricity \
    --pe rope \
    --L 168 \
    --H 24 \
    --epochs 3 \
    --use_periodic_bias

# Evaluate extrapolation
python eval_extrap.py \
    --dataset electricity \
    --pe rope \
    --L_train 168 \
    --H 24 \
    --eval_multipliers 1,2
```

### Full Experiment Grid
```bash
# Run all experiments for weather dataset
python scripts/run_temporal_matrix.py \
    --cfg configs/temporal_matrix_weather.yaml

# Run all experiments for traffic dataset  
python scripts/run_temporal_matrix.py \
    --cfg configs/temporal_matrix_traffic.yaml
```

## Datasets

- **Electricity**: 32 series, hourly, strong daily/weekly cycles
- **Traffic**: 32 series, hourly, moderate periodicity
- **Weather**: 21 series, 10-min sampling, seasonal trends

## Positional Encodings Tested

- **None**: Causal mask only (baseline)
- **Absolute**: Sinusoidal (Vaswani et al., 2017)
- **RoPE**: Rotary position embedding (Su et al., 2021)
- **xPos**: Extrapolatable position (Sun et al., 2023)
- **ALiBi**: Attention with linear biases (Press et al., 2021)

## Evaluation Metrics

- **MAE**: Mean absolute error (lower is better)
- **RMSE**: Root mean squared error
- **sMAPE**: Symmetric MAPE (0-200%, lower is better)
- **MASE**: Mean absolute scaled error (<1.0 means better than seasonal naive)
- **LES**: Length extrapolation score (≈1.0 means no degradation)

## Results

Results are logged to `reports/tables/extrapolation.csv`.

View with:
```bash
import pandas as pd
df = pd.read_csv('reports/tables/extrapolation.csv')
print(df.groupby(['dataset', 'pe'])['MASE'].mean())
```

## Project Structure
```
.
├── configs/                    # YAML experiment configs
│   ├── temporal_matrix_electricity.yaml
│   ├── temporal_matrix_traffic.yaml
│   └── temporal_matrix_weather.yaml
├── src/
│   ├── data/                  # Data loading
│   │   └── timeseries_pile.py
│   ├── models/                # Model architecture
│   │   ├── attention.py       # PE implementations
│   │   └── ts_transformer.py
│   ├── pe/                    # Positional encoding helpers
│   └── utils/                 # Metrics
├── scripts/
│   └── run_temporal_matrix.py # Experiment automation
├── train.py                   # Training script
└── eval_extrap.py             # Evaluation script
```

## Citation
```bibtex
@article{guo2024positional,
  title={Disentangling Positional Methods for Long-Horizon Time-Series Extrapolation},
  author={Guo, Zhilin and Ku, Chuhan and Wang, Alice and Bian, Chengjun},
  year={2024}
}
```

## References

- Sun et al. (2023): A Length-Extrapolatable Transformer
- Press et al. (2021): Train Short, Test Long (ALiBi)
- Su et al. (2021): RoFormer (RoPE)
```

---

## Priority 8: Create requirements.txt (5 minutes)

### File NEW: `requirements.txt`
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
huggingface-hub>=0.16.0
pyyaml>=6.0
scipy>=1.10.0
scikit-learn>=1.3.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
