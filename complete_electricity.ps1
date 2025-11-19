# Complete Missing Electricity Experiments
# ==========================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "COMPLETING MISSING ELECTRICITY EXPERIMENTS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check which seed 43 and 44 experiments are missing
Write-Host "Scanning for missing experiments..." -ForegroundColor Yellow

$results_df = python -c "import pandas as pd; df=pd.read_csv('reports/tables/results.csv'); print(df.to_json())"

# Manually check each seed 43 and 44 config
$seeds_to_run = @(43, 44)
$configs = @()

foreach ($seed in $seeds_to_run) {
    foreach ($L in @(168, 336)) {
        foreach ($H in @(24, 168, 336)) {
            foreach ($pe in @("none", "absolute", "rope", "xpos", "alibi")) {
                
                # Check if this exact config exists in results.csv
                $exists = python -c "import pandas as pd; df=pd.read_csv('reports/tables/results.csv'); print('yes' if len(df[(df['dataset']=='electricity') & (df['pe']=='$pe') & (df['L']==$L) & (df['H']==$H) & (df['seed']==$seed)]) > 0 else 'no')"
                
                if ($exists -eq 'no') {
                    $configs += @{seed=$seed; L=$L; H=$H; pe=$pe}
                }
            }
        }
    }
}

$total_missing = $configs.Count

if ($total_missing -eq 0) {
    Write-Host "✓ All electricity experiments complete!" -ForegroundColor Green
    exit 0
}

Write-Host "Found $total_missing missing experiments" -ForegroundColor Yellow
Write-Host "This will take approximately $([math]::Round($total_missing * 4 / 60, 1)) hours`n" -ForegroundColor Cyan

$response = Read-Host "Continue? (Y/n)"
if ($response -eq 'n' -or $response -eq 'N') {
    exit 0
}

Write-Host "`nStarting experiments...`n" -ForegroundColor Green

$completed = 0
$start_time = Get-Date

foreach ($config in $configs) {
    $completed++
    $seed = $config.seed
    $L = $config.L
    $H = $config.H
    $pe = $config.pe
    
    $percent = [math]::Round(($completed / $total_missing) * 100, 1)
    Write-Host "[$completed/$total_missing ($percent%)] Seed=$seed, PE=$pe, L=$L, H=$H" -ForegroundColor White
    
    # Train
    python train.py `
        --dataset electricity `
        --pe $pe `
        --L $L `
        --H $H `
        --seed $seed `
        --epochs 3 `
        --batch_size 64 `
        --max_series 32 `
        --stride 120 `
        --normalize zscore `
        --device cuda `
        --lr 0.0003 2>&1 | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR in training!" -ForegroundColor Red
        continue
    }
    
    # Save checkpoint
    $ckpt_from = "checkpoints/ts_transformer_electricity_${pe}.pt"
    $ckpt_to = "checkpoints/ts_transformer_electricity_${pe}_L${L}_H${H}_seed${seed}.pt"
    if (Test-Path $ckpt_from) {
        Copy-Item $ckpt_from $ckpt_to -Force
    }
    
    # Eval
    python eval_extrap.py `
        --dataset electricity `
        --pe $pe `
        --L_train $L `
        --H $H `
        --eval_multipliers 1,2 `
        --max_series 32 `
        --normalize zscore `
        --device cuda `
        --ckpt $ckpt_to 2>&1 | Out-Null
    
    # Progress estimate
    $elapsed = ((Get-Date) - $start_time).TotalMinutes
    $rate = $elapsed / $completed
    $remaining_time = $rate * ($total_missing - $completed)
    Write-Host "  Est. remaining: $([math]::Round($remaining_time, 0)) min`n" -ForegroundColor DarkGray
}

$total_time = ((Get-Date) - $start_time).TotalHours
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "ELECTRICITY EXPERIMENTS COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Completed $completed experiments in $([math]::Round($total_time, 1)) hours" -ForegroundColor White
Write-Host "`nNow run:" -ForegroundColor Yellow
Write-Host "  python analyze_multiseed.py" -ForegroundColor Cyan
Write-Host "  python create_figures.py" -ForegroundColor Cyan
