# Run Weather and Traffic (Complete)
# ====================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "RUNNING WEATHER + TRAFFIC" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$start_time = Get-Date

# ============================================================================
# WEATHER (with xPos)
# ============================================================================
Write-Host "PHASE 1: Weather Dataset" -ForegroundColor Cyan
Write-Host "  5 PEs × 4 configs = 20 experiments`n" -ForegroundColor White

$weather_count = 0
$weather_total = 20

foreach ($L in @(168, 336)) {
    foreach ($H in @(24, 168)) {
        foreach ($pe in @("none", "absolute", "rope", "xpos", "alibi")) {
            $weather_count++
            $percent = [math]::Round(($weather_count / $weather_total) * 100, 1)
            
            Write-Host "[$weather_count/$weather_total ($percent%)] PE=$pe, L=$L, H=$H" -ForegroundColor White
            
            # Train
            python train.py `
                --dataset weather `
                --pe $pe `
                --L $L `
                --H $H `
                --seed 42 `
                --epochs 3 `
                --batch_size 64 `
                --max_series 32 `
                --stride 120 `
                --normalize zscore `
                --device cuda `
                --lr 0.0003
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR in training!" -ForegroundColor Red
                continue
            }
            
            # Eval
            python eval_extrap.py `
                --dataset weather `
                --pe $pe `
                --L_train $L `
                --H $H `
                --eval_multipliers 1,2 `
                --max_series 32 `
                --normalize zscore `
                --device cuda
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR in evaluation!" -ForegroundColor Red
            }
        }
    }
}

$phase1_time = ((Get-Date) - $start_time).TotalMinutes
Write-Host "`nWeather complete: $([math]::Round($phase1_time, 1)) minutes`n" -ForegroundColor Green

# ============================================================================
# TRAFFIC (with xPos)
# ============================================================================
Write-Host "PHASE 2: Traffic Dataset" -ForegroundColor Cyan
Write-Host "  5 PEs × 6 configs = 30 experiments`n" -ForegroundColor White

$traffic_count = 0
$traffic_total = 30

foreach ($L in @(168, 336)) {
    foreach ($H in @(24, 168, 336)) {
        foreach ($pe in @("none", "absolute", "rope", "xpos", "alibi")) {
            $traffic_count++
            $percent = [math]::Round(($traffic_count / $traffic_total) * 100, 1)
            
            Write-Host "[$traffic_count/$traffic_total ($percent%)] PE=$pe, L=$L, H=$H" -ForegroundColor White
            
            # Train
            python train.py `
                --dataset traffic `
                --pe $pe `
                --L $L `
                --H $H `
                --seed 42 `
                --epochs 3 `
                --batch_size 64 `
                --max_series 32 `
                --stride 120 `
                --normalize zscore `
                --device cuda `
                --lr 0.0003
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR in training!" -ForegroundColor Red
                continue
            }
            
            # Eval
            python eval_extrap.py `
                --dataset traffic `
                --pe $pe `
                --L_train $L `
                --H $H `
                --eval_multipliers 1,2 `
                --max_series 32 `
                --normalize zscore `
                --device cuda
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR in evaluation!" -ForegroundColor Red
            }
        }
    }
}

$phase2_time = ((Get-Date) - $start_time).TotalMinutes - $phase1_time
Write-Host "`nTraffic complete: $([math]::Round($phase2_time, 1)) minutes`n" -ForegroundColor Green

# ============================================================================
# SUMMARY
# ============================================================================
$total_time = ((Get-Date) - $start_time).TotalHours

Write-Host "========================================" -ForegroundColor Green
Write-Host "WEATHER + TRAFFIC COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Total time: $([math]::Round($total_time, 2)) hours" -ForegroundColor White
Write-Host "  Weather: $weather_count experiments" -ForegroundColor White
Write-Host "  Traffic: $traffic_count experiments" -ForegroundColor White
Write-Host "`nNext step:" -ForegroundColor Yellow
Write-Host "  Run complete_electricity.ps1 to finish remaining electricity seeds" -ForegroundColor Cyan
