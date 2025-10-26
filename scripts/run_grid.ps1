param(
  [string[]]$Datasets = @("electricity","traffic","exchange_rate","weather"),
  [string[]]$PEs      = @("none","absolute","rope","xpos","alibi")
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

foreach($d in $Datasets){
  switch($d){
    "electricity"   { $L=256; $H=24;  $stride=30; $maxs=8 }
    "traffic"       { $L=256; $H=24;  $stride=30; $maxs=8 }
    "exchange_rate" { $L=365; $H=30;  $stride=7;  $maxs=8 }
    "weather"       { $L=144; $H=144; $stride=60; $maxs=8 }  # 10-min data: 144 = 1 day
  }
  foreach($pe in $PEs){
    Write-Host "== TRAIN $d pe=$pe ==" -ForegroundColor Cyan
    python train.py --dataset $d --pe $pe --epochs 1 --L $L --H $H --stride $stride --max_series $maxs
  }
}
