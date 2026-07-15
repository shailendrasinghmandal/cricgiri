# ============================================================================
#  build_pipeline_package.ps1 — assemble a self-contained CricGiri PIPELINE
#  package (code + all 3 model weights) for deployment on another server.
#
#  Run:  powershell -File deploy\build_pipeline_package.ps1
#  Out:  D:\cricket_final\cricgiri_pipeline_package.zip
# ============================================================================
$ErrorActionPreference = "Stop"

$Root  = (Resolve-Path "$PSScriptRoot\..").Path
$Stage = Join-Path (Split-Path $Root -Parent) "cricgiri_pipeline_pkg"
$Zip   = Join-Path (Split-Path $Root -Parent) "cricgiri_pipeline_package.zip"

Write-Host "Project : $Root"
Write-Host "Staging : $Stage`n"

if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage | Out-Null

# --- Source packages the pipeline imports -----------------------------------
# pipeline -> analytics, tracking (and those two import only each other).
# config/  holds tracking_defaults.yaml.
foreach ($d in @("pipeline", "analytics", "tracking", "config")) {
    $src = Join-Path $Root $d
    if (-not (Test-Path $src)) { throw "MISSING source dir: $src" }
    Write-Host "  copy  $d\"
    Copy-Item $src (Join-Path $Stage $d) -Recurse -Force
}

# --- scripts/ : REQUIRED, not optional --------------------------------------
# pipeline.py loads scripts/physics_gate_v2.py BY PATH at runtime, and that file
# in turn loads scripts/delivery_reconstruction.py at import time. If either is
# absent the physics gate silently degrades to placeholder values, so both are
# mandatory. The whole (source-only) dir is copied so nothing is missed.
Write-Host "  copy  scripts\  (physics_gate_v2 + delivery_reconstruction are REQUIRED)"
Copy-Item (Join-Path $Root "scripts") (Join-Path $Stage "scripts") -Recurse -Force

# --- Root-level modules imported by the package -----------------------------
# pipeline.py does `from ball_label_utils import ...` — a top-level module that
# lives at the project root, not inside a package dir. Missing it is a hard
# ImportError on the target server, so this is verified, not assumed.
foreach ($f in @("ball_label_utils.py")) {
    $p = Join-Path $Root $f
    if (-not (Test-Path $p)) { throw "MISSING REQUIRED ROOT MODULE: $p" }
    Write-Host "  copy  $f"
    Copy-Item $p (Join-Path $Stage $f) -Force
}

# --- Prune caches / stray weights inside source dirs ------------------------
Get-ChildItem $Stage -Recurse -Directory -Filter "__pycache__" -Force |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $Stage -Recurse -File -Include *.pt,*.pth,*.onnx -Force |
    Remove-Item -Force -ErrorAction SilentlyContinue

# --- Model weights : ALL THREE are mandatory --------------------------------
$ModelDst = Join-Path $Stage "models"
New-Item -ItemType Directory -Path $ModelDst -Force | Out-Null
foreach ($m in @("ball_ft_t4.pt", "ball_best_leather_new.pt", "stump_best.pt")) {
    $mp = Join-Path $Root "models\$m"
    if (-not (Test-Path $mp)) { throw "MISSING REQUIRED MODEL: $mp" }
    $mb = [math]::Round((Get-Item $mp).Length / 1MB, 1)
    Write-Host "  copy  models\$m  ($mb MB)"
    Copy-Item $mp (Join-Path $ModelDst $m) -Force
}

# --- Runner + docs + requirements -------------------------------------------
Copy-Item (Join-Path $Root "run_pipeline.py") $Stage -Force
Write-Host "  copy  run_pipeline.py"
foreach ($f in @("deploy\PIPELINE_SETUP.md", "docs\DELIVERY_API_RESPONSE_FORMAT.md")) {
    $p = Join-Path $Root $f
    if (Test-Path $p) {
        Copy-Item $p (Join-Path $Stage (Split-Path $f -Leaf)) -Force
        Write-Host ("  copy  " + (Split-Path $f -Leaf))
    }
}

@"
# CricGiri pipeline dependencies.
# torch/torchvision are installed separately (see PIPELINE_SETUP.md) so the
# correct CPU or CUDA build is chosen for the target server.
ultralytics
opencv-python-headless
numpy
scipy
filterpy
PyYAML
"@ | Set-Content -Path (Join-Path $Stage "requirements.txt") -Encoding ascii
Write-Host "  write requirements.txt"

# Runtime scratch dirs so first run never fails on a missing folder.
foreach ($d in @("outputs", "videos", "logs")) {
    New-Item -ItemType Directory -Path (Join-Path $Stage $d) -Force | Out-Null
    Set-Content -Path (Join-Path $Stage "$d\.gitkeep") -Value "" -Encoding ascii
}

# --- Zip --------------------------------------------------------------------
if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path (Get-ChildItem $Stage -Force).FullName -DestinationPath $Zip -CompressionLevel Optimal
$size = [math]::Round((Get-Item $Zip).Length / 1MB, 1)

Write-Host "`nDONE"
Write-Host "  staged : $Stage"
Write-Host "  ZIP    : $Zip  ($size MB)"
