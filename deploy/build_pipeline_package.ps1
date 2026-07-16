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

# --- HTTP API (+ backend webhook) -------------------------------------------
# Ships as api/delivery_api.py — the SAME module path the backend/Dockerfile
# already use ("uvicorn api.delivery_api:app --port 7860"), so this package is a
# drop-in replacement for the previously delivered zip. In the repo the file is
# api/pipeline_api.py so it does not collide with the older script-engine API of
# that name; only the pipeline-driven one is shipped.
$ApiDst = Join-Path $Stage "api"
New-Item -ItemType Directory -Path $ApiDst -Force | Out-Null
$apiSrc = Join-Path $Root "api\pipeline_api.py"
if (-not (Test-Path $apiSrc)) { throw "MISSING REQUIRED API FILE: $apiSrc" }
Write-Host "  copy  api\pipeline_api.py -> api\delivery_api.py  (drop-in module path)"
Copy-Item $apiSrc (Join-Path $ApiDst "delivery_api.py") -Force
$initSrc = Join-Path $Root "api\__init__.py"
if (-not (Test-Path $initSrc)) { throw "MISSING REQUIRED API FILE: $initSrc" }
Copy-Item $initSrc (Join-Path $ApiDst "__init__.py") -Force
Write-Host "  copy  api\__init__.py"

# --- Dockerfile : same CMD/port as the deployed image ------------------------
@"
# CricGiri pipeline analytics API — production container.
# Same entrypoint/port as the previously deployed image, so this is a drop-in.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:`$PATH
WORKDIR /app

# CPU-only torch by default; for a GPU host swap the index-url for cu126.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads outputs videos logs && chown -R user:user /app
USER user

ENV PYTHONUNBUFFERED=1 \
    CRICGIRI_WEBHOOK_URL=https://aistg.cricgiri.com/webhook/session-clip \
    MAX_UPLOAD_MB=200

EXPOSE 7860

# Single worker: the YOLO models are shared singletons, inference is serialised,
# and the in-memory /status job store is process-local (relies on --workers 1).
CMD ["uvicorn", "api.delivery_api:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--timeout-keep-alive", "300"]
"@ | Set-Content -Path (Join-Path $Stage "Dockerfile") -Encoding ascii
Write-Host "  write Dockerfile  (uvicorn api.delivery_api:app --port 7860)"

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

# HTTP API + backend webhook (api/pipeline_api.py)
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9
requests>=2.31.0
websockets>=12.0

# OPTIONAL — only needed if you enable PipelineConfig(use_enhanced_detection=True).
# analytics/ball_enhancement.py imports skimage lazily (inside the function), so
# the default pipeline runs fine without it. Uncomment if you turn that path on.
# scikit-image
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
