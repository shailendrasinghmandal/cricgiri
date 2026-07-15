# ============================================================================
#  build_space.ps1 — assemble a clean Hugging Face Space repo for the
#  CricGiri Delivery API (source code + the 2 production models only).
#
#  Run from anywhere:
#     powershell -ExecutionPolicy Bypass -File deploy\hf_space\build_space.ps1
#
#  Output: D:\cricket_final\cricgiri_hf_space  (ready to `git push` to HF)
# ============================================================================
$ErrorActionPreference = "Stop"

$Root  = (Resolve-Path "$PSScriptRoot\..\..").Path      # project root
$Build = Join-Path (Split-Path $Root -Parent) "cricgiri_hf_space"

Write-Host "Project root : $Root"
Write-Host "Build target : $Build`n"

# Fresh build dir (never touch an existing .git if the Space is already cloned there)
if (Test-Path $Build) {
    Get-ChildItem $Build -Force | Where-Object { $_.Name -ne ".git" } |
        Remove-Item -Recurse -Force
} else {
    New-Item -ItemType Directory -Path $Build | Out-Null
}

# Source packages the delivery API needs (source only — small).
$SrcDirs = @("api", "scripts", "analytics", "pipeline", "tracking", "config")
foreach ($d in $SrcDirs) {
    $src = Join-Path $Root $d
    if (Test-Path $src) {
        Write-Host "  copy  $d\"
        Copy-Item $src (Join-Path $Build $d) -Recurse -Force
    }
}

# Prune caches / stray weights that may sit inside source dirs.
Get-ChildItem $Build -Recurse -Directory -Filter "__pycache__" -Force |
    Remove-Item -Recurse -Force
Get-ChildItem $Build -Recurse -File -Include *.pt,*.pth,*.onnx -Force |
    Remove-Item -Force

# The 2 production models (this is the part that must reach the Space).
$ModelDst = Join-Path $Build "models"
New-Item -ItemType Directory -Path $ModelDst -Force | Out-Null
foreach ($m in @("ball_ft_t4.pt", "ball_best_leather_new.pt", "stump_best.pt")) {
    $mp = Join-Path $Root "models\$m"
    if (-not (Test-Path $mp)) { throw "MISSING model: $mp" }
    Write-Host "  copy  models\$m"
    Copy-Item $mp (Join-Path $ModelDst $m) -Force
}

# Root files HF expects: Dockerfile + README.md (with Space frontmatter) + requirements.
Copy-Item (Join-Path $PSScriptRoot "Dockerfile")        (Join-Path $Build "Dockerfile") -Force
Copy-Item (Join-Path $PSScriptRoot "README.md")         (Join-Path $Build "README.md") -Force
Copy-Item (Join-Path $PSScriptRoot "requirements.txt")  (Join-Path $Build "requirements.txt") -Force

# .gitattributes so HF stores the big model via git-LFS.
@"
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
"@ | Set-Content -Path (Join-Path $Build ".gitattributes") -Encoding ascii

# .gitignore for runtime scratch.
@"
__pycache__/
*.pyc
uploads/
outputs/
videos/*.mp4
logs/
"@ | Set-Content -Path (Join-Path $Build ".gitignore") -Encoding ascii

$sizeMB = [math]::Round((Get-ChildItem $Build -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "`nDONE. Space repo assembled at:`n  $Build  ($sizeMB MB)`n"
Write-Host "Next (one-time):"
Write-Host "  1. Create a Space at https://huggingface.co/new-space  (SDK = Docker, blank)"
Write-Host "  2. In an EMPTY terminal:"
Write-Host "       cd `"$Build`""
Write-Host "       git init && git lfs install"
Write-Host "       git remote add origin https://huggingface.co/spaces/<user>/<space-name>"
Write-Host "       git add . && git commit -m `"CricGiri delivery API`""
Write-Host "       git push -u origin main   (or: git push origin HEAD:main)"
Write-Host "  3. Watch the build on the Space page. When live, share:"
Write-Host "       https://<user>-<space-name>.hf.space/docs"
