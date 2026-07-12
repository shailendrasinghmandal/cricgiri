#!/usr/bin/env bash
# ============================================================================
#  T4 blur-augmentation finetune (imgsz 1280) — run on the Tesla T4 instance.
#  Builds the blur dataset ON the box (so you upload only the small source set,
#  not 36k augmented images), then fine-tunes ball_ft_t4.pt at full 1280.
#
#  UPLOAD to /home/ubuntu/ first (WinSCP/scp):
#     ball_ft_t4.pt                       (models/ball_ft_t4.pt)
#     build_blur_hard_dataset.py          (scripts/build_blur_hard_dataset.py)
#     ball_clean/                         (dataset/ball_clean/  — images/ + labels/)
#     t4_blur_finetune.sh                 (this file)
#  Then:  chmod +x t4_blur_finetune.sh && ./t4_blur_finetune.sh
# ============================================================================
set -e
cd /home/ubuntu

# Where ball_clean already lives (reuse the set from the ball_ft_t4 training if present,
# so you don't re-upload). Override:  BALL_CLEAN=/path/to/ball_clean ./t4_blur_finetune.sh
BALL_CLEAN="${BALL_CLEAN:-}"
if [ -z "$BALL_CLEAN" ]; then
  for c in ball_clean ~/ball_clean_extracted/ball_clean /home/ec2-user/ball_clean_extracted/ball_clean; do
    [ -d "$c/images" ] && BALL_CLEAN="$c" && break
  done
fi
if [ -z "$BALL_CLEAN" ] || [ ! -d "$BALL_CLEAN/images" ]; then
  echo "ERROR: ball_clean not found. Upload it, or set BALL_CLEAN=/path/to/ball_clean"; exit 1
fi
echo "using source dataset: $BALL_CLEAN"

echo "== 1/4  environment =="
python3 -m pip install --upgrade pip -q
python3 -m pip install -q ultralytics opencv-python-headless numpy

echo "== 2/4  build blur-augmented dataset from ball_clean =="
if [ ! -f ball_blur_aug/data.yaml ]; then
  python3 build_blur_hard_dataset.py \
    --src "$BALL_CLEAN" --out ball_blur_aug \
    --variants 2 --motion-min 9 --motion-max 31 --keep-original
fi
echo "dataset:"; head -6 ball_blur_aug/data.yaml

echo "== 3/4  finetune ball_ft_t4.pt @ imgsz 1280 (blur profile, low LR) =="
# T4 = 16GB VRAM; batch 12 fits yolov8s @1280. Drop to 8 if you hit CUDA OOM.
yolo detect train \
  model=ball_ft_t4.pt \
  data=ball_blur_aug/data.yaml \
  epochs=40 imgsz=1280 batch=12 device=0 workers=8 \
  optimizer=AdamW lr0=0.0006 lrf=0.01 cos_lr=True patience=25 \
  mosaic=0.0 close_mosaic=0 mixup=0.0 copy_paste=0.0 \
  degrees=2.0 fliplr=0.5 scale=0.2 hsv_v=0.45 hsv_s=0.5 \
  single_cls=True max_det=3 cache=disk \
  project=/home/ubuntu/runs name=ball_blur_ft_1280

echo "== 4/4  quick val =="
yolo detect val \
  model=/home/ubuntu/runs/ball_blur_ft_1280/weights/best.pt \
  data=ball_blur_aug/data.yaml imgsz=1280 conf=0.10 device=0 || true

echo
echo "DONE. Download this back to the project as models/staging/ball_blur_ft_1280.pt:"
echo "   /home/ubuntu/runs/ball_blur_ft_1280/weights/best.pt"
echo "Then locally:  venv/Scripts/python.exe scripts/eval_harness.py \\"
echo "   --video eval/clip01.mp4 --tag phase4a_1280 \\"
echo "   --ball-model models/staging/ball_blur_ft_1280.pt"
echo "   venv/Scripts/python.exe scripts/eval_harness.py --compare baseline phase4a_1280"
