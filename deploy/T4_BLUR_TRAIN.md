# Blur finetune on a Tesla T4 (imgsz 1280)

Runs the same blur-augmentation finetune that won locally at 640, but at full **1280**
(the T4's 16GB VRAM + more system RAM removes the OOM that forced 640). Expected to
lift recall further and cut false positives.

## 1. Upload to `/home/ubuntu/` (WinSCP or scp)
| Local file | Upload as |
|---|---|
| `models/ball_ft_t4.pt` | `/home/ubuntu/ball_ft_t4.pt` |
| `scripts/build_blur_hard_dataset.py` | `/home/ubuntu/build_blur_hard_dataset.py` |
| `dataset/ball_clean/` (the whole folder: `images/` + `labels/`) | `/home/ubuntu/ball_clean/` |
| `deploy/t4_blur_finetune.sh` | `/home/ubuntu/t4_blur_finetune.sh` |

> Only the **source** set (`ball_clean`, ~12k images) is uploaded — the 36k blurred
> copies are generated **on the T4**, so the upload stays small.

```bash
# scp example (from the project root, adjust host/key)
scp models/ball_ft_t4.pt scripts/build_blur_hard_dataset.py deploy/t4_blur_finetune.sh \
    ubuntu@<T4_HOST>:/home/ubuntu/
scp -r dataset/ball_clean ubuntu@<T4_HOST>:/home/ubuntu/ball_clean
```

## 2. Run it on the T4
```bash
ssh ubuntu@<T4_HOST>
cd /home/ubuntu
chmod +x t4_blur_finetune.sh
./t4_blur_finetune.sh            # build dataset -> finetune @1280 -> val
```
It builds the blur dataset (~10–15 min), then finetunes 40 epochs. On a T4 at 1280,
batch 12, expect roughly **1.5–3 h** (much faster + higher-res than the local run).
If you hit a CUDA OOM, edit the script `batch=12` → `batch=8`.

## 3. Bring the model back and measure
Download `/home/ubuntu/runs/ball_blur_ft_1280/weights/best.pt` and save it locally as
`models/staging/ball_blur_ft_1280.pt`, then:
```bash
venv/Scripts/python.exe scripts/eval_harness.py --video eval/clip01.mp4 \
    --tag phase4a_1280 --ball-model models/staging/ball_blur_ft_1280.pt
venv/Scripts/python.exe scripts/eval_harness.py --compare baseline phase4a_1280
```
Compare against the local 640 result (recall 0.56→0.68). If the 1280 model is better
and holds up across clip02–06, we switch production to it.

## Notes
- The finetune profile (low LR 0.0006, cosine, no mosaic, blur-augmented data) mirrors
  the local `train.py --finetune` path — same recipe, just at 1280 on better hardware.
- Production `ball_ft_t4.pt` is never overwritten; the new weights stay in `models/staging/`.
- Baseline to beat (local 640): detector_recall 0.68, traj_rmse 11.2px on clip01.
