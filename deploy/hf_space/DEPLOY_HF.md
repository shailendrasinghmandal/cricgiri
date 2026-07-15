# Deploy the CricGiri Delivery API to a FREE, always-on URL (Hugging Face Spaces)

This gives your team a **permanent public URL** they can use anytime — even when
your laptop is off — with **no paid plan** and **no credit card**.

Tradeoffs (be aware): the free Space runs on **CPU**, so a clip takes ~1–3 min
(not seconds like your GPU). After ~48 h of no use the Space "sleeps" and takes
~30 s to wake on the next request. For viewing model output / JSON, that's fine.

---

## One-time setup (~15 min)

### 1. Create the accounts / tools
- Sign up (free) at <https://huggingface.co/join>
- Install **Git LFS** once on your PC: <https://git-lfs.com>  (needed for the 85 MB model)

### 2. Create an empty Space
- Go to <https://huggingface.co/new-space>
- **Space name**: e.g. `cricgiri-api`
- **License**: any (e.g. MIT)
- **SDK**: choose **Docker** → **Blank**
- **Visibility**: Public (so teammates can reach it without logging in)
- Click **Create Space**

### 3. Build the Space folder (run in this project)
```powershell
powershell -File deploy\hf_space\build_space.ps1
```
This assembles `D:\cricket_final\cricgiri_hf_space` (~93 MB: source + the 2 models).

### 4. Push it to your Space
Replace `<user>` and `<space-name>` with yours:
```powershell
cd D:\cricket_final\cricgiri_hf_space
git init
git lfs install
git remote add origin https://huggingface.co/spaces/<user>/<space-name>
git add .
git commit -m "CricGiri delivery API"
git push -u origin main
```
> If HF asks for a password, use an **access token** (create one at
> <https://huggingface.co/settings/tokens>, role *write*), not your account password.

### 5. Wait for the build
Open your Space page — you'll see a **Building** log. First build is ~5–10 min
(installs torch etc.). When it turns **Running**, it's live.

---

## Share this with your team

- **Interactive page** (upload a video, see JSON):
  `https://<user>-<space-name>.hf.space/docs`
- **Health check**:
  `https://<user>-<space-name>.hf.space/health`
- **Direct call** (what a developer integrates):
  ```bash
  curl -X POST "https://<user>-<space-name>.hf.space/analyze" \
    -F "video=@clip.mp4" \
    -F "pitch_length=20.12"
  ```
  Returns the full delivery JSON (`source_video`, `fps`, `deliveries[]` with
  `track`, `bounce`, `world_trajectory`, `line`, `length`, `speed`, `swing`,
  `confidence_score`).

---

## Updating later
When you change code or the model, just re-run the build + push:
```powershell
powershell -File deploy\hf_space\build_space.ps1
cd D:\cricket_final\cricgiri_hf_space
git add .
git commit -m "update"
git push
```
The Space rebuilds automatically.

## Tuning speed vs accuracy (optional)
In the Space: **Settings → Variables and secrets** → add
`CRICGIRI_IMGSZ = 960` (faster, slightly less accurate) or keep the default
`1280` (tested accuracy). No code change or rebuild of your PC needed.
