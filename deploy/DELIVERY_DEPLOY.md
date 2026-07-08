# Deploying the Delivery Analytics API (always-on cloud URL)

Goal: a permanent HTTPS URL your team can POST videos to:
`POST https://<your-app>/analyze` (form: `video`, `pitch_length`).

The app is `api/delivery_api.py`; the container is `deploy/Dockerfile.delivery`.

---

## Two things only you can decide (this is why it isn't one-click)

### 1. Where the model weights live (they are NOT in the repo)
The pipeline needs three weights totalling **~241 MB**:
- `ball_ft_t4.pt` (89 MB), `ball_best_leather_new.pt` (155 MB), `stump_best.pt` (6 MB)

`ball_best_leather_new.pt` (155 MB) is **over GitHub's 100 MB file limit**, so it cannot be
committed to git normally. Pick one:
- **GitHub Release assets (free, simplest):** create a Release on your repo, upload the 3
  `.pt` files (Releases allow up to 2 GB per file), copy the 3 asset URLs. ⚠️ A *public*
  repo makes the weights publicly downloadable — use a **private** repo if the weights are
  proprietary.
- **Private signed URL (S3 / GCS / Azure Blob):** upload the 3 files, generate signed URLs.
- **Git LFS:** `git lfs track "*.pt"` and commit (uses your LFS quota/bandwidth).

Put the 3 URLs into `deploy/render.delivery.yaml` → `dockerBuildArgs` (or `--build-arg`).

### 2. Speed vs cost (there is no free GPU)
| Host | Speed/request | Cost | Notes |
|------|---------------|------|-------|
| Render **standard** (CPU, 2 GB) | ~2–5 min | ~$25/mo | Works. Free/512 MB **OOMs** — don't use it. |
| GPU host (RunPod / Modal / Replicate / Lambda / a GPU VM) | ~20–30 s | pay-per-use or hourly | Fast. Use a CUDA torch wheel in the Dockerfile. |

Low volume + fine with slow → CPU (Render). Need fast/interactive → GPU host.

---

## Deploy on Render (CPU path)
1. Push this branch to GitHub.
2. Host the 3 weights (see #1); paste their URLs into `deploy/render.delivery.yaml`
   `dockerBuildArgs`.
3. Render dashboard → **New → Blueprint** → select the repo → it reads
   `deploy/render.delivery.yaml`. Deploy.
4. When live, health-check: `GET https://<app>.onrender.com/health`.
5. Share this with your team:
   ```bash
   curl -X POST https://<app>.onrender.com/analyze \
     -F "video=@delivery.mp4" \
     -F "pitch_length=20.12"
   ```
6. (Recommended) uncomment `API_KEY` in the blueprint so the public URL requires an
   `X-API-Key` header, and give your team the key.

## Deploy on a GPU host (fast path)
Same `deploy/Dockerfile.delivery`, but swap the torch install line for a CUDA wheel
(e.g. `--index-url https://download.pytorch.org/whl/cu121`) and run on a GPU instance.
RunPod/Modal/Replicate can serve the container directly; any GPU VM can `docker run` it.

---

## What's already done
- API verified end-to-end (upload → schema JSON, 137.2 km/h; NO_TRACK → 200 with reason;
  concurrent requests serialised safely).
- `pitch_length` form field is the speed scale knob — pass the real value per video.
- Container + blueprint prepared. Remaining work is the two decisions above (yours).
