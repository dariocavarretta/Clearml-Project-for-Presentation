# YOLO Inference App

A self-contained web app that runs YOLO ONNX inference and renders bounding
boxes + class/confidence results in a mobile-friendly browser UI.

```
inference-app/
├── app.py               ← FastAPI backend (ONNX runtime, NMS, letterbox)
├── classes.json         ← class name list  ["target_object", ...]
├── requirements.txt
├── Dockerfile
├── docker-compose.yml   ← local / self-hosted
├── fly.toml             ← Fly.io cloud deploy
├── README.md
├── models/              ← DROP YOUR .onnx (or .engine) FILES HERE
└── static/
    └── index.html       ← full frontend (drag-drop, camera, bbox overlay)
```

---

## Option A — Fly.io (recommended — free, always-on, public HTTPS URL)

Fly.io gives you one always-on machine + a persistent volume for free.
Your app gets a public `https://yolo-inference.fly.dev` URL that anyone can open.

### 1. Install flyctl

```bash
# macOS / Linux
curl -L https://fly.io/install.sh | sh

# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex
```

### 2. Sign up / log in

```bash
fly auth signup     # first time
# or
fly auth login
```

### 3. Edit fly.toml — pick a unique app name

Open `fly.toml` and change:
```toml
app = "yolo-inference-yourname"   # must be globally unique on fly.io
```
Pick a region close to you (`ams`=Amsterdam, `lhr`=London, `lax`=LA, `ord`=Chicago).

### 4. Launch (first deploy only)

```bash
cd Project-ClearML/production-pipeline/inference-app

fly launch --no-deploy          # reads fly.toml, registers the app
fly volumes create models_vol --region ams --size 1   # 1 GB for models
fly deploy                      # builds image on Fly, deploys
```

### 5. Upload your model

```bash
# copy best.onnx into the persistent volume via sftp
fly sftp shell
> put /local/path/to/best.onnx /app/models/best.onnx
> exit
```

Or use `fly ssh console` to pull from a URL:

```bash
fly ssh console
# inside the container:
wget -O /app/models/best.onnx "https://your-storage/best.onnx"
exit
```

### 6. Open in browser

```bash
fly open          # opens https://yolo-inference-yourname.fly.dev
```

Send this URL to anyone — it works on desktop and mobile.

### Re-deploy after code changes

```bash
fly deploy        # ~60 s, zero-downtime rolling deploy
```

---

## Option B — Local machine, public via Cloudflare Tunnel (no port-forwarding needed)

Good if you want to run on your own hardware without any cloud account.

### 1. Start the app

```bash
cd Project-ClearML/production-pipeline/inference-app
cp /path/to/best.onnx ./models/
docker compose up -d
```

### 2. Install cloudflared and create a tunnel

```bash
# macOS
brew install cloudflare/cloudflare/cloudflared

# Linux
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Windows — download from https://github.com/cloudflare/cloudflared/releases
```

### 3. Start a quick public tunnel (no account needed)

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a random URL like `https://abc123.trycloudflare.com` — share that.

> For a permanent URL, create a free Cloudflare account and a named tunnel.
> See: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/

---

## Option C — Self-hosted VPS (DigitalOcean / Hetzner / etc.)

If you already have a Linux VPS:

```bash
# on the server
git clone <your-repo> && cd inference-app
cp best.onnx ./models/
docker compose up -d

# add nginx reverse-proxy + Let's Encrypt for HTTPS (optional but recommended)
# https://nginxproxymanager.com/ is the easiest way
```

---

## Using on your phone (any option)

1. Open the public URL in your phone's browser.
2. Tap **📷 Take Photo** — this opens the rear camera directly.
3. Take the photo, confirm, then tap **Run Detection**.
4. Bounding boxes are drawn on the image with class name + confidence %.

---

## Multi-class models

Edit `classes.json` to match your model's class list (order = class index):

```json
["bottle", "can", "cup", "plastic_bag"]
```

---

## GPU acceleration

Replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt`:

```
onnxruntime-gpu==1.18.0
```

The app auto-detects CUDA and uses `CUDAExecutionProvider` when available.

---

## API reference

### `GET /healthz`
```json
{ "status": "ok" }
```

### `GET /models`
```json
{ "models": ["best.onnx"] }
```

### `POST /predict`
| param  | type  | default | description |
|--------|-------|---------|-------------|
| `file` | form  | —       | Image (multipart) |
| `model`| query | first   | Filename in `models/` |
| `conf` | query | 0.25    | Confidence threshold |
| `iou`  | query | 0.45    | IoU threshold for NMS |

**Response**
```json
{
  "predictions": [
    {
      "class_id": 0,
      "class_name": "target_object",
      "confidence": 0.912,
      "box": { "x1": 120, "y1": 45, "x2": 380, "y2": 290 }
    }
  ],
  "image_width": 1280,
  "image_height": 720,
  "model": "best.onnx",
  "inference_ms": 34.2
}
```
