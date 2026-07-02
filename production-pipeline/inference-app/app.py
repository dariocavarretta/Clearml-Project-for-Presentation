"""
YOLO ONNX Inference API
-----------------------
Environment variables:
  APP_PASSWORD  — plain-text password for the login page (default: "yolo")
  MAX_HISTORY   — how many annotated images to keep (default: 10)
"""

import os
import time
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Response, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inference")

# ── config ────────────────────────────────────────────────────────────────────
APP_PASSWORD = os.environ.get("APP_PASSWORD", "yolo")
MAX_HISTORY  = int(os.environ.get("MAX_HISTORY", 10))
COOKIE_NAME  = "yolo_session"
# simple in-memory set of valid session tokens (fine for single-machine Fly deploy)
_valid_tokens: set[str] = set()

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODELS_DIR   = BASE_DIR / "models"
STATIC_DIR   = BASE_DIR / "static"
HISTORY_DIR  = BASE_DIR / "models" / "history"   # persistent Fly volume
STATS_FILE   = BASE_DIR / "models" / "stats.json" # daily stats, also on volume
CLASSES_FILE = BASE_DIR / "classes.json"
for d in (MODELS_DIR, STATIC_DIR, HISTORY_DIR):
    d.mkdir(exist_ok=True)

# ── class names ───────────────────────────────────────────────────────────────
def load_class_names() -> list[str]:
    if CLASSES_FILE.exists():
        with open(CLASSES_FILE) as f:
            return json.load(f)
    return ["target_object"]

# ── model registry ────────────────────────────────────────────────────────────
_sessions: dict[str, ort.InferenceSession] = {}

def _get_model_paths() -> list[Path]:
    return sorted(
        [p for ext in ("*.onnx", "*.engine") for p in MODELS_DIR.glob(ext)]
    )

def _load_session(model_path: Path) -> ort.InferenceSession:
    key = str(model_path)
    if key not in _sessions:
        log.info(f"Loading model: {model_path.name}")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(key, providers=providers)
        _sessions[key] = sess
        log.info(f"  input: {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}")
    return _sessions[key]

# ── pre/post processing ────────────────────────────────────────────────────────
def letterbox(img: np.ndarray, new_shape=(640, 640)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - nw) / 2, (new_shape[0] - nh) / 2
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)

def preprocess(img_bgr: np.ndarray, imgsz: int = 640):
    img_lb, ratio, (dw, dh) = letterbox(img_bgr, (imgsz, imgsz))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img_rgb.transpose(2, 0, 1), 0), ratio, dw, dh

def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thr]
    return keep

def postprocess(raw, orig_h, orig_w, ratio, dw, dh, conf_thr, iou_thr, class_names):
    pred = raw[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    nc = len(class_names)
    boxes_xywh  = pred[:, :4]
    class_scores = pred[:, 4:4 + nc]
    class_ids    = class_scores.argmax(axis=1)
    confidences  = class_scores.max(axis=1)
    mask = confidences >= conf_thr
    boxes_xywh, confidences, class_ids = boxes_xywh[mask], confidences[mask], class_ids[mask]
    if len(confidences) == 0:
        return []
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    results = []
    for idx in nms(boxes_xyxy, confidences, iou_thr):
        x1, y1, x2, y2 = boxes_xyxy[idx]
        x1 = max(0, int(round((x1 - dw) / ratio))); y1 = max(0, int(round((y1 - dh) / ratio)))
        x2 = min(orig_w, int(round((x2 - dw) / ratio))); y2 = min(orig_h, int(round((y2 - dh) / ratio)))
        cid = int(class_ids[idx])
        results.append({
            "class_id":   cid,
            "class_name": class_names[cid] if cid < len(class_names) else str(cid),
            "confidence": float(round(float(confidences[idx]), 4)),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })
    return results

# ── drawing + history ─────────────────────────────────────────────────────────
PALETTE_BGR = [
    (247, 142, 79), (216, 92, 124), (153, 211, 52), (36, 191, 251), (113, 113, 248),
    (250, 160, 96), (250, 139, 124), (183, 231, 110), (77, 212, 252), (165, 139, 247),
]

def draw_predictions(img_bgr: np.ndarray, predictions: list[dict]) -> np.ndarray:
    out = img_bgr.copy()
    for p in predictions:
        color = PALETTE_BGR[p["class_id"] % len(PALETTE_BGR)]
        x1, y1, x2, y2 = p["box"]["x1"], p["box"]["y1"], p["box"]["x2"], p["box"]["y2"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{p['class_name']} {p['confidence']*100:.0f}%"
        fs, th = 1.1, 2
        (tw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ly = y1 - 10 if y1 > lh + 12 else y1 + lh + 12
        cv2.rectangle(out, (x1, ly - lh - 6), (x1 + tw + 10, ly + baseline + 2), color, -1)
        cv2.putText(out, label, (x1 + 5, ly), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out

def save_to_history(img_bgr: np.ndarray, predictions: list[dict], meta: dict) -> str:
    """Save annotated image + JSON sidecar. Returns the stem (timestamp string)."""
    stem = str(int(time.time() * 1000))
    cv2.imwrite(str(HISTORY_DIR / f"{stem}.jpg"),
                draw_predictions(img_bgr, predictions),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    (HISTORY_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    # prune
    imgs = sorted(HISTORY_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    for old in imgs[:-MAX_HISTORY]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)
    return stem

# ── daily stats ───────────────────────────────────────────────────────────────
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def load_stats() -> dict:
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text())
        except Exception:
            pass
    return {}

def save_stats(stats: dict) -> None:
    STATS_FILE.write_text(json.dumps(stats, indent=2))

def record_prediction(n_detections: int) -> None:
    stats = load_stats()
    day = _today()
    if day not in stats:
        stats[day] = {"predictions": 0, "fp": 0, "fn": 0}
    stats[day]["predictions"] += 1
    save_stats(stats)

def record_feedback(fp: int, fn: int) -> None:
    stats = load_stats()
    day = _today()
    if day not in stats:
        stats[day] = {"predictions": 0, "fp": 0, "fn": 0}
    stats[day]["fp"] += fp
    stats[day]["fn"] += fn
    save_stats(stats)

# ── auth helpers ──────────────────────────────────────────────────────────────
def _make_token() -> str:
    return secrets.token_hex(32)

def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get(COOKIE_NAME, "")
    return token in _valid_tokens

def require_auth(request: Request):
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="YOLO Inference API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── public endpoints ──────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))

@app.post("/login")
async def login(request: Request, response: Response):
    body = await request.json()
    if body.get("password") != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password")
    token = _make_token()
    _valid_tokens.add(token)
    response.set_cookie(
        key=COOKIE_NAME, value=token,
        httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30  # 30 days
    )
    return {"ok": True}

@app.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME, "")
    _valid_tokens.discard(token)
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}

# ── authenticated endpoints ───────────────────────────────────────────────────
@app.get("/models")
async def list_models(request: Request):
    require_auth(request)
    return {"models": [p.name for p in _get_model_paths()]}

@app.get("/history")
async def list_history(request: Request):
    require_auth(request)
    imgs = sorted(HISTORY_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for img in imgs:
        mp = img.with_suffix(".json")
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        result.append({"image": img.name, "meta": meta})
    return {"entries": result}

@app.get("/history/{filename}")
async def get_history_image(filename: str, request: Request):
    require_auth(request)
    path = HISTORY_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(path))

@app.delete("/history")
async def clear_history(request: Request):
    require_auth(request)
    count = 0
    for f in HISTORY_DIR.iterdir():
        if f.suffix in (".jpg", ".json"):
            f.unlink(missing_ok=True)
            count += 1
    return {"deleted": count}

@app.post("/feedback")
async def feedback(request: Request):
    """
    Body: { "stem": "1234567890", "fp": 1, "fn": 0 }
    stem is the timestamp key returned in the predict response.
    """
    require_auth(request)
    body = await request.json()
    fp  = max(0, int(body.get("fp", 0)))
    fn  = max(0, int(body.get("fn", 0)))
    stem = body.get("stem", "")
    # update the sidecar JSON with feedback
    if stem:
        jp = HISTORY_DIR / f"{stem}.json"
        if jp.exists():
            meta = json.loads(jp.read_text())
            meta["feedback"] = {"fp": fp, "fn": fn}
            jp.write_text(json.dumps(meta, indent=2))
    record_feedback(fp, fn)
    return {"ok": True}

@app.get("/stats")
async def get_stats(request: Request):
    require_auth(request)
    stats = load_stats()
    # return last 30 days sorted ascending
    sorted_days = sorted(stats.items())[-30:]
    return {"days": [{"date": d, **v} for d, v in sorted_days]}

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Query(None),
    conf:  float = Query(0.5, ge=0.0, le=1.0),
    iou:   float = Query(0.45, ge=0.0, le=1.0),
):
    require_auth(request)

    paths = _get_model_paths()
    if not paths:
        raise HTTPException(status_code=503, detail="No model found in /models directory.")
    chosen = next((p for p in paths if p.name == model), None) if model else paths[0]
    if chosen is None:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")

    session  = _load_session(chosen)
    inp_meta = session.get_inputs()[0]
    imgsz = inp_meta.shape[-1] if isinstance(inp_meta.shape[-1], int) and inp_meta.shape[-1] > 0 else 640

    raw_bytes = await file.read()
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")
    orig_h, orig_w = img.shape[:2]

    blob, ratio, dw, dh = preprocess(img, imgsz)
    t0 = time.perf_counter()
    outputs = session.run(None, {inp_meta.name: blob})
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    class_names = load_class_names()
    preds = postprocess(outputs[0], orig_h, orig_w, ratio, dw, dh, conf, iou, class_names)

    stem = ""
    try:
        stem = save_to_history(img, preds, {
            "model": chosen.name, "conf": conf, "iou": iou,
            "inference_ms": elapsed_ms, "image_width": orig_w,
            "image_height": orig_h, "detections": len(preds),
            "predictions": preds,
        })
        record_prediction(len(preds))
    except Exception as e:
        log.warning(f"History/stats error: {e}")

    return JSONResponse({
        "predictions":  preds,
        "image_width":  orig_w,
        "image_height": orig_h,
        "model":        chosen.name,
        "inference_ms": elapsed_ms,
        "stem":         stem,   # ← used by frontend to submit feedback
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
