"""
ToroTech YOLO Inference Service — v2
Run on 4090 GPU machine, serve HTTP API for equipment PCs.

Changes from v1:
  - detect params: Query → Form (multipart compatible)
  - response: 'class' → 'class_name' + added class_id, model_version, image_size, width, height
  - async inference with GPU semaphore (non-blocking /health while GPU busy)
  - upload validation (10MB limit, image type check)
  - path traversal protection on project_id
  - ONNX export: opset 12 → 17

Usage:
    python server.py
    → http://0.0.0.0:8100

API:
    POST /detect     — detect objects in image (multipart/form-data)
    GET  /models     — list available project models
    GET  /health     — GPU + model status
"""

import asyncio
import io
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI(title="ToroTech YOLO Inference Service", version="2.0.0")

# --- Global State ---
_models: OrderedDict[str, YOLO] = OrderedDict()
_model_versions: dict[str, str] = {}  # project → active version
_config: dict = {}
_start_time: float = 0

# GPU concurrency: one inference at a time, runs in thread pool
_infer_semaphore = asyncio.Semaphore(1)
_infer_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="infer")

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _root() -> Path:
    return Path(__file__).parent.parent


def load_config():
    global _config, _start_time
    cfg_path = Path(__file__).parent / "config" / "server_config.yaml"
    if cfg_path.exists():
        _config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        _config = {"port": 8100, "default_project": "robot_test_c4"}
    _start_time = time.time()


def _validate_project_id(project_id: str) -> str:
    if not _PROJECT_ID_RE.match(project_id):
        raise ValueError(f"Invalid project_id: '{project_id}'")
    return project_id


def get_model(project: str) -> tuple[YOLO, str]:
    """Load model with LRU cache. Returns (model, version_string)."""
    _validate_project_id(project)

    if project in _models:
        _models.move_to_end(project)
        return _models[project], _model_versions.get(project, "v1")

    models_dir = _root() / "models" / "projects" / project
    if not models_dir.exists():
        raise FileNotFoundError(f"No model found for project: {project}")

    # Find best model, preferring versioned directories
    version = "v1"
    pt_path = None

    # Try versioned structure first: models/projects/{id}/v1/best.pt
    for v_dir in sorted(models_dir.iterdir(), reverse=True):
        if v_dir.is_dir() and v_dir.name.startswith("v"):
            candidate = v_dir / "best.pt"
            if candidate.exists():
                pt_path = candidate
                version = v_dir.name
                break

    # Fallback: legacy structure models/projects/{id}/train/weights/best.pt
    if pt_path is None:
        legacy = sorted(models_dir.rglob("best.pt"))
        if legacy:
            pt_path = legacy[-1]
            version = "v1"

    # Fallback: ONNX
    if pt_path is None:
        onnx_paths = sorted(models_dir.rglob("best.onnx"))
        if onnx_paths:
            pt_path = onnx_paths[-1]
            version = "v1"

    if pt_path is None:
        raise FileNotFoundError(f"No model found for project: {project}")

    # Evict LRU if > 3 models loaded
    max_models = _config.get("max_loaded_models", 3)
    while len(_models) >= max_models:
        evicted_key, _ = _models.popitem(last=False)
        _model_versions.pop(evicted_key, None)

    _models[project] = YOLO(str(pt_path))
    _model_versions[project] = version
    return _models[project], version


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    project: str = Form(default=None),
    confidence: float = Form(default=0.5, ge=0.0, le=1.0),
    version: str = Form(default=None),
):
    """Detect objects in an uploaded image.

    BREAKING from v1:
      - params are Form fields (not Query params)
      - response uses class_name (not class), adds class_id, model_version, image_size
    """
    t0 = time.perf_counter()
    proj = project or _config.get("default_project", "robot_test_c4")

    # Validate content type
    if file.content_type and file.content_type not in _ALLOWED_TYPES:
        return JSONResponse(
            {"success": False, "error": f"Unsupported image type: {file.content_type}"},
            status_code=400,
        )

    # Read with size limit
    img_bytes = await file.read()
    if len(img_bytes) > _MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"success": False, "error": f"Image too large: {len(img_bytes)} bytes (max {_MAX_UPLOAD_BYTES})"},
            status_code=413,
        )

    # Validate image
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid image file"}, status_code=400)

    img_w, img_h = img.size

    try:
        model, model_ver = get_model(proj)
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=404)

    # Async inference: queue on GPU semaphore, run in thread pool
    device = _config.get("device", 0)
    async with _infer_semaphore:
        loop = asyncio.get_event_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    _infer_pool,
                    lambda: model.predict(img, conf=confidence, device=device, verbose=False),
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                {"success": False, "error": "Inference timed out (30s)"},
                status_code=504,
            )

    objects = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append({
                "class_id": int(box.cls[0]),
                "class_name": r.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                },
                "center_x": round((x1 + x2) / 2, 1),
                "center_y": round((y1 + y2) / 2, 1),
                "width": round(x2 - x1, 1),
                "height": round(y2 - y1, 1),
            })

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "success": True,
        "project": proj,
        "model_version": model_ver,
        "count": len(objects),
        "objects": objects,
        "image_size": {"width": img_w, "height": img_h},
        "inference_ms": round(elapsed, 1),
    }


@app.get("/models")
async def list_models():
    """List all available project models."""
    models_dir = _root() / "models" / "projects"
    projects = []
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir():
                versions = []
                for v_dir in sorted(p.iterdir()):
                    if v_dir.is_dir() and v_dir.name.startswith("v"):
                        versions.append({
                            "version": v_dir.name,
                            "has_pt": (v_dir / "best.pt").exists(),
                            "has_onnx": (v_dir / "best.onnx").exists(),
                        })
                # Legacy: check train/weights/ if no versioned dirs
                if not versions:
                    versions.append({
                        "version": "v1",
                        "has_pt": bool(list(p.rglob("best.pt"))),
                        "has_onnx": bool(list(p.rglob("best.onnx"))),
                    })
                projects.append({
                    "project_id": p.name,
                    "active_version": _model_versions.get(p.name),
                    "versions": versions,
                })
    return {"projects": projects}


@app.get("/health")
async def health():
    """Health check with GPU info."""
    gpu = torch.cuda.is_available()
    return {
        "status": "ok",
        "gpu": gpu,
        "gpu_name": torch.cuda.get_device_name(0) if gpu else None,
        "gpu_memory_used_mb": round(torch.cuda.memory_allocated() / 1024 / 1024) if gpu else None,
        "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024) if gpu else None,
        "loaded_models": list(_models.keys()),
        "training_active": False,
        "uptime_s": round(time.time() - _start_time),
    }


if __name__ == "__main__":
    import uvicorn

    load_config()
    port = _config.get("port", 8100)
    print(f"Starting YOLO service v2 on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
