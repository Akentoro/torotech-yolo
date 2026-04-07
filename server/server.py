"""
ToroTech YOLO Inference Service
Run on 4090 GPU machine, serve HTTP API for equipment PCs.

Usage:
    python server.py
    → http://0.0.0.0:8100

API:
    POST /detect     — detect objects in image
    GET  /models     — list available project models
    GET  /health     — GPU + model status
"""

import io
import time
from pathlib import Path

import yaml
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI(title="ToroTech YOLO Inference Service")

_models: dict[str, YOLO] = {}
_config: dict = {}


def load_config():
    global _config
    cfg_path = Path(__file__).parent / "config" / "server_config.yaml"
    if cfg_path.exists():
        _config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        _config = {"port": 8100, "default_project": "robot_test_c4"}


def get_model(project: str) -> YOLO:
    if project not in _models:
        model_dir = Path(__file__).parent.parent / "models" / "projects" / project
        # Prefer .pt for GPU inference, fallback to .onnx
        pt_paths = sorted(model_dir.rglob("best.pt"))
        if pt_paths:
            _models[project] = YOLO(str(pt_paths[-1]))
        else:
            onnx_paths = sorted(model_dir.rglob("best.onnx"))
            if onnx_paths:
                _models[project] = YOLO(str(onnx_paths[-1]))
            else:
                raise FileNotFoundError(f"No model found for project: {project}")
    return _models[project]


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    project: str = Query(default=None),
    confidence: float = Query(default=0.5),
):
    """Detect objects in an uploaded image."""
    t0 = time.perf_counter()
    proj = project or _config.get("default_project", "robot_test_c4")

    try:
        model = get_model(proj)
    except FileNotFoundError as e:
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=404,
        )

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    device = _config.get("device", 0)
    results = model.predict(img, conf=confidence, device=device, verbose=False)

    objects = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append({
                "class": r.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                },
                "center_x": round((x1 + x2) / 2, 1),
                "center_y": round((y1 + y2) / 2, 1),
            })

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "success": True,
        "project": proj,
        "count": len(objects),
        "objects": objects,
        "inference_ms": round(elapsed, 1),
    }


@app.get("/models")
async def list_models():
    """List all available project models."""
    models_dir = Path(__file__).parent.parent / "models" / "projects"
    projects = []
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir():
                projects.append({
                    "project": p.name,
                    "has_pt": bool(list(p.rglob("best.pt"))),
                    "has_onnx": bool(list(p.rglob("best.onnx"))),
                })
    return {"projects": projects}


@app.get("/health")
async def health():
    """Health check with GPU info."""
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_models": list(_models.keys()),
    }


if __name__ == "__main__":
    import uvicorn

    load_config()
    port = _config.get("port", 8100)
    print(f"Starting YOLO service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
