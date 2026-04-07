# YOLO 訓練與推論服務 — 4090 主機獨立開發

> **部署主機**：RTX 4090 64GB（獨立 PC，非設備 PC）
> **定位**：每案訓練不同模型，提供 HTTP 推論服務給設備端呼叫
> **與設備端關係**：設備端 OpenCV 做通用處理，YOLO 做分類偵測，兩者互補

---

## 一、架構定位

```
設備 PC (C# .NET 8)                      4090 主機 (Python)
┌──────────────────────┐                 ┌──────────────────────────┐
│ Vision Skill CLI     │                 │ YOLO Service             │
│                      │  HTTP/JSON      │                          │
│ IObjectDetector      │ ◄──────────────►│ FastAPI /detect          │
│  ├── HsvDetector     │                 │ FastAPI /train/status    │
│  ├── YoloDetector ───┤── 呼叫 ──────►  │                          │
│  └── RemoteDetector  │                 │ Ultralytics YOLOv8/v11   │
│                      │                 │ ONNX Export              │
│ OpenCvSharp4         │                 │ Data Augmentation        │
│ (校正/幾何/servo)    │                 │ Label Management         │
└──────────────────────┘                 └──────────────────────────┘
```

### 兩種推論模式

| 模式 | 方式 | 延遲 | 適用場景 |
|------|------|------|---------|
| **遠端推論** | 設備端 HTTP → 4090 FastAPI → JSON 結果 | ~50-100ms | 需要大模型/高精度 |
| **本機推論** | ONNX 模型下載到設備端，YoloDotNet CPU 推論 | ~100-150ms | 離線/低延遲 |

---

## 二、4090 主機環境

### 硬體
- GPU: NVIDIA RTX 4090 (24GB VRAM) / 或 RTX 4090 with 64GB system RAM
- OS: Windows 或 Linux（建議 Ubuntu 22.04 for CUDA 最佳支援）
- CUDA: 12.x
- Python: 3.10+

### 軟體安裝

```bash
# 基礎環境
conda create -n yolo python=3.10 -y
conda activate yolo

# PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Ultralytics (YOLO)
pip install ultralytics

# 推論服務
pip install fastapi uvicorn python-multipart

# 標註工具
pip install labelimg  # 或用 CVAT (web-based)

# ONNX 匯出
pip install onnx onnxruntime-gpu
```

### 目錄結構

```
yolo-service/
├── server.py                  ← FastAPI 推論服務
├── train.py                   ← 訓練腳本
├── export_onnx.py             ← ONNX 匯出
├── augment.py                 ← 資料增強
├── config/
│   └── server_config.yaml     ← 服務設定（port, model path, 等）
├── models/
│   ├── yolov8n.pt             ← 預訓練基底模型
│   └── projects/              ← 每案訓練的模型
│       ├── robot_test_c4/
│       │   ├── best.pt
│       │   ├── best.onnx
│       │   └── data.yaml
│       ├── m1256_battery/
│       └── ...
├── datasets/
│   ├── robot_test_c4/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── data.yaml
│   └── ...
└── logs/
```

---

## 三、訓練流程

### 3.1 資料收集

在設備端用 Vision Skill CLI 或 Data Capture 工具拍照：

```bash
# 設備端：拍照存檔
vision shot --save dataset/img_001.jpg
vision shot --save dataset/img_002.jpg
# ... 或用未來的 DataCapture 工具自動多角度拍攝
```

將照片複製到 4090 主機 `datasets/{project}/images/` 目錄。

### 3.2 標註

```bash
# 方式 A: LabelImg (本機)
labelimg datasets/robot_test_c4/images/train/ \
         datasets/robot_test_c4/labels/train/ \
         datasets/robot_test_c4/classes.txt

# 方式 B: CVAT (web-based, 推薦多人協作)
# docker compose up → 瀏覽器開 http://localhost:8080

# 方式 C: Roboflow (雲端, 免費 tier 支援小資料集)
```

標註格式：YOLO txt（每行 `class_id cx cy w h`，normalized 0-1）

### 3.3 data.yaml

```yaml
# datasets/robot_test_c4/data.yaml
path: /home/user/yolo-service/datasets/robot_test_c4
train: images/train
val: images/val

nc: 3  # 類別數
names:
  0: box
  1: pcba
  2: connector
```

### 3.4 訓練

```bash
# train.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano 模型，適合工業場景

results = model.train(
    data="datasets/robot_test_c4/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,           # 4090 可以開大 batch
    device=0,           # GPU 0
    project="models/projects/robot_test_c4",
    name="train_v1",
    
    # 工業場景建議參數
    patience=20,        # early stopping
    augment=True,       # Ultralytics 內建增強
    mosaic=1.0,         # mosaic 增強
    mixup=0.1,          # mixup 增強
    degrees=15,         # 旋轉 ±15°
    translate=0.1,      # 平移 10%
    scale=0.3,          # 縮放 30%
    hsv_h=0.015,        # 色調微調
    hsv_s=0.3,          # 飽和度微調
    hsv_v=0.2,          # 亮度微調
)
```

### 3.5 訓練數據量建議

| 數據量 | 預期 mAP50 | 適用場景 |
|--------|-----------|---------|
| 50-100 張 + 增強 | 80-90% | 快速原型驗證 |
| 200-500 張 + 增強 | 90-95% | 穩定生產用 |
| 1000+ 張 | 95%+ | 高精度/多變環境 |

**工業場景技巧**：
- 光源穩定 → 少量數據就夠
- 拍攝角度固定（Eye-in-Hand 朝下）→ 不需要太多角度變化
- 背景一致 → 模型收斂快

### 3.6 ONNX 匯出

```bash
# export_onnx.py
from ultralytics import YOLO

model = YOLO("models/projects/robot_test_c4/train_v1/weights/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    simplify=True,      # ONNX 圖簡化
    opset=12,           # ONNX opset (YoloDotNet 相容)
    dynamic=False,      # 固定輸入尺寸（推論更快）
)
# → models/projects/robot_test_c4/train_v1/weights/best.onnx
```

---

## 四、推論服務 (FastAPI)

### 4.1 server.py

```python
"""YOLO 推論服務 — 跑在 4090 主機上"""
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

app = FastAPI(title="YOLO Inference Service")

# 全域模型快取
_models: dict[str, YOLO] = {}
_config: dict = {}


def load_config():
    global _config
    cfg_path = Path("config/server_config.yaml")
    if cfg_path.exists():
        _config = yaml.safe_load(cfg_path.read_text())
    else:
        _config = {"port": 8100, "default_project": "robot_test_c4"}


def get_model(project: str) -> YOLO:
    if project not in _models:
        model_dir = Path(f"models/projects/{project}")
        # 優先用 .pt（GPU 推論更快），fallback .onnx
        pt_path = sorted(model_dir.rglob("best.pt"))
        if pt_path:
            _models[project] = YOLO(str(pt_path[-1]))
        else:
            onnx_path = sorted(model_dir.rglob("best.onnx"))
            if onnx_path:
                _models[project] = YOLO(str(onnx_path[-1]))
            else:
                raise FileNotFoundError(f"No model found for project: {project}")
    return _models[project]


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    project: str = Query(default=None),
    confidence: float = Query(default=0.5),
):
    """偵測圖片中的物件，回傳 JSON"""
    t0 = time.perf_counter()
    proj = project or _config.get("default_project", "robot_test_c4")
    
    try:
        model = get_model(proj)
    except FileNotFoundError as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=404)
    
    # 讀取圖片
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    # 推論
    results = model.predict(img, conf=confidence, verbose=False)
    
    # 解析結果
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
    """列出所有可用的專案模型"""
    models_dir = Path("models/projects")
    projects = []
    for p in models_dir.iterdir():
        if p.is_dir():
            has_pt = bool(list(p.rglob("best.pt")))
            has_onnx = bool(list(p.rglob("best.onnx")))
            projects.append({
                "project": p.name,
                "has_pt": has_pt,
                "has_onnx": has_onnx,
            })
    return {"projects": projects}


@app.get("/health")
async def health():
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
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 4.2 啟動服務

```bash
cd yolo-service
python server.py
# → Uvicorn running on http://0.0.0.0:8100
```

### 4.3 API 呼叫範例

```bash
# 偵測
curl -X POST http://4090-pc:8100/detect \
  -F "file=@snapshot.jpg" \
  -F "project=robot_test_c4" \
  -F "confidence=0.5"

# 回傳:
# {
#   "success": true,
#   "project": "robot_test_c4",
#   "count": 2,
#   "objects": [
#     {"class": "box", "confidence": 0.95, "center_x": 520.3, "center_y": 390.7, ...},
#     {"class": "pcba", "confidence": 0.88, "center_x": 300.1, "center_y": 200.5, ...}
#   ],
#   "inference_ms": 12.5
# }

# 列出模型
curl http://4090-pc:8100/models

# 健康檢查
curl http://4090-pc:8100/health
```

---

## 五、設備端整合 (C#)

### 5.1 RemoteYoloDetector

在設備端 `RobotTestC4.Infrastructure/Vision/` 新增：

```csharp
/// <summary>
/// 透過 HTTP 呼叫 4090 主機的 YOLO 推論服務。
/// 實作 IObjectDetector，可透過 DI 熱切換。
/// </summary>
public class RemoteYoloDetector : IObjectDetector
{
    // POST /detect with multipart/form-data (image file)
    // Parse JSON response → DetectionResult
    // Config: appsettings.json "Vision:Yolo:RemoteUrl" = "http://4090-pc:8100"
}
```

### 5.2 本機 ONNX 推論 (備選)

```csharp
/// <summary>
/// 用 YoloDotNet NuGet 在本機 CPU/DirectML 推論。
/// ONNX 模型從 4090 主機下載到 models/ 資料夾。
/// </summary>
public class LocalYoloDetector : IObjectDetector
{
    // NuGet: YoloDotNet + Microsoft.ML.OnnxRuntime
    // Config: appsettings.json "Vision:Yolo:ModelPath" = "models/best.onnx"
}
```

### 5.3 DI 切換

```json
{
  "Vision": {
    "DefaultDetector": "hsv",
    "Yolo": {
      "Mode": "remote",
      "RemoteUrl": "http://4090-pc:8100",
      "Project": "robot_test_c4",
      "Confidence": 0.5,
      "ModelPath": "models/best.onnx",
      "UseGpu": false
    }
  }
}
```

`DefaultDetector` = `"yolo"` 時：
- `Mode: "remote"` → 注入 `RemoteYoloDetector`
- `Mode: "local"` → 注入 `LocalYoloDetector`

---

## 六、每案訓練流程 (SOP)

### 新案子啟動時

```
1. 設備端拍照收集（vision shot / DataCapture 工具）
   └→ 複製到 4090: datasets/{project_id}/images/

2. 標註（LabelImg / CVAT / Roboflow）
   └→ labels/{project_id}/labels/
   └→ 建立 data.yaml

3. 訓練（4090 主機）
   └→ python train.py --project {project_id}
   └→ 產出 models/projects/{project_id}/best.pt

4. 匯出 ONNX
   └→ python export_onnx.py --project {project_id}
   └→ 產出 best.onnx

5. 部署
   └→ 遠端：重啟 server.py（自動載入新模型）
   └→ 本機：複製 best.onnx 到設備端 models/

6. 設備端切換 detector
   └→ appsettings.json: "DefaultDetector": "yolo"
   └→ 或 CLI: vision detect --detector yolo
```

### 迭代改善

```
7. 收集誤判案例
   └→ vision detect --save-image → 存圖 + 結果

8. 補標註 + 重訓
   └→ 加入誤判圖片到 dataset
   └→ 重新 train + export

9. A/B 切換
   └→ 舊模型: models/projects/{project_id}/train_v1/
   └→ 新模型: models/projects/{project_id}/train_v2/
   └→ server.py 支援 ?version=v2 切換
```

---

## 七、授權注意

| 元件 | 授權 | 商用影響 |
|------|------|---------|
| Ultralytics (Python 訓練) | AGPL-3.0 | 訓練用，不進設備 |
| ONNX 模型檔案 | 你的資產 | 自由部署 |
| YoloDotNet (C# 推論) | MIT | 自由商用 |
| OnnxRuntime | MIT | 自由商用 |
| FastAPI | MIT | 自由商用 |

**結論**：Ultralytics AGPL 只影響 Python 訓練程式碼（留在 4090 主機）。匯出的 ONNX 模型是你的資產，設備端用 MIT 授權的 YoloDotNet 推論，完全沒有授權問題。

---

## 八、相關文件

- [AI感知架構規劃.md](AI感知架構規劃.md) — 全系統主文件
- [vision-architecture.md](vision-architecture.md) — Vision 模組技術設計
- OpenCV_Vision `vision-pipeline-guide.md` — 前處理最佳實踐
