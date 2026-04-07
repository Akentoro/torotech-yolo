# torotech-yolo

> ToroTech 工業視覺 YOLO 訓練與推論服務
> 跑在 4090 主機上，透過 HTTP API 提供偵測服務給設備端 C# 呼叫

## 快速上手

### 4090 主機

```bash
# 1. Clone
git clone https://github.com/akentoro/torotech-yolo.git
cd torotech-yolo

# 2. 建環境
conda create -n yolo python=3.10 -y && conda activate yolo
pip install -r server/requirements.txt

# 3. 啟動推論服務
cd server && python server.py
# → http://0.0.0.0:8100
```

### 設備端

```bash
# 偵測（傳圖片，收 JSON）
curl -X POST http://4090-pc:8100/detect \
  -F "file=@snapshot.jpg" \
  -F "project=robot_test_c4"

# 或在 C# 中：
# appsettings.json → "DefaultDetector": "yolo", "Yolo": { "Mode": "remote", "RemoteUrl": "http://4090-pc:8100" }
# → IObjectDetector 自動切到 RemoteYoloDetector
```

## 架構

```
設備 PC (C# .NET 8)                      4090 主機 (本 repo)
┌──────────────────────┐                 ┌──────────────────────────┐
│ Vision Skill CLI     │  HTTP/JSON      │ server.py (FastAPI)      │
│ IObjectDetector      │ ◄──────────────►│ POST /detect             │
│  └── RemoteDetector  │                 │ GET  /models             │
│                      │                 │ GET  /health             │
│ OpenCvSharp4         │                 │                          │
│ (校正/幾何/servo)    │                 │ Ultralytics YOLOv8/v11   │
└──────────────────────┘                 └──────────────────────────┘
```

## 目錄結構

```
torotech-yolo/
├── contract/          ← API 合約（雙方溝通核心）
├── server/            ← FastAPI 推論服務 + 訓練腳本
├── projects/          ← 每案設定（類別、超參數）
├── models/            ← ONNX 模型（用 GitHub Releases 管理大檔）
├── docs/              ← 完整文件
└── tests/             ← 測試
```

## API 合約

見 [contract/README.md](contract/README.md) — 設備端和 4090 端都依據此合約開發。

## 每案訓練

見 [docs/training-guide.md](docs/training-guide.md)

## 相關專案

- `Automation_eq/Robot_test_c4/` — 設備端 C# 視覺 + 動作系統
- `VISION/OpenCV_Vision/` — C# OpenCV 通用視覺引擎
