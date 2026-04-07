# torotech-yolo

> ToroTech 工業視覺 YOLO 訓練與推論服務
> 跑在 4090 主機上，透過 HTTP API 提供偵測服務給設備端 C# 呼叫

## ⚠️ v2 更新（2026-04-07）

**設備端必讀**：合約有 breaking change，見 [contract/README.md](contract/README.md)。
主要改動：
- `/detect` 參數從 Query → Form field
- Response: `class` → `class_name` + 新增 `class_id`, `model_version`, `image_size`, `width`, `height`
- ONNX export opset: 12 → 17
- 新增 GPU 推論排隊（多設備同時呼叫不會互相干擾）

## 快速上手

### 4090 主機

```bash
# 1. Clone
git clone https://github.com/akentoro/torotech-yolo.git
cd torotech-yolo

# 2. 裝 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 裝其他依賴
pip install -r server/requirements.txt

# 4. 啟動推論服務
cd server && python server.py
# → http://0.0.0.0:8100
```

### 設備端

```bash
# 偵測（v2: 參數用 -F 不用 Query string）
curl -X POST http://4090-pc:8100/detect \
  -F "file=@snapshot.jpg" \
  -F "project=robot_test_c4" \
  -F "confidence=0.5"
```

```csharp
// C# v2 — 全部用 MultipartFormDataContent
var content = new MultipartFormDataContent();
content.Add(new ByteArrayContent(imgBytes), "file", "image.jpg");
content.Add(new StringContent("robot_test_c4"), "project");
content.Add(new StringContent("0.5"), "confidence");

var response = await httpClient.PostAsync($"{baseUrl}/detect", content, ct);
var result = await response.Content.ReadFromJsonAsync<DetectResponse>(ct);

// result.ClassId, result.ClassName (不是 Class), result.ModelVersion, result.ImageSize
```

## 架構

```
設備 PC (C# .NET 8)                      4090 主機 (本 repo)
┌──────────────────────┐                 ┌──────────────────────────┐
│ Vision Skill CLI     │  HTTP/JSON      │ server.py (FastAPI v2)   │
│ IObjectDetector      │ ◄──────────────►│ POST /detect             │
│  └── RemoteDetector  │                 │ GET  /models             │
│                      │                 │ GET  /health             │
│ OpenCvSharp4         │                 │                          │
│ (校正/幾何/servo)    │                 │ Ultralytics YOLO11       │
└──────────────────────┘                 │ GPU Semaphore (排隊推論) │
                                         └──────────────────────────┘
```

## 目錄結構

```
torotech-yolo/
├── contract/          ← API 合約（雙方溝通核心，改了必須 push）
├── server/            ← FastAPI 推論服務 + 訓練腳本
├── projects/          ← 每案設定（類別、超參數）
├── models/            ← 模型檔（不進 git，用 Releases 或共享資料夾）
├── docs/              ← 文件
└── tests/             ← 測試
```

## API 合約

見 [contract/README.md](contract/README.md) — **設備端和 4090 端都依據此合約開發**。

## 每案訓練

見 [docs/training-guide.md](docs/training-guide.md)

## 相關專案

- `Automation_eq/Robot_test_c4/` — 設備端 C# 視覺 + 動作系統
- `VISION/OpenCV_Vision/` — C# OpenCV 通用視覺引擎
- `VISION/Yolo/yolo-service/` — 4090 主機完整分層架構（本 repo 的 server.py 是精簡版）
