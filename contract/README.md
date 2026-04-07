# API Contract — 設備端 ↔ 4090 推論服務

> 雙方都依據此合約開發。任何改動需同步更新此目錄下的 schema。

## POST /detect

偵測圖片中的物件。

### Request

- Method: `POST`
- Content-Type: `multipart/form-data`
- URL: `http://{4090-host}:8100/detect`

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| file | File | Yes | 圖片檔 (JPG/PNG/BMP) |
| project | string | No | 專案 ID（預設 server_config.yaml 的 default_project） |
| confidence | float | No | 信心度門檻（預設 0.5） |

### Response

```json
{
  "success": true,
  "project": "robot_test_c4",
  "count": 2,
  "objects": [
    {
      "class": "box",
      "confidence": 0.95,
      "bbox": { "x1": 400.0, "y1": 280.0, "x2": 640.0, "y2": 500.0 },
      "center_x": 520.0,
      "center_y": 390.0
    }
  ],
  "inference_ms": 12.5
}
```

### Error Response

```json
{
  "success": false,
  "error": "No model found for project: unknown_project"
}
```

## GET /models

列出所有可用的專案模型。

### Response

```json
{
  "projects": [
    { "project": "robot_test_c4", "has_pt": true, "has_onnx": true },
    { "project": "m1256_battery", "has_pt": true, "has_onnx": false }
  ]
}
```

## GET /health

健康檢查。

### Response

```json
{
  "status": "ok",
  "gpu": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "loaded_models": ["robot_test_c4"]
}
```
