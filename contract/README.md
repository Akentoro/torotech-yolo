# API Contract v2 — 設備端 ↔ 4090 推論服務

> **雙方都依據此合約開發。任何改動需同步更新此目錄下的 schema，然後 push。**

## ⚠️ v2 Breaking Changes（2026-04-07）

### 設備端 C# 必須修改：

1. **`/detect` 參數從 Query 改為 Form field**
   ```csharp
   // 舊（v1）— 不要再用
   var url = $"{baseUrl}/detect?project=xxx&confidence=0.5";
   content.Add(new ByteArrayContent(imgBytes), "file", "image.jpg");

   // 新（v2）— 全部放 MultipartFormDataContent
   content.Add(new ByteArrayContent(imgBytes), "file", "image.jpg");
   content.Add(new StringContent("robot_test_c4"), "project");
   content.Add(new StringContent("0.5"), "confidence");
   ```

2. **Response 欄位改名 + 新增**
   ```csharp
   // 舊（v1）
   obj["class"]       // ← C# 保留字，需要 @class 或 [JsonProperty]

   // 新（v2）
   obj["class_name"]  // ← 不衝突了
   obj["class_id"]    // ← 新增：int，方便 switch
   obj["width"]       // ← 新增：bbox 寬度
   obj["height"]      // ← 新增：bbox 高度

   // 頂層新增
   response["model_version"]  // ← "v1", "v2" 等
   response["image_size"]     // ← {"width": 1280, "height": 960}
   ```

---

## 端點一覽

| Method | Path | Schema | 說明 |
|--------|------|--------|------|
| POST | `/detect` | [request](detect-request.schema.json) / [response](detect-response.schema.json) | 圖片偵測 |
| GET | `/models` | [response](models-response.schema.json) | 列出專案模型 |
| GET | `/health` | [response](health-response.schema.json) | 健康檢查 |

---

## POST /detect

偵測圖片中的物件。

### Request (multipart/form-data)

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| file | File (Form) | Yes | 圖片檔 (JPEG/PNG/BMP/TIFF)，max 10MB |
| project | string (Form) | No | 專案 ID（預設 server_config 的 default_project） |
| confidence | float (Form) | No | 信心度門檻（預設 0.5） |
| version | string (Form) | No | 模型版本，如 "v1"、"v2"（預設用 active version） |

### Response (200)

```json
{
  "success": true,
  "project": "robot_test_c4",
  "model_version": "v2",
  "count": 2,
  "objects": [
    {
      "class_id": 0,
      "class_name": "box",
      "confidence": 0.95,
      "bbox": { "x1": 400.0, "y1": 280.0, "x2": 640.0, "y2": 500.0 },
      "center_x": 520.0,
      "center_y": 390.0,
      "width": 240.0,
      "height": 220.0
    }
  ],
  "image_size": { "width": 1280, "height": 960 },
  "inference_ms": 12.5
}
```

### Error Response

```json
{ "success": false, "error": "No model found for project: unknown_project" }
```

### HTTP Status Codes

| Code | 意義 | C# 端處理 |
|------|------|-----------|
| 200 | 成功 | 解析 JSON |
| 400 | 圖片無效 / 格式不支援 | 檢查圖片 |
| 404 | 模型不存在 | 報警 + fallback |
| 413 | 圖片太大 (>10MB) | 壓縮/縮小 |
| 504 | GPU 推論 timeout (30s) | 重試 1 次 |

---

## GET /models

列出所有可用的專案模型（含版本資訊）。

### Response

```json
{
  "projects": [
    {
      "project_id": "robot_test_c4",
      "active_version": "v1",
      "versions": [
        { "version": "v1", "has_pt": true, "has_onnx": true },
        { "version": "v2", "has_pt": true, "has_onnx": false }
      ]
    }
  ]
}
```

---

## GET /health

健康檢查。

### Response

```json
{
  "status": "ok",
  "gpu": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_used_mb": 2048,
  "gpu_memory_total_mb": 24564,
  "loaded_models": ["robot_test_c4"],
  "training_active": false,
  "uptime_s": 86400
}
```

---

## C# Client 建議 DTO

```csharp
public record DetectResponse(
    bool Success,
    string? Project,
    string? ModelVersion,
    int Count,
    List<DetectedObject>? Objects,
    ImageSize? ImageSize,
    double InferenceMs,
    string? Error
);

public record DetectedObject(
    int ClassId,
    string ClassName,
    double Confidence,
    BoundingBox Bbox,
    double CenterX,
    double CenterY,
    double Width,
    double Height
);

public record BoundingBox(double X1, double Y1, double X2, double Y2);
public record ImageSize(int Width, int Height);
```
