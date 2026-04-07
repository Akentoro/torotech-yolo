# Pipeline — YOLO + OpenCV 串接模組

> **設計原則**：YOLO 歸 YOLO，OpenCV 歸 OpenCV，串接的邏輯獨立一層。

## 為什麼要分離

```
server.py          — 通用 YOLO 推論服務（不依賴 OpenCV）
opencv/refiner.py  — 通用 OpenCV 精修（不依賴 YOLO）
pipeline/          — 串接層（依賴以上兩者，但兩者不依賴它）
```

這樣的好處：
1. `server.py` 可以單獨跑純 YOLO 推論（opencv_refine.enabled: false）
2. `refiner.py` 可以被其他專案引用（不綁定 FastAPI）
3. 串接邏輯集中在 `pipeline/`，改串接不影響 YOLO 和 OpenCV 的程式碼
4. 新增其他後處理步驟（如 tracking、counting）也加在 pipeline/ 裡

## 資料流

```
/detect request
  ↓
[server.py] YOLO inference → yolo_objects (list[dict])
  ↓
[pipeline/detect_pipeline.py] 
  ├── refine enabled  → [opencv/refiner.py] → RefinedObject → to_api_dict()
  └── refine disabled → passthrough (add angle=0, refined=false)
  ↓
API response JSON
```

## 設定

`server_config.yaml` 的 `opencv_refine` 區段：

| 參數 | 預設 | 說明 |
|------|------|------|
| `enabled` | true | false = 純 YOLO，不跑 OpenCV |
| `method` | adaptive | adaptive / canny / otsu |
| `margin_px` | 10 | bbox 外擴再裁切 ROI |
| `min_area` | 500 | 輪廓面積太小就放棄精修 |
| `blur_kernel` | 5 | GaussianBlur kernel |

## Fallback 機制

- refiner.py import 失敗 → 自動 disable，只回 YOLO
- 精修拋異常 → 該次全部 fallback 到 YOLO 結果
- 單個物件精修失敗 → 該物件 `refined: false`，其他正常
