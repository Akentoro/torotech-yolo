# OpenCV Refiner — YOLO 後精修模組

> YOLO 給 bbox + class → OpenCV 在 bbox 內做 sub-pixel centroid + 旋轉角度
> 被 `server.py` 的 `/detect` 整合呼叫，設備端打一次 API 就拿到精修結果

## 精修流程

```
YOLO 偵測結果                    OpenCV 精修
┌────────────────────┐          ┌──────────────────────────┐
│ class_name: "box"  │          │ 裁切 bbox ROI            │
│ confidence: 0.95   │    →     │ → GaussianBlur           │
│ center_x: 520      │          │ → AdaptiveThreshold      │
│ center_y: 390      │          │ → FindContours           │
│ bbox: 400,280~640  │          │ → Moments sub-pixel      │
│ angle: (無)        │          │ → MinAreaRect angle      │
└────────────────────┘          └──────────────────────────┘
                                         │
                                         ▼
                                ┌──────────────────────────┐
                                │ center_x: 520.37 (精修)  │
                                │ center_y: 389.82 (精修)  │
                                │ angle: -12.5° (新增)     │
                                │ contour_area: 45230      │
                                │ rectangularity: 0.87     │
                                │ refined: true            │
                                └──────────────────────────┘
```

## API Response 變化

`/detect` response 的每個 object 新增欄位：

| 欄位 | 說明 | YOLO only | + OpenCV refine |
|------|------|-----------|----------------|
| center_x | 中心 X | ±5px | ±0.5px |
| center_y | 中心 Y | ±5px | ±0.5px |
| **angle** | 旋轉角度 (deg) | 無 | -90°~0° |
| **contour_area** | 輪廓面積 | 無 | float |
| **rectangularity** | 矩形度 | 無 | 0~1 |
| **refined** | 是否精修成功 | 無 | bool |

## 設定

在 `server_config.yaml` 加：

```yaml
opencv_refine:
  enabled: true          # false = 跳過精修，只回 YOLO 原始
  method: adaptive       # adaptive / canny / otsu
  margin_px: 10          # bbox 外擴 margin
  min_area: 500          # 最小輪廓面積
  blur_kernel: 5         # GaussianBlur kernel size
```

## 使用

```python
from opencv.refiner import OpenCvRefiner

refiner = OpenCvRefiner(config={"method": "adaptive"})
refined = refiner.refine(image_np, yolo_objects)
# → list of RefinedObject with sub-pixel center + angle
```
