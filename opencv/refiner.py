"""
OpenCV Refiner — YOLO bbox 後的 sub-pixel 精修 + 角度偵測

用途：YOLO 給出粗 bbox → 這裡在 bbox 內做精修
- Sub-pixel centroid (moments)
- MinAreaRect 旋轉角度
- Contour area / aspect ratio
- 可選：HoughLinesP 邊線角度、Template Matching

設計：被 server.py 的 /detect 呼叫，不獨立對外服務。
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RefinedObject:
    """精修後的物件資料（覆蓋 YOLO 的粗定位）"""
    # 保留 YOLO 的分類資訊
    class_id: int
    class_name: str
    confidence: float

    # OpenCV 精修的精確位置（相對於原圖，不是 ROI）
    center_x: float
    center_y: float

    # YOLO 原始 bbox（不變）
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    width: float
    height: float

    # OpenCV 新增的資訊
    angle: float              # MinAreaRect 旋轉角度 (degrees)
    contour_area: float       # 輪廓面積
    rectangularity: float     # 矩形度 (0~1)
    refined: bool             # 是否成功精修（False = 用 YOLO 原始值）

    def to_api_dict(self) -> dict:
        """轉成 API response 格式"""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {
                "x1": round(self.bbox_x1, 1),
                "y1": round(self.bbox_y1, 1),
                "x2": round(self.bbox_x2, 1),
                "y2": round(self.bbox_y2, 1),
            },
            "center_x": round(self.center_x, 2),  # sub-pixel 精度
            "center_y": round(self.center_y, 2),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "angle": round(self.angle, 2),
            "contour_area": round(self.contour_area, 1),
            "rectangularity": round(self.rectangularity, 3),
            "refined": self.refined,
        }


class OpenCvRefiner:
    """
    在 YOLO bbox 內做 OpenCV 精修。

    支援的精修方法：
    1. HSV + Contour → sub-pixel centroid + angle (預設)
    2. Adaptive threshold + Contour (for variable lighting)
    3. Canny + Contour (for edge-based)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.margin_px = cfg.get("margin_px", 10)       # bbox 外擴 margin
        self.min_area = cfg.get("min_area", 500)         # 最小輪廓面積
        self.blur_kernel = cfg.get("blur_kernel", 5)     # GaussianBlur kernel
        self.method = cfg.get("method", "adaptive")      # adaptive / hsv / canny

    def refine(self, image: np.ndarray, yolo_objects: list[dict]) -> list[RefinedObject]:
        """
        對每個 YOLO bbox 做精修。

        Args:
            image: BGR numpy array (原圖)
            yolo_objects: YOLO 回傳的 objects list (每個有 class_id, class_name, confidence, bbox, center_x, center_y, width, height)

        Returns:
            list of RefinedObject (與輸入等長，精修失敗的保留 YOLO 原始值)
        """
        h, w = image.shape[:2]
        results = []

        for obj in yolo_objects:
            bbox = obj["bbox"]
            x1 = max(0, int(bbox["x1"]) - self.margin_px)
            y1 = max(0, int(bbox["y1"]) - self.margin_px)
            x2 = min(w, int(bbox["x2"]) + self.margin_px)
            y2 = min(h, int(bbox["y2"]) + self.margin_px)

            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                results.append(self._fallback(obj))
                continue

            # 精修
            refined = self._refine_roi(roi, x1, y1)
            if refined is None:
                results.append(self._fallback(obj))
                continue

            results.append(RefinedObject(
                class_id=obj["class_id"],
                class_name=obj["class_name"],
                confidence=obj["confidence"],
                center_x=refined["cx"],
                center_y=refined["cy"],
                bbox_x1=bbox["x1"],
                bbox_y1=bbox["y1"],
                bbox_x2=bbox["x2"],
                bbox_y2=bbox["y2"],
                width=obj["width"],
                height=obj["height"],
                angle=refined["angle"],
                contour_area=refined["area"],
                rectangularity=refined["rect"],
                refined=True,
            ))

        return results

    def _refine_roi(self, roi: np.ndarray, offset_x: int, offset_y: int) -> Optional[dict]:
        """在 ROI 內找最大輪廓，算 sub-pixel centroid + angle"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # 二值化（根據 method）
        if self.method == "adaptive":
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        elif self.method == "canny":
            binary = cv2.Canny(blurred, 50, 150)
            binary = cv2.dilate(binary, None, iterations=1)
        else:  # otsu
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形態學清理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 取最大輪廓
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < self.min_area:
            return None

        # Sub-pixel centroid (moments)
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"] + offset_x
        cy = M["m01"] / M["m00"] + offset_y

        # MinAreaRect → angle
        rect = cv2.minAreaRect(best)
        angle = rect[2]  # -90 ~ 0 degrees

        # Rectangularity
        bbox_area = rect[1][0] * rect[1][1]
        rectangularity = area / bbox_area if bbox_area > 0 else 0

        return {"cx": cx, "cy": cy, "angle": angle, "area": area, "rect": rectangularity}

    def _fallback(self, obj: dict) -> RefinedObject:
        """精修失敗，回傳 YOLO 原始值"""
        return RefinedObject(
            class_id=obj["class_id"],
            class_name=obj["class_name"],
            confidence=obj["confidence"],
            center_x=obj["center_x"],
            center_y=obj["center_y"],
            bbox_x1=obj["bbox"]["x1"],
            bbox_y1=obj["bbox"]["y1"],
            bbox_x2=obj["bbox"]["x2"],
            bbox_y2=obj["bbox"]["y2"],
            width=obj["width"],
            height=obj["height"],
            angle=0.0,
            contour_area=0.0,
            rectangularity=0.0,
            refined=False,
        )
