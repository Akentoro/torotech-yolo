"""
Detect Pipeline — 串接 YOLO 推論 + OpenCV 精修

設計原則：
  - server.py 保持通用（純 YOLO 推論）
  - refiner.py 保持通用（純 OpenCV 精修）
  - 本模組負責串接：YOLO objects → OpenCV refine → API response
  - 可獨立開關精修（config driven）
  - 不改動 server.py 和 refiner.py 的任何程式碼

用法：
  pipeline = DetectPipeline(config)
  api_objects = pipeline.process(image_np, yolo_objects)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DetectPipeline:
    """串接 YOLO 推論結果和 OpenCV 精修。

    YOLO objects in → (optional) OpenCV refine → API-ready dicts out
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._refine_enabled = cfg.get("enabled", False)
        self._refiner = None

        if self._refine_enabled:
            try:
                from opencv.refiner import OpenCvRefiner
                self._refiner = OpenCvRefiner(config=cfg)
                logger.info(
                    "OpenCV refine enabled (method=%s, margin=%dpx)",
                    cfg.get("method", "adaptive"),
                    cfg.get("margin_px", 10),
                )
            except ImportError:
                logger.warning("opencv.refiner not found, refine disabled")
                self._refine_enabled = False
            except Exception as e:
                logger.error("Failed to init OpenCvRefiner: %s", e)
                self._refine_enabled = False

    @property
    def refine_enabled(self) -> bool:
        return self._refine_enabled and self._refiner is not None

    def process(self, image: np.ndarray, yolo_objects: list[dict]) -> list[dict]:
        """Process YOLO detection results through the pipeline.

        Args:
            image: BGR numpy array (original image)
            yolo_objects: YOLO output dicts, each with:
                class_id, class_name, confidence, bbox, center_x, center_y, width, height

        Returns:
            API-ready dicts — same structure as input but with added
            angle, contour_area, rectangularity, refined fields.
            If refine is disabled, adds default values (angle=0, refined=false).
        """
        if not yolo_objects:
            return []

        if self.refine_enabled:
            return self._process_with_refine(image, yolo_objects)
        else:
            return self._process_passthrough(yolo_objects)

    def _process_with_refine(self, image: np.ndarray, yolo_objects: list[dict]) -> list[dict]:
        """YOLO + OpenCV refine pipeline."""
        try:
            refined_list = self._refiner.refine(image, yolo_objects)
            return [r.to_api_dict() for r in refined_list]
        except Exception as e:
            logger.error("OpenCV refine failed, falling back to YOLO only: %s", e)
            return self._process_passthrough(yolo_objects)

    def _process_passthrough(self, yolo_objects: list[dict]) -> list[dict]:
        """Passthrough: add default refine fields to YOLO objects."""
        result = []
        for obj in yolo_objects:
            out = dict(obj)
            out.setdefault("angle", 0.0)
            out.setdefault("contour_area", 0.0)
            out.setdefault("rectangularity", 0.0)
            out.setdefault("refined", False)
            result.append(out)
        return result
