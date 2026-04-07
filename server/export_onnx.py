"""
Export trained YOLO model to ONNX format.
Usage: python export_onnx.py --project robot_test_c4
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    model_dir = root / "models" / "projects" / args.project

    pt_paths = sorted(model_dir.rglob("best.pt"))
    if not pt_paths:
        print(f"ERROR: No best.pt found in {model_dir}")
        return

    model = YOLO(str(pt_paths[-1]))
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )
    print(f"ONNX exported next to {pt_paths[-1]}")


if __name__ == "__main__":
    main()
