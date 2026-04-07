"""
Export trained YOLO model to ONNX format.
Usage: python export_onnx.py --project robot_test_c4 [--version v1]

v2 changes: opset 12 → 17 (OnnxRuntime 1.17+ compatible)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--version", default=None, help="Model version (default: latest)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    model_dir = root / "models" / "projects" / args.project

    # Find model
    if args.version:
        pt_path = model_dir / args.version / "best.pt"
        if not pt_path.exists():
            print(f"ERROR: {pt_path} not found")
            return
    else:
        # Find latest versioned or legacy
        pt_paths = sorted(model_dir.rglob("best.pt"))
        if not pt_paths:
            print(f"ERROR: No best.pt found in {model_dir}")
            return
        pt_path = pt_paths[-1]

    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=args.opset,
        dynamic=False,
    )
    print(f"ONNX exported (opset={args.opset}) next to {pt_path}")


if __name__ == "__main__":
    main()
