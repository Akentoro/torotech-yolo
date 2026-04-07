"""
YOLO Training Script
Usage: python train.py --project robot_test_c4 [--epochs 100] [--model yolov8n.pt]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for a project")
    parser.add_argument("--project", required=True, help="Project ID (e.g. robot_test_c4)")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=int, default=0, help="CUDA device (0=first GPU, -1=CPU)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    data_yaml = root / "projects" / args.project / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Create it first.")
        return

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(root / "models" / "projects" / args.project),
        name="train",
        exist_ok=True,

        # Industrial scene augmentation
        patience=20,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=15,
        translate=0.1,
        scale=0.3,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
    )
    print(f"Training complete. Best model: models/projects/{args.project}/train/weights/best.pt")


if __name__ == "__main__":
    main()
