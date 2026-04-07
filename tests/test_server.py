"""Tests for the inference server v2."""
import requests
import sys


BASE_URL = "http://localhost:8100"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print(f"  GPU: {data['gpu']} ({data.get('gpu_name', 'N/A')})")
    print(f"  VRAM: {data.get('gpu_memory_used_mb', '?')}MB / {data.get('gpu_memory_total_mb', '?')}MB")
    print(f"  Loaded models: {data['loaded_models']}")
    print(f"  Training active: {data.get('training_active', '?')}")
    print(f"  Uptime: {data.get('uptime_s', '?')}s")


def test_models():
    r = requests.get(f"{BASE_URL}/models")
    assert r.status_code == 200
    data = r.json()
    for p in data["projects"]:
        versions = [v["version"] for v in p.get("versions", [])]
        print(f"  {p['project_id']}: active={p.get('active_version')} versions={versions}")


def test_detect(image_path: str, project: str = "robot_test_c4"):
    """Test detect with v2 contract (Form fields, not Query params)."""
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/detect",
            files={"file": f},
            data={"project": project, "confidence": "0.3"},  # Form fields
        )
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True

    # v2 fields
    print(f"  Success: {data['success']}")
    print(f"  Project: {data['project']}")
    print(f"  Model version: {data.get('model_version', '?')}")
    print(f"  Image size: {data.get('image_size', '?')}")
    print(f"  Count: {data['count']}")
    print(f"  Inference: {data['inference_ms']}ms")

    for obj in data.get("objects", []):
        # v2: class_name + class_id instead of class
        print(f"    [{obj['class_id']}] {obj['class_name']} "
              f"({obj['confidence']:.2f}) "
              f"at ({obj['center_x']}, {obj['center_y']}) "
              f"size={obj.get('width', '?')}x{obj.get('height', '?')}")


if __name__ == "__main__":
    print("[health]")
    test_health()
    print("\n[models]")
    test_models()

    if len(sys.argv) > 1:
        img = sys.argv[1]
        proj = sys.argv[2] if len(sys.argv) > 2 else "robot_test_c4"
        print(f"\n[detect] {img} (project={proj})")
        test_detect(img, proj)
    else:
        print("\n[detect] skipped (usage: python test_server.py <image_path> [project])")
