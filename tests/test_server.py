"""Basic tests for the inference server."""
import requests
import sys


BASE_URL = "http://localhost:8100"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print(f"  GPU: {data['gpu']} ({data.get('gpu_name', 'N/A')})")
    print(f"  Loaded models: {data['loaded_models']}")


def test_models():
    r = requests.get(f"{BASE_URL}/models")
    assert r.status_code == 200
    data = r.json()
    print(f"  Projects: {[p['project'] for p in data['projects']]}")


def test_detect(image_path: str, project: str = "robot_test_c4"):
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/detect",
            files={"file": f},
            data={"project": project, "confidence": 0.5},
        )
    assert r.status_code == 200
    data = r.json()
    print(f"  Success: {data['success']}")
    print(f"  Count: {data.get('count', 0)}")
    print(f"  Inference: {data.get('inference_ms', '?')}ms")
    for obj in data.get("objects", []):
        print(f"    {obj['class']} ({obj['confidence']:.2f}) at ({obj['center_x']}, {obj['center_y']})")


if __name__ == "__main__":
    print("[health]")
    test_health()
    print("[models]")
    test_models()

    if len(sys.argv) > 1:
        print(f"[detect] {sys.argv[1]}")
        test_detect(sys.argv[1])
    else:
        print("[detect] skipped (pass image path as argument)")
