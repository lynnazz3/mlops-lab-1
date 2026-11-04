from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.food11 import serve


class FakeModel:
    def predict(self, x):
        # Logits: class 0 (Bread) has the highest score.
        return [[5.0] + [0.0] * 10]


def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")

    with TestClient(serve.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")

    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())

        # Generate an image in memory.
        img = Image.new("RGB", (128, 128), "white")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")

        response = client.post(
            "/predict",
            files={
                "file": (
                    "bread.jpg",
                    buffer.getvalue(),
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_without_model(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")

    with TestClient(serve.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", b"fake", "image/jpeg")},
        )

    assert response.status_code == 503
