import io
import os

import mlflow
import mlflow.pyfunc
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

app = FastAPI()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model("models:/food11@champion")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model.predict(tensor.numpy())
        outputs_tensor = torch.tensor(outputs)
        probs = torch.softmax(outputs_tensor, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)

    return {
        "category": CATEGORIES[predicted_idx.item()],
        "confidence": round(confidence.item(), 4),
    }