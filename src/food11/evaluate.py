import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow


def evaluate_model(version, dataset_path, batch_size=32):
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_uri = f"models:/food11/{version}"
    model = mlflow.pyfunc.load_model(model_uri)

    correct = 0
    total = 0

    for images, labels in loader:
        with torch.no_grad():
            outputs = model.predict(images.numpy())

        predicted = np.asarray(outputs).argmax(axis=1)
        correct += int((predicted == labels.numpy()).sum())
        total += len(labels)

    accuracy = correct / total

    print(f"Model version: {version}")
    print(f"Correct predictions: {correct}")
    print(f"Total images: {total}")
    print(f"Evaluation accuracy: {accuracy:.4f}")

    client = mlflow.MlflowClient()
    registered = client.get_model_version("food11", str(version))

    # Record this evaluation against the model's source run.
    client.log_metric(
        registered.run_id,
        "test_accuracy",
        accuracy,
    )

    client.set_model_version_tag(
        "food11",
        str(version),
        "evaluation_dataset",
        "food11_processed_mini/evaluation",
    )

    client.set_model_version_tag(
        "food11",
        str(version),
        "evaluation_size",
        str(total),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument(
        "--dataset",
        default="data/food11_processed_mini/evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    evaluate_model(args.version, args.dataset, args.batch_size)


if __name__ == "__main__":
    main()
