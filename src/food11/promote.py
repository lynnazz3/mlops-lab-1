import argparse
import math
import os

from mlflow import MlflowClient

MODEL_NAME = "food11"
METRIC_NAME = "test_accuracy"


def get_accuracy(client, model_version):
    """Get the test accuracy from a registered model's source run."""
    run = client.get_run(model_version.run_id)

    if METRIC_NAME not in run.data.metrics:
        raise ValueError(f"Run {model_version.run_id} has no {METRIC_NAME} metric")

    return run.data.metrics[METRIC_NAME]


def promote_candidate(version, test_size):
    if test_size <= 0:
        raise ValueError("Test size must be positive")

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    client = MlflowClient(tracking_uri=tracking_uri)

    candidate = client.get_model_version(MODEL_NAME, str(version))
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")

    if str(candidate.version) == str(champion.version):
        raise ValueError("Candidate is already the champion; choose another version")

    candidate_accuracy = get_accuracy(client, candidate)
    champion_accuracy = get_accuracy(client, champion)

    se = math.sqrt(champion_accuracy * (1 - champion_accuracy) / test_size)

    threshold = champion_accuracy + 2 * se

    print(f"Champion version: {champion.version}")
    print(f"Champion accuracy: {champion_accuracy:.4f}")
    print(f"Candidate version: {candidate.version}")
    print(f"Candidate accuracy: {candidate_accuracy:.4f}")
    print(f"Standard error: {se:.4f}")
    print(f"Promotion threshold: {threshold:.4f}")

    if candidate_accuracy >= threshold:
        decision = "promoted"
        client.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
        print("Decision: PROMOTED to champion")
    else:
        decision = "rejected"
        client.set_registered_model_alias(MODEL_NAME, "challenger", candidate.version)
        print("Decision: Candidate assigned challenger alias")

    for key, value in {
        "promotion_decision": decision,
        "candidate_accuracy": str(candidate_accuracy),
        "champion_accuracy": str(champion_accuracy),
        "promotion_threshold": str(threshold),
    }.items():
        client.set_model_version_tag(MODEL_NAME, candidate.version, key, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)

    args = parser.parse_args()

    promote_candidate(args.version, args.test_size)


if __name__ == "__main__":
    main()
