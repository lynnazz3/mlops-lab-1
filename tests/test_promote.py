from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.food11 import promote


def make_client(champion_accuracy, candidate_accuracy):
    client = MagicMock()

    champion = SimpleNamespace(version="3", run_id="champion-run")
    candidate = SimpleNamespace(version="1", run_id="candidate-run")

    client.get_model_version.return_value = candidate
    client.get_model_version_by_alias.return_value = champion

    def get_run(run_id):
        accuracy = champion_accuracy if run_id == "champion-run" else candidate_accuracy
        return SimpleNamespace(
            data=SimpleNamespace(metrics={"test_accuracy": accuracy})
        )

    client.get_run.side_effect = get_run
    return client


def test_candidate_rejected(monkeypatch):
    client = make_client(0.7283, 0.7283)
    monkeypatch.setattr(promote, "MlflowClient", lambda **kwargs: client)

    promote.promote_candidate(version=1, test_size=1071)

    client.set_registered_model_alias.assert_called_once_with(
        "food11", "challenger", "1"
    )

    client.set_model_version_tag.assert_any_call(
        "food11", "1", "promotion_decision", "rejected"
    )


def test_candidate_promoted(monkeypatch):
    client = make_client(0.7283, 0.80)
    monkeypatch.setattr(promote, "MlflowClient", lambda **kwargs: client)

    promote.promote_candidate(version=1, test_size=1071)

    client.set_registered_model_alias.assert_called_once_with("food11", "champion", "1")

    client.set_model_version_tag.assert_any_call(
        "food11", "1", "promotion_decision", "promoted"
    )


def test_invalid_test_size():
    with pytest.raises(ValueError, match="Test size must be positive"):
        promote.promote_candidate(version=1, test_size=0)


def test_missing_accuracy():
    client = make_client(0.7283, 0.80)
    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={}))
    client.get_run.side_effect = None

    with pytest.raises(ValueError, match="no test_accuracy"):
        promote.get_accuracy(
            client,
            SimpleNamespace(run_id="missing-run"),
        )
