# Lab 5 — CI/CD and Model Quality Gates

## 1. Automated Testing

Implemented automated tests using pytest and code-quality checks using Ruff.

The tests cover data processing, FastAPI endpoints, and MLflow model promotion logic.

GitHub Actions automatically executes testing and linting when code is pushed or a pull request is opened.

## 2. Data Validation

Implemented `validate_data.py` to check dataset structure, image dimensions and color mode, minimum sample counts, and duplicate images.

Initial validation identified duplicate files across dataset splits. After removing duplicates, validation passed:

- 3,314 checks passed
- 0 checks failed

The updated dataset was tracked and pushed using DVC.

## 3. GitHub Actions CI/CD

Configured a pipeline containing three jobs:

- **test:** Runs Ruff and pytest.
- **smoke:** Builds the Docker image and verifies the FastAPI health endpoint.
- **publish:** Publishes the Docker image to GitHub Container Registry when changes are pushed to `main`.

Successfully published the project image to GHCR.

## 4. Branch Protection

Configured a branch ruleset requiring CI checks before merging into `main`.

Tested the protection using an intentionally incorrect unit-test assertion.

**First attempt:** The test failed, and merging was blocked.

**Second attempt:** After fixing the assertion, all required checks passed, allowing the pull request to be merged.

This demonstrates how branch protection prevents failing changes from entering the main branch.

## 5. Model Promotion Quality Gate

Implemented `promote.py` to compare a candidate model against the existing champion using test accuracy and a statistical threshold.

The promotion condition is:

\[
Accuracy_{candidate} \ge Accuracy_{champion} + 2 \sqrt{\frac{p(1-p)}{n}}
\]

The models were evaluated on the same Food-11 mini evaluation dataset containing 1,071 images.

| Metric | Result |
|---|---|
| Champion version | 3 |
| Candidate version | 1 |
| Champion accuracy | 72.83% |
| Candidate accuracy | 72.83% |
| Promotion threshold | 75.55% |
| Decision | Rejected |
| Champion alias afterward | Version 3 |
| Challenger alias afterward | Version 1 |

The candidate was not promoted because it did not exceed the required accuracy threshold.

MLflow model-version tags recorded the promotion decision, candidate accuracy, champion accuracy, and threshold.

## 6. Reproducibility and Limitations

The evaluation was performed inside Docker because the registered MLflow artifacts use container-local filesystem paths.

MLflow reported differences between the Python and dependency versions used for model creation and evaluation. These warnings did not prevent execution but should be addressed in a reproducible production environment.

The evaluation also recorded metrics in the existing source runs. A production implementation should instead store evaluation results in dedicated evaluation runs with explicit dataset versioning.

## 7. Triggering Model Promotion Automatically

A promotion quality gate can be triggered after a successful model training and evaluation workflow.

Possible implementations include:

- A self-hosted GitHub Actions runner with access to the Docker-hosted MLflow registry and model artifacts.
- A hosted MLflow tracking and artifact service accessible to GitHub Actions runners.

A standard GitHub-hosted runner cannot directly access an MLflow instance running only on a developer's local machine.

## Conclusion

Lab 5 establishes two complementary quality gates:

1. A code quality gate using GitHub Actions and branch protection.
2. A model quality gate using MLflow test accuracy and statistical comparison.

Together, these support safer code integration and controlled model deployment.