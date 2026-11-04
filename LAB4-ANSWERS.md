# Lab 4 — Docker Compose

## Implementation

We orchestrated three services using Docker Compose:

- **MLflow:** Tracking server and model registry, running on port 5000.
- **Inference:** FastAPI service loading the `food11@champion` registered model.
- **Frontend:** Streamlit application, accessible on port 8501.

We migrated our existing MLflow database and model artifacts from Windows paths to Linux container paths. We used a bind mount to persist MLflow data and make the saved models accessible to the inference service.

## Q1 — What happens without a volume?

Without persistent storage, files written to the container's writable filesystem disappear when the container is removed. Starting a new container from the same image creates a fresh filesystem.

## Q2 — Why use a named volume instead of a bind mount?

Named volumes are managed by Docker and are generally easier to manage for container data. Bind mounts connect a specific host directory to a container.

For this lab, we used a bind mount to preserve our migrated MLflow database and artifacts.

## Q3 — Why does `http://mlflow:5000` work?

Docker Compose creates a network and provides DNS-based service discovery. The hostname `mlflow` resolves to the MLflow container.

## Q4 — Why use an environment variable for the inference URL?

It allows the frontend to communicate with different inference servers without modifying its source code.

## Q5 — Why doesn't inference publish a port?

The frontend accesses inference through Docker Compose's internal network using `http://inference:8000`. A host port mapping isn't required for communication between containers.

## Q6 — What if MLflow isn't ready?

`depends_on` controls startup order but does not guarantee readiness. If inference attempts to load the model before MLflow is available, it may fail during startup.

## Q7 — Which services publish ports?

MLflow publishes port 5000, and Streamlit publishes port 8501. Inference listens on port 8000 internally without publishing it to the host.

## Q8 — Does changing the model alias immediately update inference?

No. Our inference service loads its model at startup. Changing the `champion` alias alone does not replace the model already loaded in memory.

## Q9 — Why does restarting inference work without rebuilding?

The inference image contains the serving application and its dependencies, while the model is selected from MLflow at startup. Restarting the container reloads the model selected by the registry alias without rebuilding the image.

We tested versions 1 and 3 and successfully loaded both model artifacts.

## Q10 — What happens to the models after Docker Compose down/up?

Our MLflow database, artifacts, and `champion` alias remained available after removing and recreating the containers.

We confirmed that the champion was still version 3.

We used a bind mount rather than a named volume. Therefore, `docker compose down -v` would remove Docker-managed named volumes but would not delete the files in our bind-mounted `mlflow-migration` directory.

## Q11 — What are Docker Compose's limitations?

Docker Compose is primarily intended for managing containers on a single Docker host. For multi-node scheduling, automated recovery from machine failure, and production-grade scaling, we would generally use an orchestrator such as Kubernetes.
