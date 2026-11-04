
FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow==3.1.4" "SQLAlchemy>=2.0,<2.1"

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--default-artifact-root", "/mlflow-data/mlruns"]
