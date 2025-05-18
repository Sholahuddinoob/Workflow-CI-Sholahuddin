FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir pandas scikit-learn mlflow

CMD ["mlflow", "run", "."]
