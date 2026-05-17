# Dockerfile — Customer Churn Prediction MLOps
# Author: Suresh D R | DV Analytics

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY monitoring/ ./monitoring/
COPY tests/ ./tests/

RUN mkdir -p models data/reference data/current reports

ENV AWS_REGION=ap-south-1
ENV S3_BUCKET=customer-churn-project-2024
ENV PYTHONUNBUFFERED=1

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
