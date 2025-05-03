# Python 3.9 slim sürümünü temel imaj olarak kullan
FROM python:3.9-slim

LABEL authors="namnam"

WORKDIR /app

COPY pyproject.toml .

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    pip install --upgrade pip && \
    pip install . && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY main.py .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
