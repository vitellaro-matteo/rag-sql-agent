FROM python:3.11-slim AS base

WORKDIR /app

# System deps for FAISS
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]"

COPY . .

# Seed database and build FAISS index at build time
RUN python -m scripts.seed_db && \
    python -m scripts.build_index

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
