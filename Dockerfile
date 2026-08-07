# Bitcoin Analyser — Railway-ready image.
# python:3.12-slim, and deliberately WITHOUT torch/transformers, keeps the
# image and RAM footprint small enough for the $5 Railway plan. FinBERT
# (torch/transformers) is local-only (requirements-local.txt); macro/sentiment.py
# lazy-loads it and falls back to neutral sentiment when absent.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

COPY requirements.txt .

# TA-Lib needs a system C library that slim lacks — indicators/technical.py
# falls back to pure-pandas implementations, so strip it from the install.
RUN grep -vi '^TA-Lib' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

# Railway injects PORT; 8050 is the local default (see autonomous.py).
EXPOSE 8050

CMD ["python", "autonomous.py"]
