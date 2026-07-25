# Bitcoin Analyser — Railway-ready image.
# python:3.12-slim + CPU-only torch keeps the image and RAM footprint small
# enough for the $5 Railway plan (set SKIP_FINBERT=1 to avoid loading the
# ~1.5 GB FinBERT/torch weights entirely).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

COPY requirements.txt .

# 1. CPU-only torch first (the CUDA wheel is ~5x bigger and useless here).
# 2. TA-Lib needs a system C library that slim lacks — indicators/technical.py
#    falls back to pure-pandas implementations, so strip it from the install.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -vi '^TA-Lib' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

# Railway injects PORT; 8050 is the local default (see autonomous.py).
EXPOSE 8050

CMD ["python", "autonomous.py"]
