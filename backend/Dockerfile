# ============================================================
# Backend Dockerfile — Python 3.11 + TA-Lib from source
# ============================================================

FROM python:3.11-slim AS base

# ── System dependencies required for TA-Lib compilation ──────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Build TA-Lib C library from source ───────────────────────
RUN wget -q https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make -j$(nproc) -C src \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# ── Python dependencies ───────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# Install PyTorch CPU-only first (smaller image; GPU users can override)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir TA-Lib
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────
COPY . .

# Create data and models directories
RUN mkdir -p /app/data /app/models

# ── Runtime ───────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
