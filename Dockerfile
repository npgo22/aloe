# -- Build stage: compile Rust extension with maturin --
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install maturin
RUN pip install --no-cache-dir maturin

# Copy Rust source first (better layer caching)
COPY rust/ rust/
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# Build wheel
RUN maturin build --release --out dist

# -- Runtime stage --
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /app/dist/*.whl /tmp/

# Install the built wheel (pulls in all dependencies)
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

EXPOSE 8080

ENTRYPOINT ["python", "-m", "aloe", "gui"]
