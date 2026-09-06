# Build stage
FROM rust:1.98-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y curl ca-certificates gnupg cmake g++ make pkg-config && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . .

# Build frontend for embedding
RUN npm --prefix frontend ci
RUN npm --prefix frontend run build:embedded

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:trixie-slim

# Install ca-certificates for HTTPS if needed
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /app/target/release/aloe /usr/local/bin/aloe

# Expose port for GUI
EXPOSE 8080

# Default to GUI
CMD ["aloe", "gui"]
