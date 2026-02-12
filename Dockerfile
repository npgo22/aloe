# Build stage
FROM rust:1.80-slim AS builder

WORKDIR /app

# Copy source
COPY . .

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install ca-certificates for HTTPS if needed
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /app/target/release/aloe /usr/local/bin/aloe

# Expose port for GUI
EXPOSE 8080

# Default to GUI
CMD ["aloe", "gui"]