# Build stage
FROM rust:1.94-slim AS builder

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
#
# distroless/cc provides glibc, libstdc++, libgcc and ca-certificates -- exactly
# what this binary links against (verified with ldd: libstdc++.so.6, libm.so.6,
# libgcc_s.so.1, libc.so.6) -- and nothing else. No shell, no package manager.
#
# The previous debian:trixie-slim base carried the whole Debian userland, and
# that is where every CVE in this image came from: openssl, libssl3t64,
# bsdutils, libblkid1, libmount1, libsmartcols1 -- none of which the binary
# actually uses. Dropping the userland removes them at the source rather than
# chasing point releases.
#
# `cc` (not `static`) because the binary is dynamically linked; `nonroot` so it
# does not run as uid 0.
FROM gcr.io/distroless/cc-debian13:nonroot

COPY --from=builder /app/target/release/aloe /usr/local/bin/aloe

EXPOSE 8080
USER nonroot

# distroless has no shell, so the entrypoint must be an absolute path -- a bare
# "aloe" would rely on PATH resolution that does not exist here.
ENTRYPOINT ["/usr/local/bin/aloe"]
CMD ["gui"]
