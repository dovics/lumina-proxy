# Build stage
FROM rust:1.89-alpine AS builder

# Install minimal build dependencies (no tray/X11/GTK needed for Docker)
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    make \
    gcc \
    openssl-dev \
    perl

WORKDIR /app

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./

# Create dummy main.rs for dependency compilation
RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Build dependencies only (no-default-features = no tray, smaller build)
RUN cargo build --release --no-default-features && rm -rf target/release/deps/lumina* src/main.rs

# Copy actual source code
COPY src ./src

# Build the application without tray feature
RUN touch src/main.rs && cargo build --release --no-default-features

# Runtime stage - minimal image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/lumina /app/lumina

# Copy default config
COPY config.yaml /app/config.yaml

# Create non-root user and directories for logs/stats
RUN useradd -m nonroot && \
    mkdir -p /app/logs && \
    chown -R nonroot:nonroot /app

# Switch to non-root user
USER nonroot

EXPOSE 8080

ENTRYPOINT ["/app/lumina"]
CMD ["/app/config.yaml"]
