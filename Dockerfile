# Build stage
FROM rust:1.85-alpine AS builder

# Install build dependencies for tray icon and other system libraries
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    libxcb-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libayatana-appindicator-dev \
    gtk+3.0-dev \
    gdk-pixbuf-dev \
    pango-dev \
    atk-dev \
    fontconfig \
    fontconfig-dev \
    make \
    gcc

WORKDIR /app

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./

# Create dummy main.rs for dependency compilation
RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Build dependencies only (this caches the dependency compilation)
RUN cargo build --release && rm -rf target/release/deps/lumina* src/main.rs

# Copy actual source code
COPY src ./src

# Build the application
RUN touch src/main.rs && cargo build --release

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

# Create directories for logs and stats
RUN mkdir -p /app/logs && chown -R nonroot:nonroot /app

# Switch to non-root user
USER nonroot

EXPOSE 8080

ENTRYPOINT ["/app/lumina"]
CMD ["/app/config.yaml"]
