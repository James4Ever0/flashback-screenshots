#!/bin/bash
# Build script for android-go-bootstrap-su
# Cross-compiles a static Go binary for Android targets.

set -e

MODULE="android-go-bootstrap-su"
BINARY="screenshot-worker"
OUTPUT_DIR="build"

# Ensure dependencies are resolved
echo "==> Running go mod tidy..."
go mod tidy

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build flags for fully static binary
LDFLAGS="-s -w"

# ---------------------------------------------------------------------------
# Android ARM64 (most common modern devices)
# ---------------------------------------------------------------------------
echo "==> Building for Android ARM64..."
GOOS=android GOARCH=arm64 CGO_ENABLED=0 \
    go build -ldflags="$LDFLAGS" \
    -o "$OUTPUT_DIR/${BINARY}-android-arm64" \
    ./cmd/screenshot-worker

echo "    -> $OUTPUT_DIR/${BINARY}-android-arm64"

# NOTE: Android ARMv7 requires CGO (external linker). Skipped — use ARM64
# for all modern devices (Android 5.0+). Uncomment below with NDK if needed.
# echo "==> Building for Android ARMv7..."
# CC=arm-linux-androideabi-gcc GOOS=android GOARCH=arm GOARM=7 CGO_ENABLED=1 \
#     go build -ldflags="$LDFLAGS" \
#     -o "$OUTPUT_DIR/${BINARY}-android-arm" \
#     ./cmd/screenshot-worker

# NOTE: Android x86_64 also requires CGO. Build with NDK if needed.
# echo "==> Building for Android x86_64..."
# CC=x86_64-linux-android-gcc GOOS=android GOARCH=amd64 CGO_ENABLED=1 \
#     go build -ldflags="$LDFLAGS" \
#     -o "$OUTPUT_DIR/${BINARY}-android-amd64" \
#     ./cmd/screenshot-worker

# ---------------------------------------------------------------------------
# Linux ARM64 (fallback for Termux / root shell)
# ---------------------------------------------------------------------------
echo "==> Building for Linux ARM64..."
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 \
    go build -ldflags="$LDFLAGS" \
    -o "$OUTPUT_DIR/${BINARY}-linux-arm64" \
    ./cmd/screenshot-worker

echo "    -> $OUTPUT_DIR/${BINARY}-linux-arm64"

# ---------------------------------------------------------------------------
# Linux AMD64 (for local testing / emulator)
# ---------------------------------------------------------------------------
echo "==> Building for Linux AMD64..."
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 \
    go build -ldflags="$LDFLAGS" \
    -o "$OUTPUT_DIR/${BINARY}-linux-amd64" \
    ./cmd/screenshot-worker

echo "    -> $OUTPUT_DIR/${BINARY}-linux-amd64"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==> Build complete! Binaries in $OUTPUT_DIR/:"
ls -lh "$OUTPUT_DIR"

echo ""
echo "==> Deployment example (ARM64 device):"
echo "    adb push $OUTPUT_DIR/${BINARY}-android-arm64 /data/local/tmp/${BINARY}"
echo "    adb shell su -c \"chmod +x /data/local/tmp/${BINARY}\""
echo "    adb shell su -c \"/data/local/tmp/${BINARY} daemon -c /data/local/tmp/config.toml\""
echo ""
echo "==> Or run via the Kotlin app (Runtime.exec):"
echo "    ProcessBuilder(\"su\", \"-c\", \"/data/local/tmp/${BINARY} daemon\")"
echo ""
echo "==> Note: Android x86_64 / ARMv7 require CGO + Android NDK."
echo "    Use GOOS=linux GOARCH=amd64 or arm64 for Termux/root-shell fallback."
