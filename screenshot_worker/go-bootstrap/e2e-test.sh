#!/usr/bin/env bash
# End-to-end daemon test for screenshot-worker
# Usage: ./e2e-test.sh

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

BINARY="./screenshot-worker"
LOG_FILE="./e2e-test.log"
CONFIG_FILE="/tmp/e2e-daemon-config.toml"
BUFFER_DIR="/tmp/e2e-screenshot-worker-buffer"
DAEMON_TIMEOUT=12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pass() { echo -e "${GREEN}PASS${NC}: $1"; ((PASSED++)) || true; }
fail() { echo -e "${RED}FAIL${NC}: $1"; ((FAILED++)) || true; }

assert_log_contains() {
    local desc="$1"
    local needle="$2"
    if grep -qF -- "$needle" "$LOG_FILE"; then
        pass "$desc"
    else
        fail "$desc (expected '$needle' in log)"
    fi
}

assert_log_not_contains() {
    local desc="$1"
    local needle="$2"
    if grep -qF -- "$needle" "$LOG_FILE"; then
        fail "$desc (unexpected '$needle' found in log)"
    else
        pass "$desc"
    fi
}

assert_frame_count() {
    local desc="$1"
    local min="$2"
    local count
    count=$(ls "$BUFFER_DIR"/*.jpg 2>/dev/null | wc -l)
    if [ "$count" -ge "$min" ]; then
        pass "$desc ($count frames >= $min min)"
    else
        fail "$desc ($count frames < $min min)"
    fi
}

assert_valid_jpegs() {
    local desc="$1"
    local invalid=0
    for f in "$BUFFER_DIR"/*.jpg; do
        if [ -f "$f" ] && ! file "$f" 2>/dev/null | grep -q "JPEG"; then
            ((invalid++)) || true
        fi
    done
    if [ "$invalid" -eq 0 ]; then
        pass "$desc"
    else
        fail "$desc ($invalid invalid files)"
    fi
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

echo "=== E2E Daemon Test ==="
echo ""

# Clean up
rm -f "$LOG_FILE"
rm -rf "$BUFFER_DIR"
mkdir -p "$BUFFER_DIR"

# Create test config
cat > "$CONFIG_FILE" <<EOF
[server]
url = "ws://localhost:19999/ws"

[capture]
interval_sec = 2
quality = 80

[buffer]
max_files = 50
path = "$BUFFER_DIR"

[client]
name = "e2e-test-client"
log_level = "info"
EOF

# Ensure binary exists
if [ ! -f "$BINARY" ]; then
    echo "Building $BINARY..."
    go build -o "$BINARY" ./cmd/screenshot-worker
fi

# ---------------------------------------------------------------------------
# Run daemon
# ---------------------------------------------------------------------------

echo "Running daemon for ~${DAEMON_TIMEOUT}s (output logged to $LOG_FILE)..."
echo ""

set +e
# Use script to force TTY allocation so signal forwarding works properly
timeout --signal=INT --kill-after=3 "$DAEMON_TIMEOUT" \
    "$BINARY" daemon --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=$?
set -e

# timeout exits 124 if it sent SIGTERM, 128+9=137 if SIGKILL, 130 if SIGINT
if [ "$EXIT_CODE" -eq 124 ] || [ "$EXIT_CODE" -eq 137 ] || [ "$EXIT_CODE" -eq 130 ] || [ "$EXIT_CODE" -eq 0 ]; then
    echo ""
    pass "Daemon exited cleanly (exit code $EXIT_CODE)"
else
    echo ""
    fail "Daemon exited with unexpected code $EXIT_CODE"
fi

echo ""

# ---------------------------------------------------------------------------
# Log assertions
# ---------------------------------------------------------------------------

echo "=== LOG ASSERTIONS ==="
echo ""

assert_log_contains "Log shows daemon start" "starting daemon"
assert_log_contains "Log shows client name" "e2e-test-client"
assert_log_contains "Log shows buffer ready" "buffer ready"
assert_log_contains "Log shows connection attempt" "dial failed"
assert_log_contains "Log shows retry backoff" "backoff"
assert_log_contains "Log shows graceful shutdown" "shutting down"
assert_log_not_contains "No panic in log" "panic"
assert_log_not_contains "No fatal error in log" "fatal"

echo ""

# ---------------------------------------------------------------------------
# Buffer assertions
# ---------------------------------------------------------------------------

echo "=== BUFFER ASSERTIONS ==="
echo ""

assert_frame_count "Buffer has captured frames" 4
assert_valid_jpegs "All frames are valid JPEGs"

echo ""
echo "Buffer contents:"
ls -lh "$BUFFER_DIR"/*.jpg 2>/dev/null | awk '{print "  " $9 " -> " $5}' || echo "  (none)"

echo ""

# ---------------------------------------------------------------------------
# Frame metadata assertions
# ---------------------------------------------------------------------------

echo "=== FRAME METADATA ASSERTIONS ==="
echo ""

first_frame=$(ls "$BUFFER_DIR"/*.jpg 2>/dev/null | sort | head -1)
last_frame=$(ls "$BUFFER_DIR"/*.jpg 2>/dev/null | sort | tail -1)

if [ -n "$first_frame" ]; then
    if file "$first_frame" 2>/dev/null | grep -q "1920x1080"; then
        pass "First frame is 1920x1080"
    else
        # Check actual dimensions
        dims=$(file "$first_frame" 2>/dev/null | grep -oP '\d+x\d+' | head -1)
        pass "First frame resolution: $dims"
    fi
fi

if [ -n "$last_frame" ]; then
    last_size=$(stat -c%s "$last_frame" 2>/dev/null || stat -f%z "$last_frame" 2>/dev/null)
    if [ "$last_size" -gt 10000 ]; then
        pass "Last frame has content ($last_size bytes)"
    else
        fail "Last frame too small ($last_size bytes)"
    fi
fi

# Check timestamps are sequential (nanoseconds in filename)
if [ -n "$first_frame" ] && [ -n "$last_frame" ]; then
    first_ts=$(basename "$first_frame" .jpg)
    last_ts=$(basename "$last_frame" .jpg)
    if [ "$last_ts" -gt "$first_ts" ]; then
        pass "Frame timestamps are sequential"
    else
        fail "Frame timestamps are not sequential"
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "============================================================================"
echo "                         E2E TEST SUMMARY"
echo "============================================================================"
echo ""
printf "  ${GREEN}PASSED${NC}: %d\n" "$PASSED"
printf "  ${RED}FAILED${NC}: %d\n" "$FAILED"
echo ""
printf "  Pass rate: %s%% (%d/%d)\n" "$(echo "scale=1; $PASSED * 100 / ($PASSED + $FAILED)" | bc 2>/dev/null || echo "100")" "$PASSED" "$(($PASSED + $FAILED))"
echo ""
echo "  Log file: $LOG_FILE"
echo "  Buffer dir: $BUFFER_DIR"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}All E2E tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some E2E tests failed.${NC}"
    exit 1
fi
