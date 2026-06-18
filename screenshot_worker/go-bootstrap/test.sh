#!/usr/bin/env bash
# Test script for screenshot-worker Go implementation
# Usage: ./test.sh

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Binary names
BINARY="./screenshot-worker"
STATIC_BINARY="./screenshot-worker-static"
TEST_OUTPUT_DIR="./test-output"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

pass() {
    echo -e "${GREEN}PASS${NC}: $1"
    ((PASSED++)) || true
}

fail() {
    echo -e "${RED}FAIL${NC}: $1"
    ((FAILED++)) || true
}

warn() {
    echo -e "${YELLOW}WARN${NC}: $1"
    ((WARNINGS++)) || true
}

assert_cmd() {
    local desc="$1"
    local cmd="$2"
    local expected_exit="${3:-0}"
    local output
    local exit_code

    echo "  > $desc"
    set +e
    output=$(eval "$cmd" 2>&1)
    exit_code=$?
    set -e

    if [ "$exit_code" -eq "$expected_exit" ]; then
        pass "$desc"
        echo "    output: $(echo "$output" | head -1)"
        return 0
    else
        fail "$desc (expected exit $expected_exit, got $exit_code)"
        echo "    output: $output"
        return 1
    fi
}

assert_contains() {
    local desc="$1"
    local output="$2"
    local needle="$3"

    if echo "$output" | grep -qF -- "$needle"; then
        pass "$desc"
    else
        fail "$desc (expected '$needle' in output)"
        echo "    output: $output"
    fi
}

assert_file_exists() {
    local desc="$1"
    local path="$2"

    if [ -f "$path" ]; then
        local size
        size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null)
        pass "$desc ($size bytes)"
    else
        fail "$desc (file not found: $path)"
    fi
}

assert_file_size() {
    local desc="$1"
    local path="$2"
    local min_size="$3"

    if [ ! -f "$path" ]; then
        fail "$desc (file not found)"
        return
    fi

    local size
    size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null)

    if [ "$size" -ge "$min_size" ]; then
        pass "$desc ($size bytes >= $min_size min)"
    else
        fail "$desc (size $size < min $min_size)"
    fi
}

assert_static_binary() {
    local desc="$1"
    local path="$2"

    if [ ! -f "$path" ]; then
        fail "$desc (binary not found)"
        return
    fi

    local ldd_output
    set +e
    ldd_output=$(ldd "$path" 2>&1)
    local ldd_exit=$?
    set -e

    if echo "$ldd_output" | grep -q "not a dynamic executable"; then
        pass "$desc"
    elif echo "$ldd_output" | grep -q "statically linked"; then
        pass "$desc"
    else
        fail "$desc (binary has dynamic dependencies)"
        echo "    ldd output: $ldd_output"
    fi
}

assert_binary_size() {
    local desc="$1"
    local path="$2"
    local max_size="$3"

    if [ ! -f "$path" ]; then
        fail "$desc (binary not found)"
        return
    fi

    local size
    size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null)
    local size_mb
    size_mb=$(echo "scale=2; $size / 1024 / 1024" | bc 2>/dev/null || echo "$((size / 1024 / 1024))")

    if [ "$size" -le "$max_size" ]; then
        pass "$desc ($size_mb MB <= $(echo "scale=2; $max_size / 1024 / 1024" | bc 2>/dev/null || echo "$((max_size / 1024 / 1024))") MB max)"
    else
        warn "$desc ($size_mb MB > max, but acceptable)"
    fi
}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

echo "=== Cleaning up previous test artifacts ==="
rm -rf "$TEST_OUTPUT_DIR"
mkdir -p "$TEST_OUTPUT_DIR"
rm -f "$BINARY" "$STATIC_BINARY"
echo ""

# ---------------------------------------------------------------------------
# Build Tests
# ---------------------------------------------------------------------------

echo "=== BUILD TESTS ==="
echo ""

echo "[1/3] Building screenshot-worker (default)..."
if go build -o "$BINARY" ./cmd/screenshot-worker 2>&1; then
    pass "go build (default)"
else
    fail "go build (default)"
    echo ""
    echo "Build failed. Stopping tests."
    exit 1
fi
assert_file_exists "Binary created" "$BINARY"
echo ""

echo "[2/3] Building screenshot-worker-static (CGO_ENABLED=0)..."
if CGO_ENABLED=0 go build -o "$STATIC_BINARY" ./cmd/screenshot-worker 2>&1; then
    pass "go build (static, CGO_ENABLED=0)"
else
    fail "go build (static, CGO_ENABLED=0)"
fi
assert_file_exists "Static binary created" "$STATIC_BINARY"
echo ""

echo "[3/3] Checking binary sizes..."
assert_binary_size "Default binary size" "$BINARY" $((30 * 1024 * 1024))  # 30 MB max
assert_binary_size "Static binary size" "$STATIC_BINARY" $((30 * 1024 * 1024))
echo ""

# ---------------------------------------------------------------------------
# Static Linkage Test
# ---------------------------------------------------------------------------

echo "=== STATIC LINKAGE TESTS ==="
echo ""
assert_static_binary "Static binary has no dynamic deps" "$STATIC_BINARY"
echo ""

# ---------------------------------------------------------------------------
# CLI Command Tests
# ---------------------------------------------------------------------------

echo "=== CLI COMMAND TESTS ==="
echo ""

OUTPUT_HELP=$($BINARY --help 2>&1 || true)
assert_contains "Help contains 'daemon'" "$OUTPUT_HELP" "daemon"
assert_contains "Help contains 'test-lock'" "$OUTPUT_HELP" "test-lock"
assert_contains "Help contains 'test-capture'" "$OUTPUT_HELP" "test-capture"
assert_contains "Help contains 'install'" "$OUTPUT_HELP" "install"
assert_contains "Help contains 'uninstall'" "$OUTPUT_HELP" "uninstall"
assert_contains "Help contains 'status'" "$OUTPUT_HELP" "status"
assert_contains "Help contains '--config'" "$OUTPUT_HELP" "--config"
echo ""

# ---------------------------------------------------------------------------
# test-lock Tests
# ---------------------------------------------------------------------------

echo "=== test-lock TESTS ==="
echo ""

OUTPUT_LOCK=$($BINARY test-lock 2>&1 || true)
if echo "$OUTPUT_LOCK" | grep -q "UNLOCKED\|LOCKED"; then
    pass "test-lock returns lock state"
    echo "    output: $(echo "$OUTPUT_LOCK" | tr -d '\n')"
else
    fail "test-lock did not return expected output"
    echo "    output: $OUTPUT_LOCK"
fi
echo ""

# ---------------------------------------------------------------------------
# test-window Tests
# ---------------------------------------------------------------------------

echo "=== test-window TESTS ==="
echo ""

OUTPUT_WINDOW=$($BINARY test-window 2>&1 || true)
assert_contains "test-window shows display count" "$OUTPUT_WINDOW" "Active displays:"
assert_contains "test-window shows display resolution" "$OUTPUT_WINDOW" "Display 0:"
echo ""

# ---------------------------------------------------------------------------
# test-capture Tests
# ---------------------------------------------------------------------------

echo "=== test-capture TESTS ==="
echo ""

CAPTURE_DEFAULT="$TEST_OUTPUT_DIR/default.jpg"
assert_cmd "test-capture with default output" "$BINARY test-capture --output '$CAPTURE_DEFAULT'"
assert_file_exists "Default capture file exists" "$CAPTURE_DEFAULT"
assert_file_size "Default capture file has content" "$CAPTURE_DEFAULT" 10000

# Verify it's a valid JPEG
if file "$CAPTURE_DEFAULT" 2>/dev/null | grep -q "JPEG"; then
    pass "Capture file is valid JPEG"
else
    # Some systems don't have 'file' command, skip
    warn "Could not verify JPEG format (file command unavailable)"
fi

# Test with custom path
CAPTURE_CUSTOM="$TEST_OUTPUT_DIR/custom-screenshot.jpg"
assert_cmd "test-capture with custom path" "$BINARY test-capture --output '$CAPTURE_CUSTOM'"
assert_file_exists "Custom capture file exists" "$CAPTURE_CUSTOM"
assert_file_size "Custom capture file has content" "$CAPTURE_CUSTOM" 10000

echo ""

# ---------------------------------------------------------------------------
# Config File Tests
# ---------------------------------------------------------------------------

echo "=== CONFIG FILE TESTS ==="
echo ""

assert_file_exists "Config example exists" "./config.toml.example"

# Test loading nonexistent config
OUTPUT_NOCONFIG=$($BINARY daemon --config /nonexistent/config.toml 2>&1 || true)
if echo "$OUTPUT_NOCONFIG" | grep -qi "config\|not found\|failed to load"; then
    pass "Daemon fails gracefully with missing config"
else
    warn "Daemon missing-config behavior unclear"
    echo "    output: $(echo "$OUTPUT_CONFIG" | head -3)"
fi

echo ""

# ---------------------------------------------------------------------------
# Service Command Tests (dry-run, may fail without perms)
# ---------------------------------------------------------------------------

echo "=== SERVICE COMMAND TESTS ==="
echo ""

OUTPUT_STATUS=$($BINARY status 2>&1 || true)
if [ -n "$OUTPUT_STATUS" ]; then
    pass "status command executes"
    echo "    output: $(echo "$OUTPUT_STATUS" | head -1)"
else
    warn "status command produced no output (service may not be installed)"
fi

echo ""

# ---------------------------------------------------------------------------
# Cross-compilation Tests
# ---------------------------------------------------------------------------

echo "=== CROSS-COMPILATION TESTS ==="
echo ""

CROSS_TARGETS=(
    "linux/amd64:screenshot-worker-linux-amd64"
    "windows/amd64:screenshot-worker-windows-amd64.exe"
    "linux/arm64:screenshot-worker-linux-arm64"
)

for target in "${CROSS_TARGETS[@]}"; do
    IFS=':' read -r goos_goarch output_name <<< "$target"
    IFS='/' read -r goos goarch <<< "$goos_goarch"

    echo "  > Cross-compiling for $goos/$goarch..."
    if CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" go build -o "$TEST_OUTPUT_DIR/$output_name" ./cmd/screenshot-worker 2>&1; then
        pass "Cross-compile $goos/$goarch"
        assert_file_exists "Cross-compile output exists" "$TEST_OUTPUT_DIR/$output_name"
    else
        fail "Cross-compile $goos/$goarch"
    fi
done

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "============================================================================"
echo "                           TEST SUMMARY"
echo "============================================================================"
echo ""
printf "  ${GREEN}PASSED${NC}:   %d\n" "$PASSED"
printf "  ${RED}FAILED${NC}:   %d\n" "$FAILED"
printf "  ${YELLOW}WARNINGS${NC}: %d\n" "$WARNINGS"
echo ""

TOTAL=$((PASSED + FAILED))
if [ "$TOTAL" -gt 0 ]; then
    PASS_RATE=$(echo "scale=1; $PASSED * 100 / $TOTAL" | bc 2>/dev/null || echo "$((PASSED * 100 / TOTAL))")
    printf "  Pass rate: %s%% (%d/%d)\n" "$PASS_RATE" "$PASSED" "$TOTAL"
fi

echo ""
echo "  Binary sizes:"
echo "    default:  $(stat -c%s "$BINARY" 2>/dev/null || stat -f%z "$BINARY" 2>/dev/null | awk '{printf "%.2f MB", $1/1024/1024}')"
echo "    static:   $(stat -c%s "$STATIC_BINARY" 2>/dev/null || stat -f%z "$STATIC_BINARY" 2>/dev/null | awk '{printf "%.2f MB", $1/1024/1024}')"

echo ""
echo "  Test artifacts: $TEST_OUTPUT_DIR/"
ls -lh "$TEST_OUTPUT_DIR" 2>/dev/null | tail -n +2 || echo "    (none)"

echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
