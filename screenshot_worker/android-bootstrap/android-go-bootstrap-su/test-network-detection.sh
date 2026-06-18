#!/system/bin/sh
# Network Detection Test Script for android-go-bootstrap-su
# Run as: su -c "sh /data/local/tmp/test-network-detection.sh"
# Output saved to: /data/local/tmp/network-test-results.txt

# shellcheck shell=sh

OUTFILE="/data/local/tmp/network-test-results.txt"
TMPDIR="/data/local/tmp"

# Clear previous output
rm -f "$OUTFILE"

echo "========================================"
echo "  Network Detection Test"
echo "  $(date)"
echo "  Device: $(getprop ro.product.model)"
echo "  Android: $(getprop ro.build.version.release) (API $(getprop ro.build.version.sdk))"
echo "========================================"
echo ""

# --- Helper: run a command via su if not already root ---
run_su() {
    if [ "$(id -u)" = "0" ]; then
        sh -c "$1" 2>&1
    else
        su -c "$1" 2>&1
    fi
}

# --- Helper: section header ---
section() {
    echo ""
    echo "----------------------------------------"
    echo "  $1"
    echo "----------------------------------------"
}

# --- Helper: command output block ---
cmd_block() {
    echo ""
    echo "[$1]"
    echo "Command: $2"
    echo "---"
    run_su "$2"
    echo "---"
    echo "Exit: $?"
}

# --- Helper: parse and report ---
report() {
    echo ""
    echo "[RESULT] $1"
}

# ==========================================
# 0. Root / Environment Check
# ==========================================
section "0. Environment"

echo "UID: $(id -u)"
echo "User: $(id -un 2>/dev/null || whoami 2>/dev/null || echo unknown)"
echo "Shell: $SHELL"
echo "PATH: $PATH"

cmd_block "which ip" "which ip"
cmd_block "which ifconfig" "which ifconfig"
cmd_block "ip --version" "ip --version 2>/dev/null || ip -V 2>/dev/null || echo 'ip version not supported'"

# ==========================================
# 1. Strategy A: Kernel Routing Table
# ==========================================
section "1. Strategy A — Kernel Routing Table (Primary)"

# 1a. ip route get
cmd_block "ip route get 8.8.8.8" "ip route get 8.8.8.8"
report "Default route via ip route get"
ROUTE_DEV=$(run_su "ip route get 8.8.8.8 2>/dev/null" | sed -n 's/.*dev \([^ ]*\).*/\1/p')
if [ -n "$ROUTE_DEV" ]; then
    echo "  Egress interface: $ROUTE_DEV"
    case "$ROUTE_DEV" in
        wlan*|wl*) echo "  Inferred: WiFi" ;;
        rmnet*|ccmni*|usb[0-9]|cellular*) echo "  Inferred: Cellular" ;;
        eth*|usb[0-9]*) echo "  Inferred: Ethernet" ;;
        tun*|ppp*) echo "  Inferred: VPN" ;;
        *) echo "  Inferred: Unknown ($ROUTE_DEV)" ;;
    esac
else
    echo "  Could not determine egress interface"
fi

# 1b. /proc/net/route
cmd_block "cat /proc/net/route" "cat /proc/net/route"
report "Default route via /proc/net/route"
DEFAULT_IFACE=$(run_su "cat /proc/net/route" | awk '$2 == "00000000" && $3 == "00000000" && $8 ~ /^0+$/ {print $1; exit}')
if [ -z "$DEFAULT_IFACE" ]; then
    DEFAULT_IFACE=$(run_su "cat /proc/net/route" | awk '$2 == "00000000" && $3 == "00000000" {print $1; exit}')
fi
if [ -n "$DEFAULT_IFACE" ]; then
    echo "  Default iface: $DEFAULT_IFACE"
else
    echo "  No default route found"
fi

# 1c. Full routing table for context
cmd_block "ip route show" "ip route show"

# ==========================================
# 2. Strategy B: Structured dumpsys connectivity
# ==========================================
section "2. Strategy B — dumpsys connectivity (Secondary)"

cmd_block "dumpsys connectivity" "dumpsys connectivity"

report "Structured parse of dumpsys connectivity"

# Try Android 12+ "Active default network"
ACTIVE_BLOCK=$(run_su "dumpsys connectivity" | awk '/Active default network:/{found=1} found{print; if(NR>found_line+25) exit}' found_line=0)
if [ -n "$ACTIVE_BLOCK" ]; then
    echo "  Found 'Active default network' block (Android 12+)"
    echo "$ACTIVE_BLOCK" | head -30
    TRANSPORT=$(echo "$ACTIVE_BLOCK" | grep -oE 'TRANSPORT_[A-Z]+' | head -1)
    if [ -n "$TRANSPORT" ]; then
        echo "  Transport: $TRANSPORT"
    fi
else
    echo "  'Active default network' block not found"
fi

# Try Android 10-11 mDefaultRequest
DEFAULT_REQ=$(run_su "dumpsys connectivity" | grep -A 30 "mDefaultRequest" | head -40)
if [ -n "$DEFAULT_REQ" ]; then
    echo ""
    echo "  mDefaultRequest context:"
    echo "$DEFAULT_REQ"
fi

# Extract all NetworkAgentInfo with their transport
echo ""
echo "  All NetworkAgentInfo transports:"
run_su "dumpsys connectivity" | grep -E 'NetworkAgentInfo|TRANSPORT_|mNetworkCapabilities' | head -50

# Check for metered/unmetered
echo ""
echo "  Metered status:"
run_su "dumpsys connectivity" | grep -iE 'metered|NOT_METERED' | head -10

# ==========================================
# 3. Strategy C: Service-specific dumpsys
# ==========================================
section "3. Strategy C — Service-specific dumpsys (Fallback)"

# WiFi state
cmd_block "dumpsys wifi" "dumpsys wifi"
report "WiFi state"
WIFI_CURSTATE=$(run_su "dumpsys wifi" | grep -oE 'curState=[A-Za-z]+' | head -1)
WIFI_SSID=$(run_su "dumpsys wifi" | grep -oE 'SSID: "[^"]+"' | head -1)
if [ -n "$WIFI_CURSTATE" ]; then
    echo "  $WIFI_CURSTATE"
    echo "  $WIFI_SSID"
else
    echo "  curState not found in dumpsys wifi"
fi

# Telephony / cellular
cmd_block "dumpsys telephony.registry" "dumpsys telephony.registry"
report "Cellular state"
CELL_REG=$(run_su "dumpsys telephony.registry" | grep -oE 'mDataRegState=[0-9]+' | head -1)
CELL_TYPE=$(run_su "dumpsys telephony.registry" | grep -oE 'mDataNetworkType=[A-Z0-9]+' | head -1)
if [ -n "$CELL_REG" ]; then
    echo "  Data reg state: $CELL_REG"
    echo "  Data network type: $CELL_TYPE"
else
    echo "  No telephony data found"
fi

# ==========================================
# 4. Strategy D: getprop Heuristic
# ==========================================
section "4. Strategy D — getprop (Last Resort)"

cmd_block "getprop dhcp" "getprop | grep dhcp"
report "DHCP properties"
for prop in $(run_su "getprop" | grep -oE 'dhcp\.[^\]]+' | sort -u); do
    val=$(run_su "getprop $prop" 2>/dev/null)
    if [ -n "$val" ]; then
        echo "  $prop = $val"
    fi
done

cmd_block "getprop gsm" "getprop | grep gsm"
report "GSM properties"
for prop in gsm.network.type gsm.sim.state gsm.operator.numeric; do
    val=$(run_su "getprop $prop" 2>/dev/null)
    echo "  $prop = ${val:-(empty)}"
done

cmd_block "getprop net" "getprop | grep 'net\.'"
report "Network properties"
for prop in net.dns1 net.dns2; do
    val=$(run_su "getprop $prop" 2>/dev/null)
    echo "  $prop = ${val:-(empty)}"
done

# ==========================================
# 5. Interface Inspection
# ==========================================
section "5. Interface Inspection"

cmd_block "ip link show" "ip link show"
report "Interface list"
run_su "ip link show" | grep '^[0-9]*:' | awk '{print $2}' | sed 's/:$//' | while read -r iface; do
    case "$iface" in
        lo) ;; # skip loopback
        *)
            ADDR=$(run_su "ip addr show dev $iface 2>/dev/null" | grep -oE 'inet [0-9.]+' | head -1)
            if [ -n "$ADDR" ]; then
                echo "  $iface: $ADDR"
            fi
            ;;
    esac
done

# /proc/net/dev counters
report "RX/TX counters ( /proc/net/dev )"
run_su "cat /proc/net/dev" | grep -E 'wlan|rmnet|eth|ccmni|usb' | while read -r line; do
    IFACE=$(echo "$line" | cut -d: -f1 | tr -d ' ')
    RX=$(echo "$line" | awk '{print $2}')
    TX=$(echo "$line" | awk '{print $10}')
    echo "  $IFACE: RX=$RX TX=$TX"
done

# ==========================================
# 6. Connectivity Probe
# ==========================================
section "6. Connectivity Probe"

report "TCP reachability to well-known hosts"
for host in "8.8.8.8" "1.1.1.1"; do
    # Use /dev/tcp if available (bash feature, may not work on sh)
    # Fall back to ping or nc
    if run_su "ping -c 1 -W 2 $host" >/dev/null 2>&1; then
        echo "  $host: REACHABLE (ping)"
    else
        echo "  $host: UNREACHABLE (ping)"
    fi
done

# Try nc (netcat) if available
if run_su "which nc" >/dev/null 2>&1; then
    echo ""
    echo "  Netcat available"
    for host_port in "8.8.8.8:53" "1.1.1.1:53"; do
        if run_su "nc -z -w 2 $(echo $host_port | cut -d: -f1) $(echo $host_port | cut -d: -f2)" >/dev/null 2>&1; then
            echo "  $host_port: OPEN"
        else
            echo "  $host_port: CLOSED/TIMEOUT"
        fi
    done
else
    echo "  nc not available"
fi

# ==========================================
# 7. Cross-Check Summary
# ==========================================
section "7. Cross-Check Summary"

echo ""
echo "  Routing table says:    ${ROUTE_DEV:-unknown}"
echo "  /proc/net/route says:  ${DEFAULT_IFACE:-unknown}"
echo "  dumpsys wifi says:     ${WIFI_CURSTATE:-unknown} / ${WIFI_SSID:-unknown}"
echo "  dumpsys telephony:     ${CELL_REG:-unknown} / ${CELL_TYPE:-unknown}"
echo ""
echo "  Recommended network type from routing:"
if [ -n "$ROUTE_DEV" ]; then
    case "$ROUTE_DEV" in
        wlan*|wl*) echo "    WiFi (from ip route)" ;;
        rmnet*|ccmni*) echo "    Cellular (from ip route)" ;;
        eth*) echo "    Ethernet (from ip route)" ;;
        tun*|ppp*) echo "    VPN (from ip route) — allow uploads" ;;
        *) echo "    Unknown ($ROUTE_DEV)" ;;
    esac
elif [ -n "$DEFAULT_IFACE" ]; then
    case "$DEFAULT_IFACE" in
        wlan*) echo "    WiFi (from /proc/net/route)" ;;
        rmnet*|ccmni*) echo "    Cellular (from /proc/net/route)" ;;
        eth*) echo "    Ethernet (from /proc/net/route)" ;;
        tun*|ppp*) echo "    VPN (from /proc/net/route)" ;;
        *) echo "    Unknown ($DEFAULT_IFACE)" ;;
    esac
else
    echo "    None (no default route)"
fi

# ==========================================
# Save output
# ==========================================
section "Done"
echo ""
echo "Output saved to: $OUTFILE"
echo "To collect: adb pull $OUTFILE"
