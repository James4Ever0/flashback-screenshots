#!/system/bin/sh
# Run this after test-network-detection.sh to extract a concise summary.
# Usage: su -c "sh /data/local/tmp/grep-network-summary.sh"

SRC="/data/local/tmp/network-test-results.txt"
DST="/data/local/tmp/network-summary.txt"

if [ ! -f "$SRC" ]; then
    echo "Source file not found: $SRC"
    echo "Run test-network-detection.sh first."
    exit 1
fi

rm -f "$DST"

echo "========================================" >> "$DST"
echo "  Network Detection Summary"            >> "$DST"
echo "  $(date)"                               >> "$DST"
echo "========================================" >> "$DST"

# --- Device info ---
echo "" >> "$DST"
echo "[Device]" >> "$DST"
grep -E "Device:|Android:" "$SRC" >> "$DST"

# --- Strategy A: Routing table ---
echo "" >> "$DST"
echo "[Strategy A] Kernel Routing Table" >> "$DST"
grep -A3 "\[ip route get 8\.8\.8\.8\]" "$SRC" | grep -v "^---$" | grep -v "^Command:" >> "$DST"
echo "" >> "$DST"
grep "Egress interface:" "$SRC" >> "$DST"
grep "Inferred:" "$SRC" | head -1 >> "$DST"

echo "" >> "$DST"
echo "[/proc/net/route default iface]" >> "$DST"
grep "Default iface:" "$SRC" >> "$DST"

# --- Strategy B: dumpsys connectivity ---
echo "" >> "$DST"
echo "[Strategy B] dumpsys connectivity" >> "$DST"
grep "Found 'Active default network'" "$SRC" >> "$DST"
grep "Transport:" "$SRC" | head -1 >> "$DST"
grep "mDefaultRequest context:" "$SRC" >> "$DST"

# Extract all TRANSPORT lines (first 20 only)
echo "" >> "$DST"
echo "[All TRANSPORT_ flags found]" >> "$DST"
grep -oE "TRANSPORT_[A-Z]+" "$SRC" | sort -u | head -20 >> "$DST"

# Metered status
echo "" >> "$DST"
echo "[Metered status]" >> "$DST"
grep -iE "metered|NOT_METERED" "$SRC" | head -5 >> "$DST"

# --- Strategy C: WiFi / Cellular ---
echo "" >> "$DST"
echo "[Strategy C] WiFi State" >> "$DST"
grep "curState=" "$SRC" | head -1 >> "$DST"
grep "SSID:" "$SRC" | head -1 >> "$DST"

echo "" >> "$DST"
echo "[Strategy C] Cellular State" >> "$DST"
grep "Data reg state:" "$SRC" >> "$DST"
grep "Data network type:" "$SRC" >> "$DST"

# --- Strategy D: getprop ---
echo "" >> "$DST"
echo "[Strategy D] getprop" >> "$DST"
grep "dhcp\." "$SRC" | grep -v "(empty)" | head -10 >> "$DST"
grep "gsm\." "$SRC" | head -5 >> "$DST"
grep "net\.dns" "$SRC" | head -3 >> "$DST"

# --- Interface list ---
echo "" >> "$DST"
echo "[Interfaces with IPs]" >> "$DST"
grep -E "^  (wlan|rmnet|eth|ccmni|usb|tun|ppp|bond|wl)[^:]*:" "$SRC" >> "$DST"

# --- Cross-check summary ---
echo "" >> "$DST"
echo "[Cross-Check Summary]" >> "$DST"
grep "Routing table says:" "$SRC" >> "$DST"
grep "/proc/net/route says:" "$SRC" >> "$DST"
grep "dumpsys wifi says:" "$SRC" >> "$DST"
grep "dumpsys telephony:" "$SRC" >> "$DST"
grep "Recommended network type" "$SRC" >> "$DST"

echo "" >> "$DST"
echo "========================================" >> "$DST"
echo "Full output: $SRC" >> "$DST"
echo "Summary saved to: $DST" >> "$DST"

echo "Summary written to: $DST"
echo "Size: $(wc -c < "$DST") bytes (source was $(wc -c < "$SRC") bytes)"
