// Package netcheck detects the active network type by querying the kernel
// routing table via "ip route get". This is the most reliable cross-version
// approach on Android because it reads the actual egress interface chosen by
// the OS for default traffic.
package netcheck

import (
	"strings"
	"sync"
	"time"

	"android-go-bootstrap-su/internal/root"
)

const cacheTTL = 5 * time.Second

type cached struct {
	netType   string
	isWifi    bool
	iface     string
	timestamp time.Time
}

var (
	mu     sync.Mutex
	last   cached
)

// Type returns the active network type: "WiFi", "Cellular", "Ethernet", "VPN", "None", or "Unknown".
func Type() string {
	return refresh().netType
}

// IsWiFi returns true if the active default route goes through a WiFi interface.
func IsWiFi() bool {
	return refresh().isWifi
}

// Interface returns the name of the egress interface (e.g. "wlan0", "ccmni1").
func Interface() string {
	return refresh().iface
}

func refresh() cached {
	mu.Lock()
	defer mu.Unlock()

	if time.Since(last.timestamp) < cacheTTL {
		return last
	}

	iface := detectIface()
	netType := classify(iface)
	isWifi := netType == "WiFi" || netType == "Ethernet"

	last = cached{
		netType:   netType,
		isWifi:    isWifi,
		iface:     iface,
		timestamp: time.Now(),
	}
	return last
}

// detectIface runs "ip route get 8.8.8.8" and extracts the egress interface.
func detectIface() string {
	out, err := root.RunSu("ip route get 8.8.8.8")
	if err != nil {
		return ""
	}
	line := string(out)

	// Parse: "8.8.8.8 via 10.x.x.x dev ccmni1 table ccmni1 src 10.x.x.x uid 0"
	// or:    "8.8.8.8 dev wlan0 table 1003 src 192.168.x.x uid 0"
	idx := strings.Index(line, "dev ")
	if idx == -1 {
		return ""
	}
	afterDev := line[idx+4:]
	fields := strings.Fields(afterDev)
	if len(fields) == 0 {
		return ""
	}
	return strings.TrimSpace(fields[0])
}

// classify maps an interface name to a network type.
func classify(iface string) string {
	switch {
	case iface == "":
		return "None"
	case strings.HasPrefix(iface, "wlan"):
		return "WiFi"
	case strings.HasPrefix(iface, "wl") && !strings.HasPrefix(iface, "wlan"):
		// e.g. wl0.1 — rare OEM variant, still WiFi
		return "WiFi"
	case iface == "bond0":
		// Samsung bonded interface (usually WiFi + cellular)
		return "WiFi"
	case strings.HasPrefix(iface, "rmnet") || strings.HasPrefix(iface, "ccmni"):
		return "Cellular"
	case strings.HasPrefix(iface, "eth"):
		return "Ethernet"
	case strings.HasPrefix(iface, "usb"):
		// USB tethering or ethernet
		return "Ethernet"
	case strings.HasPrefix(iface, "tun") || strings.HasPrefix(iface, "ppp"):
		return "VPN"
	default:
		return "Unknown"
	}
}
