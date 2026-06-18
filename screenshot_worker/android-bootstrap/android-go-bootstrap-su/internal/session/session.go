// Package session detects screen lock and display state via su + dumpsys.
package session

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"android-go-bootstrap-su/internal/root"
)

var (
	cacheLocked    bool
	cacheTime      time.Time
	cacheMu        sync.Mutex
	cacheDuration  = 1 * time.Second
)

// IsLocked checks if the screen is locked using dumpsys window/keyguard.
// Results are cached for 1 second to avoid spamming shell commands.
func IsLocked() (bool, error) {
	cacheMu.Lock()
	defer cacheMu.Unlock()

	if time.Since(cacheTime) < cacheDuration {
		return cacheLocked, nil
	}

	locked, err := checkLocked()
	if err != nil {
		return false, err
	}

	cacheLocked = locked
	cacheTime = time.Now()
	return locked, nil
}

// IsScreenOn checks if the screen is on using dumpsys power/display.
func IsScreenOn() (bool, error) {
	out, err := root.RunSu("dumpsys power")
	if err != nil {
		return false, fmt.Errorf("dumpsys power failed: %w", err)
	}
	output := string(out)

	// Android uses mWakefulness=Awake or mWakefulness=Asleep
	if strings.Contains(output, "mWakefulness=Awake") ||
		strings.Contains(output, "Wakefulness: Awake") ||
		strings.Contains(output, "Display Power: state=ON") {
		return true, nil
	}
	if strings.Contains(output, "mWakefulness=Asleep") ||
		strings.Contains(output, "Wakefulness: Asleep") ||
		strings.Contains(output, "Display Power: state=OFF") {
		return false, nil
	}

	// Default to true if we can't determine
	return true, nil
}

func checkLocked() (bool, error) {
	// Try dumpsys window for keyguard state
	out, err := root.RunSu("dumpsys window")
	if err == nil {
		output := string(out)
		// Look for keyguard indicators
		if strings.Contains(output, "mDreamingLockscreen=true") ||
			strings.Contains(output, "showing=true") && strings.Contains(output, "keyguard") {
			return true, nil
		}
		if strings.Contains(output, "mDreamingLockscreen=false") ||
			strings.Contains(output, "mInputRestricted=false") {
			return false, nil
		}
	}

	// Fallback: check device policy / user state
	out, err = root.RunSu("dumpsys device_policy")
	if err == nil {
		output := string(out)
		if strings.Contains(output, "Screen locked") {
			return true, nil
		}
	}

	// Final fallback: assume unlocked if we can't tell
	return false, nil
}
