// Package root handles superuser (su) detection and permission requests.
package root

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// CheckRoot runs "su -c echo test" and returns true if exit code == 0.
func CheckRoot() bool {
	cmd := exec.Command("su", "-c", "echo test")
	out, err := cmd.CombinedOutput()
	return err == nil && strings.TrimSpace(string(out)) == "test"
}

// RequestRoot prints a message to stderr asking the user to grant root
// via their superuser app (Magisk, KernelSU, etc.).
func RequestRoot() {
	fmt.Fprintln(os.Stderr, "Root access is required. Please grant root permission in your superuser app.")
}

// RequireRoot blocks until root is available or timeout is reached.
// If autoRequest is false, it exits immediately on missing root.
func RequireRoot(autoRequest bool, retrySec, maxRetrySec int) error {
	if CheckRoot() {
		return nil
	}

	if !autoRequest {
		return fmt.Errorf("root not available and auto_request is disabled")
	}

	RequestRoot()

	elapsed := 0
	for elapsed < maxRetrySec {
		time.Sleep(time.Duration(retrySec) * time.Second)
		elapsed += retrySec
		if CheckRoot() {
			return nil
		}
		fmt.Fprintf(os.Stderr, "Still waiting for root... (%ds/%ds)\n", elapsed, maxRetrySec)
	}

	return fmt.Errorf("timed out waiting for root after %d seconds", maxRetrySec)
}

// RunSu executes a command with su privileges and returns combined output.
func RunSu(command string) ([]byte, error) {
	cmd := exec.Command("su", "-c", command)
	return cmd.CombinedOutput()
}

// RunSuOk executes a command with su and returns true if exit code == 0.
func RunSuOk(command string) bool {
	_, err := RunSu(command)
	return err == nil
}
