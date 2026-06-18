// Package service handles install/uninstall/status of the background service.
package service

import (
	"fmt"
	"os"
	"os/exec"
)

// Installer provides platform-specific service management.
type Installer interface {
	Install(system bool) error
	Uninstall() error
	Status() (string, error)
}

type noopInstaller struct{}

func (n *noopInstaller) Install(system bool) error { return fmt.Errorf("service install not supported on this platform") }
func (n *noopInstaller) Uninstall() error          { return fmt.Errorf("service uninstall not supported on this platform") }
func (n *noopInstaller) Status() (string, error)   { return "unknown", nil }

// exePath returns the path to the current executable.
func exePath() string {
	path, err := os.Executable()
	if err != nil {
		return "screenshot-worker"
	}
	return path
}

// runCmd executes a command and returns its output.
func runCmd(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("%s failed: %w (output: %s)", name, err, string(out))
	}
	return string(out), nil
}
