//go:build linux

package service

import (
	"fmt"
	"os"
	"path/filepath"
)

// New creates a Linux installer.
func New() Installer {
	return &linuxInstaller{}
}

const systemdUserUnit = `[Unit]
Description=Screenshot Worker
After=graphical-session.target

[Service]
Type=simple
ExecStart=%s daemon --config %s
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
`

type linuxInstaller struct{}

func (l *linuxInstaller) Install(system bool) error {
	exe := exePath()
	configPath := defaultConfigPath()

	var unitDir, unitPath string
	if system {
		unitDir = "/etc/systemd/system"
		unitPath = filepath.Join(unitDir, "screenshot-worker.service")
	} else {
		home, _ := os.UserHomeDir()
		unitDir = filepath.Join(home, ".config", "systemd", "user")
		unitPath = filepath.Join(unitDir, "screenshot-worker.service")
	}

	if err := os.MkdirAll(unitDir, 0755); err != nil {
		return fmt.Errorf("failed to create unit dir: %w", err)
	}

	unitContent := fmt.Sprintf(systemdUserUnit, exe, configPath)
	if err := os.WriteFile(unitPath, []byte(unitContent), 0644); err != nil {
		return fmt.Errorf("failed to write unit file: %w", err)
	}

	// Reload and start
	scope := "--user"
	if system {
		scope = ""
	}

	if _, err := runCmd("systemctl", scope, "daemon-reload"); err != nil {
		return fmt.Errorf("daemon-reload failed: %w", err)
	}
	if _, err := runCmd("systemctl", scope, "enable", "screenshot-worker"); err != nil {
		return fmt.Errorf("enable failed: %w", err)
	}
	if _, err := runCmd("systemctl", scope, "start", "screenshot-worker"); err != nil {
		return fmt.Errorf("start failed: %w", err)
	}

	fmt.Printf("Installed systemd service: %s\n", unitPath)
	fmt.Printf("View logs: journalctl %s -u screenshot-worker -f\n", scope)
	return nil
}

func (l *linuxInstaller) Uninstall() error {
	home, _ := os.UserHomeDir()
	userPath := filepath.Join(home, ".config", "systemd", "user", "screenshot-worker.service")
	systemPath := "/etc/systemd/system/screenshot-worker.service"

	// Try user first, then system
	unitPath := userPath
	scope := "--user"
	if _, err := os.Stat(userPath); os.IsNotExist(err) {
		if _, err := os.Stat(systemPath); err == nil {
			unitPath = systemPath
			scope = ""
		} else {
			return fmt.Errorf("service not installed")
		}
	}

	if _, err := runCmd("systemctl", scope, "stop", "screenshot-worker"); err != nil {
		fmt.Printf("Warning: stop failed: %v\n", err)
	}
	if _, err := runCmd("systemctl", scope, "disable", "screenshot-worker"); err != nil {
		fmt.Printf("Warning: disable failed: %v\n", err)
	}
	if err := os.Remove(unitPath); err != nil {
		return fmt.Errorf("failed to remove unit file: %w", err)
	}
	if _, err := runCmd("systemctl", scope, "daemon-reload"); err != nil {
		return fmt.Errorf("daemon-reload failed: %w", err)
	}

	fmt.Println("Uninstalled screenshot-worker service")
	return nil
}

func (l *linuxInstaller) Status() (string, error) {
	out, err := runCmd("systemctl", "--user", "is-active", "screenshot-worker")
	if err != nil {
		// Try system scope
		out, err = runCmd("systemctl", "is-active", "screenshot-worker")
		if err != nil {
			return "not installed", nil
		}
	}
	return out, nil
}

func defaultConfigPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "screenshot-worker", "config.toml")
}
