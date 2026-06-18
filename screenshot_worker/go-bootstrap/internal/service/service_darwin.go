//go:build darwin

package service

import (
	"fmt"
	"os"
	"path/filepath"
)

// New creates a macOS installer.
func New() Installer {
	return &darwinInstaller{}
}

const launchdPlist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.screenshot.worker</string>
    <key>ProgramArguments</key>
    <array>
        <string>%s</string>
        <string>daemon</string>
        <string>--config</string>
        <string>%s</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>%s</string>
    <key>StandardErrorPath</key>
    <string>%s</string>
</dict>
</plist>
`

type darwinInstaller struct{}

func (d *darwinInstaller) Install(system bool) error {
	exe := exePath()
	configPath := defaultConfigPathDarwin()
	logPath := filepath.Join(os.TempDir(), "screenshot-worker.log")

	var plistDir, plistPath string
	if system {
		plistDir = "/Library/LaunchDaemons"
		plistPath = filepath.Join(plistDir, "com.screenshot.worker.plist")
	} else {
		home, _ := os.UserHomeDir()
		plistDir = filepath.Join(home, "Library", "LaunchAgents")
		plistPath = filepath.Join(plistDir, "com.screenshot.worker.plist")
	}

	if err := os.MkdirAll(plistDir, 0755); err != nil {
		return fmt.Errorf("failed to create plist dir: %w", err)
	}

	plistContent := fmt.Sprintf(launchdPlist, exe, configPath, logPath, logPath)
	if err := os.WriteFile(plistPath, []byte(plistContent), 0644); err != nil {
		return fmt.Errorf("failed to write plist: %w", err)
	}

	if _, err := runCmd("launchctl", "load", "-w", plistPath); err != nil {
		return fmt.Errorf("launchctl load failed: %w", err)
	}

	fmt.Printf("Installed launchd agent: %s\n", plistPath)
	fmt.Println("Note: Screen Recording and Accessibility permissions must be granted manually in")
	fmt.Println("      System Preferences -> Security & Privacy -> Privacy")
	return nil
}

func (d *darwinInstaller) Uninstall() error {
	home, _ := os.UserHomeDir()
	userPath := filepath.Join(home, "Library", "LaunchAgents", "com.screenshot.worker.plist")
	systemPath := "/Library/LaunchDaemons/com.screenshot.worker.plist"

	plistPath := userPath
	if _, err := os.Stat(userPath); os.IsNotExist(err) {
		if _, err := os.Stat(systemPath); err == nil {
			plistPath = systemPath
		} else {
			return fmt.Errorf("service not installed")
		}
	}

	if _, err := runCmd("launchctl", "unload", "-w", plistPath); err != nil {
		fmt.Printf("Warning: unload failed: %v\n", err)
	}
	if err := os.Remove(plistPath); err != nil {
		return fmt.Errorf("failed to remove plist: %w", err)
	}

	fmt.Println("Uninstalled screenshot-worker agent")
	return nil
}

func (d *darwinInstaller) Status() (string, error) {
	out, err := runCmd("launchctl", "list", "com.screenshot.worker")
	if err != nil {
		return "not installed", nil
	}
	return out, nil
}

func defaultConfigPathDarwin() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "screenshot-worker", "config.toml")
}
