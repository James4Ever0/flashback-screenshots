//go:build windows

package service

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// New creates a Windows installer.
func New() Installer {
	return &windowsInstaller{}
}

const taskXML = `<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Screenshot Worker</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>%s</Command>
      <Arguments>daemon --config "%s"</Arguments>
    </Exec>
  </Actions>
</Task>
`

type windowsInstaller struct{}

func (w *windowsInstaller) Install(system bool) error {
	exe := exePath()
	configPath := defaultConfigPathWindows()

	// Ensure config dir exists
	configDir := filepath.Dir(configPath)
	os.MkdirAll(configDir, 0755)

	// Write XML to temp file
	tmpFile := filepath.Join(os.TempDir(), "screenshot-worker-task.xml")
	xmlContent := fmt.Sprintf(taskXML, exe, configPath)
	if err := os.WriteFile(tmpFile, []byte(xmlContent), 0644); err != nil {
		return fmt.Errorf("failed to write task XML: %w", err)
	}
	defer os.Remove(tmpFile)

	// Create task with schtasks
	if _, err := runCmd("schtasks", "/Create", "/XML", tmpFile, "/TN", "ScreenshotWorker", "/F"); err != nil {
		return fmt.Errorf("schtasks create failed: %w", err)
	}

	fmt.Println("Installed Task Scheduler task: ScreenshotWorker")
	fmt.Println("The task runs at user logon in the interactive session.")
	return nil
}

func (w *windowsInstaller) Uninstall() error {
	if _, err := runCmd("schtasks", "/Delete", "/TN", "ScreenshotWorker", "/F"); err != nil {
		if strings.Contains(err.Error(), "does not exist") {
			return fmt.Errorf("task not installed")
		}
		return fmt.Errorf("schtasks delete failed: %w", err)
	}
	fmt.Println("Uninstalled ScreenshotWorker task")
	return nil
}

func (w *windowsInstaller) Status() (string, error) {
	out, err := runCmd("schtasks", "/Query", "/TN", "ScreenshotWorker", "/FO", "LIST")
	if err != nil {
		return "not installed", nil
	}
	// Extract Task State line
	lines := strings.Split(out, "\n")
	for _, line := range lines {
		if strings.Contains(line, "Task State:") || strings.Contains(line, "运行状态") {
			return strings.TrimSpace(strings.SplitN(line, ":", 2)[1]), nil
		}
	}
	return "unknown", nil
}

func defaultConfigPathWindows() string {
	localAppData := os.Getenv("LOCALAPPDATA")
	if localAppData == "" {
		localAppData = os.Getenv("APPDATA")
	}
	if localAppData == "" {
		localAppData = os.TempDir()
	}
	return filepath.Join(localAppData, "screenshot-worker", "config.toml")
}
