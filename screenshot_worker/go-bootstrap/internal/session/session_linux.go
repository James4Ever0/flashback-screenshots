//go:build linux

package session

import (
	"os"

	"github.com/godbus/dbus/v5"
)

// linuxMonitor checks session lock via logind D-Bus.
type linuxMonitor struct{}

// NewMonitor creates a session lock monitor for Linux.
func NewMonitor() Monitor {
	return &linuxMonitor{}
}

// IsLocked queries org.freedesktop.login1.Session for LockedHint.
func (m *linuxMonitor) IsLocked() (bool, error) {
	// Get the current session path from logind
	conn, err := dbus.ConnectSystemBus()
	if err != nil {
		// Fallback: assume not locked if D-Bus is unavailable
		return false, nil
	}
	defer conn.Close()

	// Try to get session from PID
	sessionPath, err := getSessionPath(conn)
	if err != nil {
		return false, nil
	}

	obj := conn.Object("org.freedesktop.login1", dbus.ObjectPath(sessionPath))
	var locked bool
	if err := obj.Call("org.freedesktop.DBus.Properties.Get", 0,
		"org.freedesktop.login1.Session", "LockedHint").Store(&locked); err != nil {
		return false, nil
	}

	return locked, nil
}

func getSessionPath(conn *dbus.Conn) (string, error) {
	// Get session for current PID
	pid := os.Getpid()
	manager := conn.Object("org.freedesktop.login1", "/org/freedesktop/login1")
	var sessionPath dbus.ObjectPath
	if err := manager.Call("org.freedesktop.login1.Manager.GetSessionByPID", 0, uint32(pid)).Store(&sessionPath); err != nil {
		// Fallback: try getting user session
		uid := uint32(os.Getuid())
		if err := manager.Call("org.freedesktop.login1.Manager.GetUserByUID", 0, uid).Store(&sessionPath); err != nil {
			return "", err
		}
		return string(sessionPath), nil
	}
	return string(sessionPath), nil
}
