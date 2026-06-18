// Package session provides screen lock state detection.
package session

// Monitor reports whether the session is currently locked.
type Monitor interface {
	IsLocked() (bool, error)
}
