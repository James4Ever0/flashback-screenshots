// Package briefing periodically outputs SERVICE_BRIEFING JSON lines to stdout
// so that external tools (logcat, Kotlin app) can parse service status.
package briefing

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"android-go-bootstrap-su/internal/netcheck"
)

// Briefing holds the current service state for output.
type Briefing struct {
	Status            string `json:"status"`
	UptimeMs          int64  `json:"uptime_ms"`
	Captures          int64  `json:"captures"`
	BufferFiles       int    `json:"buffer_files"`
	BufferBytes       int64  `json:"buffer_bytes"`
	WebsocketConnected bool  `json:"websocket_connected"`
	FramesSent        int64  `json:"frames_sent"`
	LastCaptureMs     int64  `json:"last_capture_ms"`
	LastSendMs        int64  `json:"last_send_ms"`
	ScreenLocked      bool   `json:"screen_locked"`
	ScreenOn          bool   `json:"screen_on"`
	NetworkType       string `json:"network_type"`
	UploadBlocked     bool   `json:"upload_blocked"`
	LastError         string `json:"last_error"`
	DisplayID         int    `json:"display_id"`
	ClientName        string `json:"client_name"`
	Timestamp         int64  `json:"timestamp"`
}

// Emitter prints SERVICE_BRIEFING lines at a fixed interval.
type Emitter struct {
	interval   time.Duration
	startTime  time.Time
	clientName string
	displayID  int
	wifiOnly   bool

	// Mutable state (set by daemon)
	Captures      int64
	BufferFiles   int
	BufferBytes   int64
	WSConnected   bool
	FramesSent    int64
	LastCaptureMs int64
	LastSendMs    int64
	ScreenLocked  bool
	ScreenOn      bool
	LastError     string
}

// NewEmitter creates a briefing emitter.
func NewEmitter(intervalSec int, clientName string, displayID int, wifiOnly bool) *Emitter {
	return &Emitter{
		interval:   time.Duration(intervalSec) * time.Second,
		startTime:  time.Now(),
		clientName: clientName,
		displayID:  displayID,
		wifiOnly:   wifiOnly,
	}
}

// Run blocks, printing SERVICE_BRIEFING lines until ctx is cancelled.
func (e *Emitter) Run(ctx context.Context) {
	ticker := time.NewTicker(e.interval)
	defer ticker.Stop()

	// Print immediately on start
	e.emit()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			e.emit()
		}
	}
}

func (e *Emitter) emit() {
	now := time.Now()
	netType := netcheck.Type()
	blocked := e.wifiOnly && !netcheck.IsWiFi()

	// Convert absolute timestamps to elapsed milliseconds
	var lastCaptureMs, lastSendMs int64
	if e.LastCaptureMs > 0 {
		lastCaptureMs = now.UnixMilli() - e.LastCaptureMs
	}
	if e.LastSendMs > 0 {
		lastSendMs = now.UnixMilli() - e.LastSendMs
	}

	b := Briefing{
		Status:            "running",
		UptimeMs:          now.Sub(e.startTime).Milliseconds(),
		Captures:          e.Captures,
		BufferFiles:       e.BufferFiles,
		BufferBytes:       e.BufferBytes,
		WebsocketConnected: e.WSConnected,
		FramesSent:        e.FramesSent,
		LastCaptureMs:     lastCaptureMs,
		LastSendMs:        lastSendMs,
		ScreenLocked:      e.ScreenLocked,
		ScreenOn:          e.ScreenOn,
		NetworkType:       netType,
		UploadBlocked:     blocked,
		LastError:         e.LastError,
		DisplayID:         e.displayID,
		ClientName:        e.clientName,
		Timestamp:         now.UnixMilli(),
	}

	jsonBytes, _ := json.Marshal(b)
	fmt.Printf("SERVICE_BRIEFING %s\n", string(jsonBytes))
}

