module screenshot-worker

go 1.21

require (

	// Configuration parsing.
	github.com/BurntSushi/toml v1.4.0

	// D-Bus for Wayland xdg-desktop-portal and logind session lock.
	github.com/godbus/dbus/v5 v5.1.0

	// WebSocket client.
	github.com/gorilla/websocket v1.5.3
	// Cross-platform screen capture.
	// Linux: pure Go X11 (jezek/xgb) + D-Bus portal for Wayland.
	// macOS: CGO (CoreGraphics / ScreenCaptureKit).
	// Windows: pure Go (syscall + lxn/win).
	github.com/kbinani/screenshot v0.0.0-20240820160931-a8a2c5d0e191

	// Structured logging.
	github.com/rs/zerolog v1.33.0

	// CLI framework.
	github.com/spf13/cobra v1.8.1
)

require (
	github.com/gen2brain/shm v0.1.0 // indirect; X11 SHM extension
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/jezek/xgb v1.1.1 // indirect; pure Go X11 protocol
	github.com/lxn/win v0.0.0-20210218163916-a377121e959e // indirect; Windows API
	github.com/mattn/go-colorable v0.1.13 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	golang.org/x/sys v0.24.0 // indirect
)
