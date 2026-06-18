# Screenshot Worker — Go Bootstrap

Single-module Go bootstrap using `github.com/kbinani/screenshot`. Works on X11 and Wayland (auto-detected at runtime), macOS, and Windows — no platform-specific builds needed.

## Why One Module?

The `kbinani/screenshot` library auto-detects the display server at runtime via `XDG_SESSION_TYPE`:

- `XDG_SESSION_TYPE=x11` → uses pure Go X11 (`jezek/xgb`)
- `XDG_SESSION_TYPE=wayland` → uses D-Bus portal (`godbus/dbus`)
- macOS / Windows → native platform APIs

There is no compile-time split. One binary runs everywhere.

## Prerequisites

- Go 1.21+
- Linux: nothing else (no `-dev` packages, no `grim`)
- macOS: Xcode Command Line Tools (for CGO)
- Windows: nothing else

## Quick Start

```bash
# Build for current platform
go build -o screenshot-worker ./cmd/screenshot-worker

# Linux fully static binary (recommended for distribution)
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o screenshot-worker ./cmd/screenshot-worker
```

## Cross-Compilation

```bash
# Linux AMD64 — from macOS, Windows, or Linux
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o screenshot-worker-linux ./cmd/screenshot-worker

# macOS Apple Silicon — from macOS host only (needs CGO)
GOOS=darwin GOARCH=arm64 go build -o screenshot-worker-mac ./cmd/screenshot-worker

# macOS Intel — from macOS host only
GOOS=darwin GOARCH=amd64 go build -o screenshot-worker-mac-intel ./cmd/screenshot-worker

# Windows — from any host
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o screenshot-worker.exe ./cmd/screenshot-worker

# ARM64 Linux (e.g., Raspberry Pi, Ampere)
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o screenshot-worker-linux-arm64 ./cmd/screenshot-worker
```

## Runtime Dependencies

| Platform | Needs |
|----------|-------|
| Linux X11 | `DISPLAY` env var set, X server running |
| Linux Wayland | `xdg-desktop-portal` + backend running |
| macOS | Nothing (frameworks built into OS) |
| Windows | Nothing (GDI built into OS) |

### Installing portal backend (Wayland only)

```bash
# Debian / Ubuntu
sudo apt-get install xdg-desktop-portal xdg-desktop-portal-wlr

# Arch
sudo pacman -S xdg-desktop-portal xdg-desktop-portal-wlr

# Fedora
sudo dnf install xdg-desktop-portal xdg-desktop-portal-wlr
```

## glibc / Compatibility

Linux builds are **fully static** when `CGO_ENABLED=0`:

```bash
$ ldd screenshot-worker
        not a dynamic executable
```

This means one Linux binary runs on:
- Ubuntu 16.04, 22.04, 24.04
- Alpine Linux (musl)
- CentOS 7
- Any Linux with a compatible kernel

No glibc compatibility concerns. No Docker build needed for older distros.

## Config

Copy `config.toml.example` to `~/.config/screenshot-worker/config.toml`:

```toml
[server]
url = "ws://localhost:8080/ws"

[capture]
interval_sec = 5
quality = 80

[buffer]
max_files = 1000

[client]
name = "my-pc"
log_level = "info"
```

## Run

```bash
# Foreground daemon
./screenshot-worker daemon --config ~/.config/screenshot-worker/config.toml

# One-off tests
./screenshot-worker test-lock
./screenshot-worker test-capture --output /tmp/test.png
./screenshot-worker test-window

# Install as systemd user service
./screenshot-worker install
./screenshot-worker status
```

## Library Internals

```
kbinani/screenshot
├── nix.go                 # Display enumeration (Xinerama via xgb)
├── nix_xwindow.go         # X11 capture (xgb + SHM)
├── nix_wayland.go         # Wayland capture (D-Bus portal)
├── nix_dbus_available.go  # Runtime dispatch: X11 vs Wayland
├── darwin.go              # macOS (CGO — CoreGraphics)
├── windows.go             # Windows (pure Go — GDI BitBlt)
└── screenshot.go          # Platform-agnostic helpers
```

- **Linux X11**: Speaks X11 wire protocol directly over `/tmp/.X11-unix/X0`. No `libX11.so`, no `libxcb.so`.
- **Linux Wayland**: Calls `org.freedesktop.portal.Screenshot.Screenshot` over D-Bus. The portal backend asks the compositor for a frame and returns a file path.
- **No `grim`**, no `scrot`, no shell-out.

## Wayland Compositor Compatibility

| Compositor | Portal Backend | Status |
|-----------|---------------|--------|
| GNOME (Mutter) | `xdg-desktop-portal-gnome` | Full |
| KDE (KWin) | `xdg-desktop-portal-kde` | Full |
| Sway | `xdg-desktop-portal-wlr` | Full |
| Hyprland | `xdg-desktop-portal-hyprland` | Full |
| River | `xdg-desktop-portal-wlr` | Full |

If the portal is unavailable, the capture tick fails and retries on the next interval.
