# screenshot-worker-x11

X11-only build of the screenshot worker. Uses `xcap = "=0.4.0"` which is pre-PipeWire and has a lighter dependency footprint than 0.5.x+.

## What This Variant Does

- **Linux**: Captures via X11 (`libxcb` + `libxrandr`). No Wayland support.
- **macOS**: Uses `CoreGraphics` via xcap 0.4.0.
- **Windows**: Uses GDI/DXGI via xcap 0.4.0.

## Build

```bash
cargo build --release
```

## Linux Runtime Dependencies

```bash
# Debian / Ubuntu
sudo apt-get install libxcb1 libxrandr2

# Arch
sudo pacman -S libxcb libxrandr

# Fedora
sudo dnf install libxcb libXrandr
```

## Why xcap 0.4.0?

| Version | Linux Deps | Wayland | Notes |
|---------|-----------|---------|-------|
| 0.4.0 | `libxcb`, `libxrandr` | No | Light, builds easily |
| 0.4.1+ | adds `libwayshot` | Partial | Heavier |
| 0.5.0+ | adds PipeWire, zbus | Partial | Breaks cross-compilation |

For an X11-only deployment, 0.4.0 is the sweet spot. You trade Wayland support for build simplicity.

## Config

Copy `config.toml.example` to `~/.config/screenshot-worker/config.toml` and edit:

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
./screenshot-worker test-capture --output /tmp/test.jpg
./screenshot-worker test-window

# Install as systemd user service
./screenshot-worker install
./screenshot-worker status
```
