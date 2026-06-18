# screenshot-worker-universal

Universal build of the screenshot worker. Supports **Wayland + X11** on Linux via `xdg-desktop-portal` with `grim`/`scrot` fallback. Uses `xcap = "=0.4.0"` on macOS and Windows.

## What This Variant Does

- **Linux Wayland**: Primary capture via `ashpd` + `xdg-desktop-portal`. Falls back to `grim` (if `WAYLAND_DISPLAY` is set).
- **Linux X11**: Falls back to `scrot` (if `DISPLAY` is set).
- **macOS**: Uses `CoreGraphics` via xcap 0.4.0.
- **Windows**: Uses GDI/DXGI via xcap 0.4.0.

## Build

```bash
cargo build --release
```

## Linux Runtime Dependencies

### Wayland path
```bash
# Debian / Ubuntu
sudo apt-get install xdg-desktop-portal grim

# Arch
sudo pacman -S xdg-desktop-portal grim

# Fedora
sudo dnf install xdg-desktop-portal grim
```

### X11 fallback path
```bash
# Debian / Ubuntu
sudo apt-get install scrot

# Arch
sudo pacman -S scrot

# Fedora
sudo dnf install scrot
```

> The binary does **not** link against PipeWire, libwayland, or libEGL. It talks to the portal over D-Bus and shells out to `grim`/`scrot` for the actual pixel capture. This keeps the binary portable and avoids the xcap 0.5+ dependency bloat.

## Why Not xcap on Linux Here?

| Approach | Static Binary? | Linked System Libs | Notes |
|----------|---------------|-------------------|-------|
| xcap 0.4.0 (X11) | No | libxcb, libxrandr | OK for X11-only |
| xcap 0.5.0+ | No | +PipeWire, +D-Bus, +EGL | Too heavy, breaks CI |
| ashpd + grim/scrot | No | Only D-Bus (runtime) | Wayland-correct, lighter |

This crate takes the `ashpd + shell-out` approach for Linux to get proper Wayland support without linking heavy desktop libraries.

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

## Wayland Compositor Compatibility

| Compositor | Portal Support | grim Support | Status |
|-----------|---------------|--------------|--------|
| GNOME (Mutter) | Yes (built-in) | No (wlroots only) | Portal primary, no grim fallback |
| KDE (KWin) | Yes (built-in) | No (wlroots only) | Portal primary, no grim fallback |
| Sway | Yes (wlroots portal) | Yes | Full support |
| Hyprland | Yes (wlroots portal) | Yes | Full support |
| River | Yes (wlroots portal) | Yes | Full support |

If the portal is unavailable and `grim` is not installed, the capture tick will fail and retry on the next interval.
