# Plan: Core Screenshot Worker — Simplified

## Goal
Build a minimal, resilient client binary that captures screenshots and streams them to a server over WebSocket.

## Phase 1: Bootstrap
- Choose Rust (standalone binary, cross-compile friendly, safe concurrency).
- Create project scaffold with `tokio`, `tokio-tungstenite`, `serde`, `tracing`.
- **Screenshot library per platform:**
  - macOS / Windows: `xcap` (well-supported, minimal system deps).
  - Linux: `ashpd` + `xdg-desktop-portal` (Wayland-correct), with fallback to shell-out `grim` (Wayland) or `scrot` (X11). See `compatibility-advice.txt` for why `xcap` is avoided on Linux.

## Phase 2: Capture Loop
- Configurable interval (default 5s).
- **Check if screen is locked before every capture.** If locked, skip the tick and wait for the next interval.
  - macOS: `CGSessionCopyCurrentDictionary` → read `CGSessionScreenIsLocked`.
  - Windows: `WTSRegisterSessionNotification` / `WTSUnRegisterSessionNotification` to track `WTS_SESSION_LOCK` / `WTS_SESSION_UNLOCK` events.
  - Linux (systemd): D-Bus `org.freedesktop.login1.Session` property `LockedHint`.
  - Linux (fallback): Check `xscreensaver-command -time` or `gnome-screensaver-command -q`.
- Capture full screen using platform-agnostic crate.
- Attach UTC timestamp with timezone to each frame.
- Extract focused window name and app name (platform-specific sys APIs).
- **Write every captured frame to a local disk ring buffer.**
  - Directory: `~/.local/share/screenshot-worker/buffer/` (Linux), `~/Library/Application Support/screenshot-worker/buffer/` (macOS), `%LOCALAPPDATA%\screenshot-worker\buffer\` (Windows).
  - Filename: `<ISO8601-timestamp>.jpg` (e.g., `2024-06-06T14-30-00+08-00.jpg`).
  - Configurable max file count (default 1000). When full, delete oldest before writing new.
  - This survives process crashes, reboots, and long disconnections.

## Phase 3: WebSocket Transport
- Read server URL from config file.
- On start: connect → send client hostname → begin heartbeat.
- Batch recent screenshots; send compressed JPEG/PNG payloads.
- Auto-reconnect with exponential backoff (max 60s).

## Phase 4: CLI Commands
The binary exposes subcommands via `clap`:
- `screenshot-worker daemon [--config <path>]` — Read config and run forever (main service mode).
- `screenshot-worker test-lock` — Check screen lock state, print result, exit.
- `screenshot-worker test-capture [--output <path>]` — Take one screenshot, save to disk, print path, exit.
- `screenshot-worker test-window` — Print currently focused window name + app name, exit.
- `screenshot-worker install [--system]` — Install as a background service (see Phase 6).
- `screenshot-worker uninstall` — Remove the background service.
- `screenshot-worker status` — Print whether the service is running.

## Phase 5: Server Handler (Minimal)
- Accept WebSocket connections.
- Parse client name from first message.
- Store incoming screenshots in date-based directory (`data/<client>/<YYYY-MM-DD>/`).
- Expose a simple REST endpoint: `GET /clients` (list), `GET /clients/<name>/latest` (redirect to image).

## Phase 6: Service Installation & Packaging
### Why This Matters
A screenshot worker needs to survive reboots and run without a terminal. Each platform has a different mechanism. **Windows is the tricky one** — see the caveat below.

### Linux — systemd (user service by default)
```bash
screenshot-worker install          # Installs ~/.config/systemd/user/screenshot-worker.service
screenshot-worker install --system # Installs /etc/systemd/system/screenshot-worker.service (needs sudo)
```
- Uses `systemctl --user daemon-reload && systemctl --user enable --now screenshot-worker`.
- User service runs in the user's session (can access X11/Wayland display).
- System service is available but may lack display access on modern Linux; document this limitation.

### macOS — launchd (user agent by default)
```bash
screenshot-worker install          # Installs ~/Library/LaunchAgents/com.screenshot.worker.plist
screenshot-worker install --system # Installs /Library/LaunchDaemons/ (needs sudo; not recommended)
```
- Uses `launchctl load -w ~/Library/LaunchAgents/com.screenshot.worker.plist`.
- User agent runs in the user's session (can capture screen, access accessibility APIs).
- **Caveat**: Screen Recording and Accessibility permissions must be granted manually in System Preferences → Security & Privacy. The binary cannot request these programmatically.

### Windows — Task Scheduler (NOT a Windows Service)
```powershell
screenshot-worker.exe install      # Creates a Task Scheduler task via schtasks.exe
screenshot-worker.exe uninstall    # Deletes the task
```
- **Why not NSSM or a native Windows Service?** Since Windows Vista, services run in Session 0 (isolated from the user desktop). A process in Session 0 **cannot capture the interactive user's screen** — APIs return black images or fail. This makes NSSM and `windows-service` crate unsuitable for screen capture.
- **Solution**: Use **Task Scheduler** with:
  - Trigger: "At logon of any user" or "At logon of specific user"
  - Action: run `screenshot-worker.exe daemon --config "C:\ProgramData\screenshot-worker\config.toml"`
  - Settings: "Run only when user is logged on" (required for screen access), "Restart on failure" every 1 minute
  - The task runs in the user's session, not Session 0.
- For development/testing, `screenshot-worker.exe daemon` runs in the foreground.

### Single Config File
- `config.toml` shipped alongside the binary (Linux/macOS: `/etc/screenshot-worker/` or `~/.config/screenshot-worker/`; Windows: `%ProgramData%\screenshot-worker\`).
- Buffer settings:
  ```toml
  [buffer]
  max_files = 1000          # max frames in disk ring buffer
  path = "/var/lib/screenshot-worker/buffer"  # optional override
  ```

## Out of Scope for This Plan
- Keylogging, file access, OCR, encryption, Android, web dashboard, AI agents.

## Known Compatibility Caveat
- Linux builds cannot be fully static binaries if using desktop capture APIs.
  See `compatibility-advice.txt` for the full per-platform dependency matrix
  and cross-compilation guidance.
