# Plan: Core Screenshot Worker — Detailed

## 1. Technology Stack
| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Rust | Standalone binary, memory safety, easy cross-compilation |
| Async runtime | tokio | Industry standard, integrates with tungstenite |
| WebSocket client | tokio-tungstenite | Async, supports TLS natively |
| Screenshots (macOS/Win) | xcap | Well-supported, minimal system deps |
| Screenshots (Linux) | ashpd + portal / grim / scrot | Wayland-correct; avoids xcap's PipeWire/D-Bus bloat |
| Window info | sysinfo + platform APIs (x11/wayland/ cocoa/ winapi) | Focused window & app name must come from client |
| Serialization | serde + MessagePack | Compact wire format |
| Logging | tracing + tracing-subscriber | Structured logs, file + stdout |
| Config | toml | Human-readable, single file |
| CLI | clap | Subcommands, flags, help generation |
| Service install | platform-specific | systemd (Linux), launchd (macOS), Task Scheduler (Windows) |

## 2. Project Layout
```
screenshot-worker/
├── Cargo.toml
├── config.toml
├── src/
│   ├── main.rs
│   ├── cli.rs            # clap subcommands & argument parsing
│   ├── capture.rs        # Screenshot + metadata extraction
│   ├── session.rs        # Screen lock / session state detection
│   ├── disk_buffer.rs    # Disk-based ring buffer for frame storage
│   ├── platform/
│   │   ├── linux.rs      # Linux session monitor + capture fallback
│   │   ├── macos.rs      # macOS session monitor + capture
│   │   └── windows.rs    # Windows session monitor + capture
│   ├── service.rs        # Install / uninstall / status commands
│   ├── transport.rs      # WebSocket connection manager
│   ├── config.rs         # Config loading & validation
│   ├── heartbeat.rs      # Keepalive & reconnect logic
│   └── logging.rs        # File-based tracing setup
└── packaging/
    ├── screenshot-worker.service
    ├── com.screenshot.worker.plist
    └── install.ps1
```

## 3. Configuration Schema (`config.toml`)
```toml
[server]
url = "wss://observer.example.com/ws"
# url = "ws://localhost:8080/ws"  # for debug

[capture]
interval_sec = 5
quality = 80          # JPEG quality 0-100

[buffer]
max_files = 1000      # max frames in disk ring buffer
# path = "/custom/buffer/dir"  # optional; defaults to platform data dir

[client]
name = "dev-laptop-01"
log_level = "info"
log_file = "/var/log/screenshot-worker.log"
```

## 4. Capture Module (`capture.rs`)

### 4.1 Screen Lock Detection (`session.rs`)
Before every capture tick, check if the session is locked. If locked, skip capture and log at `debug` level.

| Platform | Method | Notes |
|----------|--------|-------|
| **macOS** | `CGSessionCopyCurrentDictionary()` then read `CGSessionScreenIsLocked` key | CoreGraphics API; no extra entitlements needed |
| **Windows** | `WTSRegisterSessionNotification(hwnd, NOTIFY_FOR_THIS_SESSION)` → track `WTS_SESSION_LOCK` / `WTS_SESSION_UNLOCK` | Maintain atomic bool; register on start, unregister on shutdown |
| **Linux (systemd)** | D-Bus `org.freedesktop.login1.Session` property `LockedHint` | Most reliable; works on GNOME, KDE, modern desktops |
| **Linux (fallback)** | Shell out to `xscreensaver-command -time` or `gnome-screensaver-command -q` | For non-systemd or minimal X11 environments |
| **Linux (Wayland)** | `xdg-desktop-portal` screenshot request may itself fail when locked | Can double as implicit lock detection |

**Rust approach**: Expose a platform trait `SessionMonitor` with `is_locked() -> bool`. Implementations per platform file in `src/platform/`.

### 4.2 Screenshot Capture
- **macOS / Windows**: `capture_frame() -> Result<CaptureFrame>`:
  - Check `session.is_locked()`; return `Ok(None)` if true.
  - Call `xcap::Monitor::all()` → capture primary monitor.
  - Compress to JPEG using `image` crate.
- **Linux**: `capture_frame_linux() -> Result<CaptureFrame>`:
  - Check `session.is_locked()`; return `Ok(None)` if true.
  - Primary: `ashpd::screenshot::ScreenshotRequest` via xdg-desktop-portal (works on Wayland and X11).
  - Fallback 1: shell out to `grim -g "$(slurp)"` (if `WAYLAND_DISPLAY` is set).
  - Fallback 2: shell out to `scrot` (if `DISPLAY` is set).
  - This avoids linking libpipewire, libdbus, libwayland, and libEGL into the binary.
  - See `compatibility-advice.txt` for why `xcap` on Linux was rejected.

### 4.3 Frame Metadata
Populate `CaptureFrame` struct:
```rust
struct CaptureFrame {
    timestamp: DateTime<FixedOffset>,  // with explicit timezone
    image_jpeg: Vec<u8>,
    focused_window: Option<String>,
    app_name: Option<String>,
    session_locked: bool,  // included in metadata even when frame is skipped
}
```

- Platform-specific `focused_window` extraction:
  - Linux: `xcb` or `libwnck` for X11; D-Bus for Wayland (may require compositor support).
  - macOS: `CGWindowListCopyWindowInfo` + `kCGWindowOwnerName`.
  - Windows: `GetForegroundWindow` + `GetWindowText` / `GetWindowThreadProcessId`.

## 5. CLI Module (`cli.rs`)
Uses `clap` derive macros for typed argument parsing.

```rust
#[derive(Parser)]
enum Cli {
    /// Run the worker daemon (foreground)
    Daemon {
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
    },
    /// Test: check screen lock state and exit
    TestLock,
    /// Test: capture one screenshot and exit
    TestCapture {
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    /// Test: print focused window name and exit
    TestWindow,
    /// Install as a background service
    Install {
        #[arg(long, help = "Install as system service (requires admin)")]
        system: bool,
    },
    /// Uninstall the background service
    Uninstall,
    /// Check service status
    Status,
}
```

**Test commands** are deliberately stateless: they perform one action, print JSON or human-readable output, and exit with code 0 on success, 1 on error. This makes them scriptable and useful for CI or remote debugging.

## 6. Disk Buffer Module (`disk_buffer.rs`)
A **disk-based ring buffer** stores captured frames locally before (and during) transmission. This survives process crashes, OS reboots, and long disconnections — an in-memory ring buffer would lose data in all three cases.

### Design
- **Directory** (platform defaults):
  - Linux: `~/.local/share/screenshot-worker/buffer/`
  - macOS: `~/Library/Application Support/screenshot-worker/buffer/`
  - Windows: `%LOCALAPPDATA%\screenshot-worker\buffer\`
- **Filename schema**: `<ISO8601-timestamp>.jpg` (e.g., `2024-06-06T14-30-00+08-00.jpg`).
  - Timestamp is parseable from filename — no sidecar metadata needed.
  - Sorting filenames lexicographically sorts by time.
- **Write path**:
  1. Capture frame → compress to JPEG.
  2. Generate filename from `Utc::now()`.
  3. Call `disk_buffer.write(filename, jpeg_bytes)`.
  4. If current file count ≥ `max_files`, delete oldest N files (oldest = lexicographically first) before writing.
- **Read path** (used by transport):
  1. On startup, scan buffer directory → sort filenames → build in-memory index of pending frames.
  2. Transport drains the buffer by reading oldest first, sending over WebSocket, then deleting on successful ACK.
  3. If send fails, the file remains in the buffer for retry on next connect.
- **Concurrency**: `tokio::sync::RwLock` on the index. File writes are atomic (write to temp, rename).

```rust
struct DiskBuffer {
    dir: PathBuf,
    max_files: usize,
}

impl DiskBuffer {
    async fn write(&self, timestamp: DateTime<Utc>, data: Vec<u8>) -> Result<()>;
    async fn oldest(&self) -> Result<Option<(PathBuf, Vec<u8>)>>;
    async fn remove(&self, path: &Path) -> Result<()>;
    async fn len(&self) -> usize;
    async fn trim(&self) -> Result<()>;  // delete oldest if over max_files
}
```

### Why not in-memory?
| Scenario | In-memory ring buffer | Disk ring buffer |
|----------|----------------------|------------------|
| Process crash | Data lost | Survives |
| OS reboot | Data lost | Survives |
| Disconnect > buffer size × interval | Data lost | Survives (capacity = disk space) |
| Memory pressure | Limited by RAM | Limited by disk |

## 7. Transport Module (`transport.rs`)
- `ConnectionManager` struct with states: `Disconnected | Connecting | Connected`.
- Connection flow:
  1. On startup: scan `disk_buffer` for pending frames; begin draining oldest first.
  2. Open WebSocket to `config.server.url`.
  3. Send `ClientHello { name: config.client.name, version: "0.1.0" }`.
  4. Spawn read loop (handle server commands: `RequestLatest`, `Ping`, `AckFrame { filename }`).
  5. Spawn write loop (read oldest frame from disk buffer, send, await ACK, delete file).
- Reconnect strategy:
  - Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, capped at 60s.
  - Jitter: ±20% to avoid thundering herd.
  - On reconnect: resume draining disk buffer from where it left off — no special "resend" logic needed because data is already on disk.

## 8. Heartbeat & Resilience (`heartbeat.rs`)
- Send `Ping` every 30s; expect `Pong` within 10s.
- If no `Pong`: mark connection dead, trigger reconnect.
- On OS sleep/wake: detect via `tokio::signal` or platform hooks, force reconnect.
- Disk buffer continues to accumulate frames during disconnections; no memory pressure regardless of outage duration.

## 9. Service Installation (`service.rs`)
The `install`, `uninstall`, and `status` subcommands manipulate the platform's background-service mechanism. The binary contains the service definition templates as embedded strings (via `include_str!`) so no external files are needed for installation.

### 9.1 Linux — systemd
- **Default**: user service (`~/.config/systemd/user/screenshot-worker.service`).
  ```ini
  [Unit]
  Description=Screenshot Worker
  After=graphical-session.target

  [Service]
  Type=simple
  ExecStart=%h/.local/bin/screenshot-worker daemon --config %h/.config/screenshot-worker/config.toml
  Restart=always
  RestartSec=5

  [Install]
  WantedBy=default.target
  ```
- **`--system` flag**: system service (`/etc/systemd/system/screenshot-worker.service`). Requires `sudo`. Note: system services may lack `DISPLAY` / `WAYLAND_DISPLAY` access on modern Linux; prefer user services.
- **Install flow**: write unit file → `systemctl --user daemon-reload` → `systemctl --user enable --now screenshot-worker`.
- **Status flow**: `systemctl --user is-active screenshot-worker`.

### 9.2 macOS — launchd
- **Default**: user agent (`~/Library/LaunchAgents/com.screenshot.worker.plist`).
  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" ...>
  <plist version="1.0">
  <dict>
    <key>Label</key><string>com.screenshot.worker</string>
    <key>ProgramArguments</key>
    <array>
      <string>/usr/local/bin/screenshot-worker</string>
      <string>daemon</string>
      <string>--config</string>
      <string>/usr/local/etc/screenshot-worker/config.toml</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/var/log/screenshot-worker.log</string>
    <key>StandardErrorPath</key><string>/var/log/screenshot-worker.log</string>
  </dict>
  </plist>
  ```
- **`--system` flag**: daemon (`/Library/LaunchDaemons/`). Not recommended — daemons run as root and cannot access the user's screen recording permission.
- **Install flow**: write plist → `launchctl load -w ~/Library/LaunchAgents/com.screenshot.worker.plist`.
- **Caveat**: Screen Recording and Accessibility permissions must be granted manually by the user in System Preferences → Security & Privacy. The binary cannot request these programmatically. Document this clearly in the install output.

### 9.3 Windows — Task Scheduler (NOT a Windows Service)
- **Why not NSSM or native Windows Service?** Since Windows Vista, services run in Session 0 (isolated from the user desktop). A process in Session 0 **cannot capture the interactive user's screen** — capture APIs return black images or fail entirely. NSSM and the `windows-service` crate both produce Session-0 services, making them unsuitable for screen capture.
- **Solution**: Use **Task Scheduler** with `schtasks.exe`.
  ```xml
  <!-- Task XML template embedded in binary -->
  <Task>
    <RegistrationInfo><Description>Screenshot Worker</Description></RegistrationInfo>
    <Triggers>
      <LogonTrigger><Enabled>true</Enabled></LogonTrigger>
    </Triggers>
    <Settings>
      <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
      <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
      <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
      <AllowStartOnDemand>true</AllowStartOnDemand>
      <Enabled>true</Enabled>
      <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
      <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd></IdleSettings>
      <AllowHardTerminate>true</AllowHardTerminate>
      <StartWhenAvailable>true</StartWhenAvailable>
      <RunOnlyIfIdle>false</RunOnlyIfIdle>
      <WakeToRun>false</WakeToRun>
      <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
      <Priority>7</Priority>
      <RestartOnFailure>
        <Interval>PT1M</Interval>
        <Count>999</Count>
      </RestartOnFailure>
    </Settings>
    <Actions>
      <Exec>
        <Command>C:\ProgramData\screenshot-worker\screenshot-worker.exe</Command>
        <Arguments>daemon --config "C:\ProgramData\screenshot-worker\config.toml"</Arguments>
      </Exec>
    </Actions>
    <Principals>
      <Principal>
        <LogonType>InteractiveToken</LogonType>
        <RunLevel>LeastPrivilege</RunLevel>
      </Principal>
    </Principals>
  </Task>
  ```
- Key settings:
  - `LogonTrigger` — starts when the user logs on.
  - `InteractiveToken` — runs in the user's session (not Session 0), so screen capture works.
  - `RestartOnFailure` — restart every 1 minute, up to 999 times.
  - `DisallowStartIfOnBatteries=false` — continue on battery.
- **Install flow**: write XML to temp file → `schtasks.exe /Create /XML /TN "ScreenshotWorker"` → delete temp file.
- **Status flow**: `schtasks.exe /Query /TN "ScreenshotWorker" /FO LIST`.
- **For development**: `screenshot-worker.exe daemon` runs in the foreground console.

## 10. Server Handler (Minimal — can be separate binary or same repo)
- Language: Rust with `axum` + `tokio-tungstenite`.
- Endpoints:
  - `WS /ws` — client ingress.
  - `GET /api/clients` — JSON list of connected + recently seen clients.
  - `GET /api/clients/{name}/latest` — 302 redirect to stored image path.
- Storage:
  - Flat files: `data/{client_name}/{YYYY-MM-DD}/{HH-MM-SS}.jpg`.
  - SQLite index for fast latest lookup: `clients(id, name, last_seen)`, `frames(id, client_id, path, timestamp)`.
- No auth in v0; assume network-level security or VPN.

## 11. Logging & Observability
- `tracing` with `tracing-appender` for daily log rotation.
- Log every: connect, disconnect, capture event, server command, error.
- Include frame counts, bytes sent, reconnect count in periodic stats log.

## 12. Build & Distribution
- `cargo build --release` per target.
- **Linux builds**: Use `x86_64-unknown-linux-gnu` (glibc) target. `musl` static builds are incompatible with desktop portal capture. The binary is dynamically linked; ship via `.deb`/`.rpm` with declared runtime deps.
- **macOS builds**: `cargo build --target aarch64-apple-darwin` / `x86_64-apple-darwin`. Fully static Rust code; system frameworks are always dynamic but universally present.
- **Windows builds**: `cargo build --target x86_64-pc-windows-msvc`. Statically links Rust deps; no extra runtime libraries.
- **Cross-compilation**: Use `cargo-zigbuild` or GitHub Actions runners. Avoid `cross-rs` for Linux builds due to missing PipeWire/D-Bus headers in default images.
- Packaging scripts install binary + config + service file + dependency declarations.

## 13. Milestones
| # | Deliverable | ETA |
|---|-------------|-----|
| 1 | Capture + save to disk locally | Day 1 |
| 2 | Session lock detection + window title extraction | Day 2 |
| 3 | Disk buffer ring (write, read, trim, crash recovery) | Day 3 |
| 4 | WebSocket client + reconnect + buffer drain | Day 4 |
| 5 | CLI subcommands (test-lock, test-capture, test-window) | Day 5 |
| 6 | Service install/uninstall/status per platform | Day 6 |
| 7 | Server handler + storage + frame ACK | Day 7 |
| 8 | Cross-platform build + release | Day 8 |
