# Plan: Android Client — Simplified

> Android is a fundamentally different platform from desktop. Treat as a separate sub-project.

## Goal
An Android service that captures screenshots and sends them to the same server.

## Key Differences from Desktop
| Aspect | Desktop | Android |
|--------|---------|---------|
| Runtime | Native binary | APK (Kotlin/Java or Rust via JNI) |
| Permissions | User/root for some features | `MediaProjection` API, system alert window |
| Background | systemd/launchd/service | Foreground service with persistent notification |
| Boot start | Service registration | `BOOT_COMPLETED` receiver |
| Screenshot | Library call | `MediaProjection` + `ImageReader` |
| Standalone | Single binary | APK + dependencies |

## Approach

### Option A: Native Binary (Rooted Devices)
- Compile Rust binary for `aarch64-linux-android` using NDK.
- Run `screencap` shell command (requires root or adb shell).
- Very limited; not recommended for general use.

### Option B: APK with MediaProjection (Recommended)
- Kotlin + Android Studio.
- Foreground service with `START_STICKY` for resilience.
- `MediaProjection` initiated once (requires user consent dialog).
- Capture to `ImageReader` → encode JPEG → upload via OkHttp/WebSocket.
- Store server config in `SharedPreferences`.
- Auto-restart on boot via `BroadcastReceiver`.

## Deferred / Out of Scope
- System app / priv-app installation (requires OEM signing).
- Accessibility service for window titles (possible but intrusive).
- Keylogging on Android (separate massive topic).
