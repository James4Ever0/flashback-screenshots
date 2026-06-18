# Plan: Advanced Features (Keylogging, File Access, Encryption)

> ⚠️ These features significantly increase the sensitivity and security surface area of the system. Implement only with explicit authorization and legal compliance.

## 1. Keylogging

### Rationale
The prompt mentions keylogging as a client-side capability. This is a high-risk feature that should be isolated.

### Implementation Options
| Approach | Pros | Cons |
|----------|------|------|
| Separate binary (`keylogd`) | Process isolation, optional install, easier audit | Extra packaging, IPC needed |
| Same binary, feature flag | Simpler distribution | Increases attack surface for all installs |

**Recommendation: Separate binary `keylogd`**.

### Technical Details
- **Linux**: `evdev` crate to read `/dev/input/event*` (requires root or `input` group).
- **macOS**: `CGEventTap` (requires Accessibility permissions + user approval).
- **Windows**: `SetWindowsHookEx(WH_KEYBOARD_LL)` (requires no elevation for global hooks).
- Data format: `{ timestamp, window, key_sequence }`.
- Transport: Same WebSocket to server, but tagged `type: "keylog"`.
- Storage: Separate table/collection on server; never mix with screenshot frames.
- Privacy: Configurable blackout windows (e.g., exclude certain apps like password managers).

## 2. Local File Read Access

### Rationale
Server-requested file retrieval from client machines. Useful for log collection, config audit.

### Implementation
- **Protocol extension**: Server sends `ReadFile { path: String, max_bytes: u64 }` over WebSocket.
- **Client response**: `FileData { path, content_base64, truncated: bool }` or `FileError { path, reason }`.
- **Security guardrails**:
  - Whitelist of readable paths (configurable regex/prefix list).
  - Max file size limit (default 1MB).
  - No directory traversal (canonicalize path, verify against whitelist).
  - Read-only: never write.
- **Alternative**: Use existing SSH/SCP infrastructure rather than building into this tool. Document this as preferred alternative.

## 3. Encryption

### Rationale
Protect screenshot data in transit and optionally at rest.

### In-Transit (Transport Layer)
- **WebSocket over TLS (`wss://`)** — mandatory for production.
- Self-signed certs supported for LAN/debug via config flag `insecure_skip_verify = true`.

### Application-Layer Encryption (Optional)
- **Shared AES-256-GCM key** pre-configured on both sides.
- Use case: defense in depth even if TLS is compromised or terminated by a proxy.
- Config: `encryption = "none" | "aes256gcm"`.
- Implementation:
  - Derive key from pre-shared passphrase using PBKDF2 or Argon2.
  - Encrypt frame payload before WebSocket send.
  - Server decrypts before storage.
  - Include nonce + tag with each message.

### At-Rest (Server)
- Optional: encrypt stored JPEGs with AES-256-GCM using server-side key.
- Key management: environment variable or KMS integration (out of scope for v0).

## 4. Implementation Order
1. Encryption config + TLS enforcement (foundation).
2. File read protocol (lower risk than keylogging).
3. Keylog daemon (last, requires highest scrutiny).

## 5. Compliance & Ethics Checklist
- [ ] Legal review: keylogging requires explicit consent in most jurisdictions.
- [ ] Audit logging: log every keylog enable/disable, every file read request.
- [ ] Opt-in flags: all advanced features default to `false`.
- [ ] Documentation: clear warning labels in README and config comments.
