# Screenshot Worker — Rust Bootstrap

Simplified Rust bootstrap for the screenshot worker client. Two Linux build variants are provided; macOS and Windows are supported in both.

## Folder Layout

```
rust-bootstrap/
├── Cargo.toml          # workspace root
├── README.md           # this file
├── x11/                # X11-only Linux build (lightweight, minimal deps)
│   ├── Cargo.toml
│   └── README.md
└── universal/          # Universal Linux build (Wayland + X11 via portal/fallback)
    ├── Cargo.toml
    └── README.md
```

## Which One to Use?

| Variant | Linux Display Server | Dependency Weight | Use When |
|---------|---------------------|-------------------|----------|
| `x11/` | X11 only | Light | You run X11 and want the smallest binary/fastest build |
| `universal/` | Wayland + X11 | Medium | You need Wayland support or run a mixed environment |

Both variants share the same source structure; only the Linux capture backend differs.

## Quick Start

```bash
# X11-only build
cd x11
cargo build --release

# Universal build
cd universal
cargo build --release
```

## Cross-Compilation

```bash
# Linux x86_64 (from any host with cross)
cd x11 && cross build --target x86_64-unknown-linux-gnu
cd universal && cross build --target x86_64-unknown-linux-gnu

# macOS (requires macOS host or GitHub Actions)
cd x11 && cargo build --target aarch64-apple-darwin

# Windows (requires Windows host or cross)
cd x11 && cargo build --target x86_64-pc-windows-msvc
```

## Runtime Dependencies

| Variant | Linux Packages Needed |
|---------|----------------------|
| `x11/` | `libxcb1`, `libxrandr2` |
| `universal/` | `xdg-desktop-portal`, `grim` (Wayland), `scrot` (X11 fallback) |

See the individual crate READMEs for full details.
