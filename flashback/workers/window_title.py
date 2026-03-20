"""Window title tracking worker for flashback."""

from datetime import datetime
from typing import Optional

from flashback.core.config import Config
from flashback.workers.base import IntervalWorker

# Platform-specific imports
try:
    import Xlib.display
    HAS_XLIB = True
except ImportError:
    HAS_XLIB = False

try:
    import pygetwindow
    HAS_PYGETWINDOW = True
except ImportError:
    HAS_PYGETWINDOW = False


class WindowTitleWorker(IntervalWorker):
    """Tracks the active window title."""

    def __init__(self, **kwargs):
        poll_interval = kwargs.get('config', Config()).get(
            "workers.window_title.poll_interval_seconds", 1
        ) if 'config' in kwargs else 1
        super().__init__(interval_seconds=poll_interval, **kwargs)
        self.last_window_title: Optional[str] = None
        self.last_screenshot_timestamp: Optional[float] = None
        self._warned_missing_deps = False

        if not HAS_XLIB and not HAS_PYGETWINDOW:
            import platform
            self._platform = platform.system()
            self._warned_missing_deps = True
            print("[WARN] Window title capture disabled: missing platform dependency")
            if self._platform == "Linux":
                print("[INFO] To enable window titles on Linux, install: pip install flashback-screenshots[linux]")
                print("[INFO] Or: pip install python-xlib")
            elif self._platform == "Windows":
                print("[INFO] To enable window titles on Windows, install: pip install flashback-screenshots[windows]")
                print("[INFO] Or: pip install pygetwindow pywin32")
            else:
                print(f"[INFO] Window title capture not supported on {self._platform}")

    def run_iteration(self):
        """Capture current window title."""
        title = self.get_active_window_title()

        if title and title != self.last_window_title:
            self.last_window_title = title

            # Update the most recent screenshot with window title
            # We do this by getting the latest screenshot and updating it
            timestamp = datetime.now().timestamp()
            self.db.update_window_title(timestamp, title)
            print(f"[{self.name}] Window: {title[:60]}...")

    def get_active_window_title(self) -> Optional[str]:
        """Get the active window title (platform-specific)."""
        if HAS_XLIB:
            return self._get_x11_window_title()
        elif HAS_PYGETWINDOW:
            return self._get_windows_window_title()
        return None

    def _get_x11_window_title(self) -> Optional[str]:
        """Get active window title on X11."""
        try:
            display = Xlib.display.Display()
            window = display.get_input_focus().focus
            if window:
                wm_name = window.get_wm_name()
                display.close()
                return wm_name or "Unknown"
            display.close()
        except Exception:
            pass
        return None

    def _get_windows_window_title(self) -> Optional[str]:
        """Get active window title on Windows."""
        try:
            window = pygetwindow.getActiveWindow()
            return window.title if window else None
        except Exception:
            return None
