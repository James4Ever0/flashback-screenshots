"""Window title tracking worker for flashback."""

import time
from typing import Optional

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
    """Tracks the active window title (runs in separate process)."""

    def __init__(self, config_path=None, db_path=None):
        # Start with default interval, will be updated in _init_resources
        super().__init__(interval_seconds=1, config_path=config_path, db_path=db_path)
        self.last_window_title: Optional[str] = None
        self.last_screenshot_timestamp: Optional[float] = None
        self._warned_missing_deps = False
        self._platform = None

    def _init_resources(self):
        """Initialize resources in child process."""
        super()._init_resources()

        poll_interval = self.config.get("workers.window_title.poll_interval_seconds", 1)
        self.interval_seconds = poll_interval
        self.max_screenshot_age = self.config.get(
            "workers.window_title.max_screenshot_age_seconds", 30
        )

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
        """Capture current window title and update the latest screenshot."""
        title = self.get_active_window_title()

        if not title:
            return

        if title == self.last_window_title:
            return

        self.last_window_title = title

        # Find the latest screenshot without a window title
        try:
            screenshot = self.db.get_latest_without_window_title()
            if not screenshot:
                self.logger.debug("No screenshot without window title found")
                return

            # Check if screenshot is recent enough
            age_seconds = time.time() - screenshot.timestamp
            if age_seconds > self.max_screenshot_age:
                self.logger.debug(
                    f"Screenshot too old ({age_seconds:.1f}s > {self.max_screenshot_age}s), "
                    f"skipping window title update"
                )
                return

            # Update that specific screenshot with the window title
            self.db.update_window_title(screenshot.timestamp, title)
            self.logger.info(
                f"Updated screenshot {screenshot.timestamp_formatted} (age: {age_seconds:.1f}s) "
                f"with window title: {title[:60]}..."
            )

        except Exception as e:
            self.logger.exception(f"Failed to update window title: {e}")

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
