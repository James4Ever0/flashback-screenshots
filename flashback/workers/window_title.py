"""Window title tracking worker for flashback."""

import time
import shutil
from typing import Optional
import platform
import subprocess
import copy

from flashback.workers.base import IntervalWorker

# Platform-specific imports
HAS_XDOTOOL = False
HAS_XLIB = False
HAS_PYGETWINDOW = False

if platform.system() == "Linux":
    HAS_XDOTOOL = shutil.which("xdotool") is not None
    try:
        import Xlib.display
        import Xlib.error
        import Xlib.X

        HAS_XLIB = True
    except ImportError:
        HAS_XLIB = False
elif platform.system() == "Windows":
    try:
        import pygetwindow

        HAS_PYGETWINDOW = True
    except ImportError:
        HAS_PYGETWINDOW = False


class QueuedList:
    def __init__(self, size:int):
        self._size=size
        self._list = []
    def enque(self, item):
        self._list.append(item)
        while len(self._list) > self._size:
            self._list.pop(0)
    def dump(self):
        return copy.deepcopy(self._list)
    def deque_filo(self):
        return self._list.pop(-1)
    def deque_fifo(self):
        return self._list.pop(0)

class WindowTitleWorker(IntervalWorker):
    """Tracks the active window title (runs in separate process)."""

    def __init__(self, config_path=None, db_path=None):
        # Start with default interval, will be updated in _init_resources
        super().__init__(interval_seconds=1, config_path=config_path, db_path=db_path)
        self.last_window_title: Optional[str] = None
        self.last_screenshot_timestamp: Optional[float] = None
        self._warned_missing_deps = False
        self._platform = None
        self._queued_titles = QueuedList(100)

    def _init_resources(self):
        """Initialize resources in child process."""
        super()._init_resources()
        self.config._set_linux_xorg_display_env()

        poll_interval = self.config.get("workers.window_title.poll_interval_seconds", 1)
        self.interval_seconds = poll_interval
        self.max_screenshot_age = self.config.get(
            "workers.window_title.max_screenshot_age_seconds", 3
        )

        current_platform = platform.system()
        self._platform = current_platform

        # Check if we have any supported method for this platform
        has_support = False
        if current_platform == "Linux":
            has_support = HAS_XDOTOOL or HAS_XLIB
        elif current_platform == "Windows":
            has_support = HAS_PYGETWINDOW
        else:
            has_support = False  # Other platforms not supported

        if not has_support and not self._warned_missing_deps:
            self._warned_missing_deps = True
            print(
                f"[WARN] Window title capture disabled on {current_platform}: missing platform dependency"
            )
            if current_platform == "Linux":
                print("[INFO] To enable window titles on Linux, you can:")
                print("[INFO]   1. Install xdotool: sudo apt install xdotool (or equivalent)")
                print("[INFO]   2. Install python-xlib: pip install python-xlib")
                print(
                    "[INFO]   3. Or install the Linux extras: pip install flashback-screenshot[linux]"
                )
            elif current_platform == "Windows":
                print(
                    "[INFO] To enable window titles on Windows, install: pip install flashback-screenshot[windows]"
                )
                print("[INFO] Or: pip install pygetwindow pywin32")
            else:
                print(f"[INFO] Window title capture not supported on {current_platform}")

    def update_last_window_title(self):
        title = self.get_active_window_title()

        if not title:
            return

        if title == self.last_window_title:
            return self.last_window_title

        self.last_window_title = title

        return title

    def find_nearest_window_title(self, timestamp:float):
        titles = self._queued_titles.dump()
        if not titles: return
        titles.sort(key=lambda it: abs(it['timestamp'] - timestamp))
        return titles[0]

    def run_iteration(self):
        """Capture current window title and update the latest screenshot."""
        # Find the latest screenshot without a window title
        title = self.update_last_window_title()
        self._queued_titles.enque(dict(title=title, timestamp=time.time()))

        try:
            screenshot = self.db.get_latest_without_window_title()
            if not screenshot:
                self.logger.debug("No screenshot without window title found")
                return

            nearest_title = self.find_nearest_window_title(screenshot.timestamp)
            # Check if screenshot is recent enough
            age_seconds = nearest_title['timestamp'] - screenshot.timestamp
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
        if platform.system() == "Linux":
            if HAS_XLIB:
                title = self._get_x11_window_title()
                if title:
                    return title
            if HAS_XDOTOOL:
                title = self._get_xdotool_window_title()
                if title:
                    return title
        elif platform.system() == "Windows" and HAS_PYGETWINDOW:
            return self._get_windows_window_title()
        return None

    def _get_xdotool_window_title(self) -> Optional[str]:
        """Get active window title using xdotool."""
        try:
            active_window_id = subprocess.check_output(["xdotool", "getactivewindow"], timeout=2)
            active_window_id = active_window_id.decode("utf-8").strip()
            try:
                int(active_window_id)
            except ValueError:
                self.logger.debug(f"Invalid active window ID: {active_window_id}")
                return None
            result = subprocess.run(
                ["xdotool", "getwindowname", active_window_id],
                capture_output=True,
                text=True,
                shell=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        return None

    def _assign_display_failsafe_close(self, display: Xlib.display.Display):
        display_close = copy.copy(display.close)
        def display_close_failsafe():
            try:
                display_close()
            except Exception as e:
                self.logger.debug(f"Failed to close display with exception: {e}")
        setattr(display, "close", display_close_failsafe)
        return display

    def _get_x11_window_title(self) -> Optional[str]:
        """Get active window title on X11 using _NET_ACTIVE_WINDOW property."""
        display = None
        try:
            display = Xlib.display.Display()
            if display:
                display = self._assign_display_failsafe_close(display)
            root = display.screen().root

            # Get _NET_ACTIVE_WINDOW property
            net_active_window = display.intern_atom("_NET_ACTIVE_WINDOW")
            net_wm_name = display.intern_atom("_NET_WM_NAME")

            # Get active window ID
            window_id_prop = root.get_full_property(net_active_window, Xlib.X.AnyPropertyType)
            if not window_id_prop:
                # Fallback to old method
                display.close()
                return self._get_x11_window_title_fallback()

            active_window_id = window_id_prop.value[0]
            if not active_window_id:
                display.close()
                return self._get_x11_window_title_fallback()

            # Get window object
            window_obj = display.create_resource_object("window", active_window_id)

            # Get window name
            try:
                name_prop = window_obj.get_full_property(net_wm_name, 0)
                if name_prop:
                    window_name = name_prop.value
                    if isinstance(window_name, bytes):
                        window_name = window_name.decode("utf-8", errors="ignore")
                    display.close()
                    return window_name or "Unknown"
            except Xlib.error.XError:
                pass

            # Fallback to WM_NAME
            wm_name = window_obj.get_wm_name()
            if wm_name:
                if isinstance(wm_name, bytes):
                    wm_name = wm_name.decode("utf-8", errors="ignore")
                display.close()
                return wm_name or "Unknown"

            display.close()
            return "Unknown"
        except Exception:
            if display:
                display.close()
            # Fallback to old method
            return self._get_x11_window_title_fallback()

    def _get_x11_window_title_fallback(self) -> Optional[str]:
        """Fallback method using get_input_focus (less reliable)."""
        display = None
        try:
            display = Xlib.display.Display()

            if display:
                display = self._assign_display_failsafe_close(display)

            window = display.get_input_focus().focus
            if window:
                wm_name = window.get_wm_name()
                display.close()
                if wm_name:
                    if isinstance(wm_name, bytes):
                        wm_name = wm_name.decode("utf-8", errors="ignore")
                    return wm_name or "Unknown"
            if display:
                display.close()
        except Exception:
            if display:
                display.close()
        return None

    def _get_windows_window_title(self) -> Optional[str]:
        """Get active window title on Windows."""
        try:
            window = pygetwindow.getActiveWindow()
            return window.title if window else None
        except Exception:
            return None
