"""Screenshot capture worker for flashback."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    mss = None  # type: ignore

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    pyautogui = None  # type: ignore

from flashback.core.screen_lock import is_screen_locked
from flashback.workers.base import IntervalWorker


class ScreenshotBackend(ABC):
    """Abstract base class for screenshot backends."""

    @abstractmethod
    def capture(self) -> Image.Image:
        """Capture a screenshot and return a PIL Image."""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass


class MssBackend(ScreenshotBackend):
    """Screenshot backend using mss library."""

    def __init__(self):
        if not HAS_MSS:
            raise RuntimeError("mss not installed. Run: pip install mss")
        self.sct = mss.mss()

    def capture(self) -> Image.Image:
        """Capture screenshot using mss."""
        monitor = self.sct.monitors[1]  # Primary monitor
        screenshot = self.sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    def close(self):
        """Close mss instance."""
        if self.sct:
            self.sct.close()


class PyAutoGuiBackend(ScreenshotBackend):
    """Screenshot backend using pyautogui library."""

    def __init__(self):
        if not HAS_PYAUTOGUI:
            raise RuntimeError("pyautogui not installed. Run: pip install pyautogui")

    def capture(self) -> Image.Image:
        """Capture screenshot using pyautogui."""
        return pyautogui.screenshot()

    def close(self):
        """No cleanup needed for pyautogui."""
        pass


class ScreenshotWorker(IntervalWorker):
    """Captures screenshots at regular intervals (runs in separate process)."""

    def __init__(self, config_path=None, db_path=None):
        # Don't initialize backend here - do it in _init_resources
        self._config_path = config_path
        self._db_path = db_path
        self._interval: Optional[int] = None
        self._quality: Optional[int] = None
        self._backend: Optional[ScreenshotBackend] = None
        self._backend_name: Optional[str] = None
        super().__init__(interval_seconds=60, config_path=config_path, db_path=db_path)

    def _init_resources(self):
        """Initialize resources in child process."""
        super()._init_resources()

        # Now we can access self.config
        self._interval = self.config.screenshot_interval
        self._quality = self.config.get("screenshot.quality", 85)
        self.interval_seconds = self._interval

        # Get configured backend
        self._backend_name = self.config.get("screenshot.backend", "mss")
        self._init_backend()

        self.logger.info(
            f"Screenshot worker initialized "
            f"(backend: {self._backend_name}, interval: {self._interval}s, quality: {self._quality})"
        )

    def _init_backend(self):
        """Initialize the screenshot backend based on configuration."""
        backends = {
            "mss": (HAS_MSS, MssBackend, "mss not installed. Run: pip install mss"),
            "pyautogui": (HAS_PYAUTOGUI, PyAutoGuiBackend, "pyautogui not installed. Run: pip install pyautogui"),
        }

        # Try the configured backend first
        requested = self._backend_name.lower()

        if requested in backends:
            has_lib, backend_class, error_msg = backends[requested]
            if has_lib:
                try:
                    self._backend = backend_class()
                    return
                except Exception as e:
                    self.logger.error(f"Failed to initialize {requested} backend: {e}")
            else:
                self.logger.warning(f"{requested} backend requested but not available: {error_msg}")

        # Fallback: try any available backend
        self.logger.info("Attempting to use fallback backend...")
        for name, (has_lib, backend_class, error_msg) in backends.items():
            if has_lib:
                try:
                    self._backend = backend_class()
                    self._backend_name = name
                    self.logger.info(f"Using fallback backend: {name}")
                    return
                except Exception as e:
                    self.logger.error(f"Failed to initialize fallback {name}: {e}")

        # No backend available
        raise RuntimeError(
            "No screenshot backend available. "
            "Install one of: mss (pip install mss), pyautogui (pip install pyautogui)"
        )

    def run_iteration(self):
        """Capture a single screenshot."""
        # Check if screen is locked and skip if configured
        if self.config.get("screenshot.no_screenshot_on_locked_screen", True):
            if is_screen_locked():
                self.logger.debug("Screen is locked, skipping screenshot")
                return
            else:
                self.logger.debug("Screen is not locked, continuing")
        else:
            self.logger.debug("User does not require screenlock detection")

        timestamp = datetime.now()
        timestamp_float = timestamp.timestamp()
        filename = timestamp.strftime("%Y%m%d_%H%M%S.png")
        filepath = self.config.screenshot_dir / filename

        self.logger.debug(f"Capturing screenshot: {filename}")

        try:
            # Capture screenshot using backend
            img = self._backend.capture()

            # Save with compression
            img.save(filepath, "PNG", optimize=True)

            # Record in database
            self.db.insert_screenshot(timestamp_float, str(filepath))
            self.logger.info(f"Captured: {filename}")
            self.logger.debug(f"Screenshot saved to: {filepath}")
        except Exception as e:
            self.logger.exception(f"Failed to capture screenshot: {e}")

    def stop(self):
        """Stop the worker and cleanup."""
        super().stop()
        if self._backend:
            self._backend.close()
            if hasattr(self, 'logger'):
                self.logger.debug(f"Screenshot backend ({self._backend_name}) closed")
