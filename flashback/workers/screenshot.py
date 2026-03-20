"""Screenshot capture worker for flashback."""

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

from flashback.workers.base import IntervalWorker


class ScreenshotWorker(IntervalWorker):
    """Captures screenshots at regular intervals (runs in separate process)."""

    def __init__(self, config_path=None, db_path=None):
        # Don't initialize mss here - do it in run_iteration
        # Store paths for later initialization
        self._config_path = config_path
        self._db_path = db_path
        self._interval: Optional[int] = None
        self._quality: Optional[int] = None
        self.sct = None
        super().__init__(interval_seconds=60, config_path=config_path, db_path=db_path)

    def _init_resources(self):
        """Initialize resources in child process."""
        super()._init_resources()

        # Now we can access self.config
        self._interval = self.config.screenshot_interval
        self._quality = self.config.get("screenshot.quality", 85)
        self.interval_seconds = self._interval

        if not HAS_MSS:
            raise RuntimeError("mss not installed. Run: pip install mss")

        # Initialize mss in the child process (not in parent)
        self.sct = mss.mss()
        self.logger.info(f"Screenshot worker initialized (interval: {self._interval}s, quality: {self._quality})")

    def run_iteration(self):
        """Capture a single screenshot."""
        timestamp = datetime.now()
        timestamp_float = timestamp.timestamp()
        filename = timestamp.strftime("%Y%m%d_%H%M%S.png")
        filepath = self.config.screenshot_dir / filename

        self.logger.debug(f"Capturing screenshot: {filename}")

        try:
            # Capture screenshot
            self.logger.debug(f"Available monitor count: {len(self.sct.monitors)}")
            monitor = self.sct.monitors[1]  # Primary monitor
            self.logger.debug(f"Monitor selected: {monitor}")
            screenshot = self.sct.grab(monitor)

            # Save with compression
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
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
        if self.sct:
            self.sct.close()
            if hasattr(self, 'logger'):
                self.logger.debug("MSS closed")
