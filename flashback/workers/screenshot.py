"""Screenshot capture worker for flashback."""

from datetime import datetime
from pathlib import Path

from PIL import Image

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    mss = None  # type: ignore

from flashback.core.logger import get_logger
from flashback.workers.base import IntervalWorker

logger = get_logger("workers.screenshot")


class ScreenshotWorker(IntervalWorker):
    """Captures screenshots at regular intervals."""

    def __init__(self, **kwargs):
        config = kwargs.get('config')
        interval = config.screenshot_interval if config else 60
        super().__init__(interval_seconds=interval, **kwargs)
        self.quality = self.config.get("screenshot.quality", 85)

        if not HAS_MSS:
            raise RuntimeError("mss not installed. Run: pip install mss")

        self.sct = mss.mss()
        logger.info(f"Screenshot worker initialized (interval: {interval}s, quality: {self.quality})")

    def run_iteration(self):
        """Capture a single screenshot."""
        timestamp = datetime.now()
        timestamp_float = timestamp.timestamp()
        filename = timestamp.strftime("%Y%m%d_%H%M%S.png")
        filepath = self.config.screenshot_dir / filename

        logger.debug(f"Capturing screenshot: {filename}")

        try:
            # Capture screenshot
            monitor = self.sct.monitors[1]  # Primary monitor
            screenshot = self.sct.grab(monitor)

            # Save with compression
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            img.save(filepath, "PNG", optimize=True)

            # Record in database
            self.db.insert_screenshot(timestamp_float, str(filepath))
            logger.info(f"Captured: {filename}")
            logger.debug(f"Screenshot saved to: {filepath}")
        except Exception as e:
            logger.exception(f"Failed to capture screenshot: {e}")

    def stop(self):
        """Stop the worker and cleanup."""
        super().stop()
        if hasattr(self, 'sct'):
            self.sct.close()
            logger.debug("MSS closed")
