"""OCR worker for flashback."""

from pathlib import Path

from PIL import Image

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None  # type: ignore

from flashback.core.database import ScreenshotRecord
from flashback.core.logger import timed
from flashback.workers.base import QueueWorker


class OCRWorker(QueueWorker):
    """Performs OCR on screenshots using Tesseract (runs in separate process)."""

    def __init__(self, config_path=None, db_path=None):
        self._languages = None
        super().__init__(config_path=config_path, db_path=db_path)

    def _init_resources(self):
        """Initialize resources in child process."""
        super()._init_resources()

        self.poll_interval = self.config.get("workers.ocr.work_interval_seconds", 1)
        self.batch_size = self.config.get("workers.ocr.batch_size", 5)
        self._languages = "+".join(self.config.get_ocr_languages())

        if not HAS_TESSERACT:
            raise RuntimeError(
                "pytesseract not installed. Run: pip install pytesseract"
            )

        self.logger.info(f"OCR worker initialized (languages: {self._languages})")

    def get_items(self) -> list:
        """Get screenshots without OCR."""
        items = self.db.get_unprocessed_ocr(limit=self.batch_size * 2)
        self.logger.debug(f"Found {len(items)} items needing OCR")
        return items

    @timed("workers.ocr")
    def process_item(self, item: ScreenshotRecord):
        """Process a single screenshot with OCR."""
        screenshot_path = item.screenshot_path
        timestamp = item.timestamp
        ocr_filename = Path(screenshot_path).stem + ".txt"

        self.logger.info(f"Processing: {ocr_filename}")
        self.logger.debug(f"Timestamp: {timestamp}, Languages: {self._languages}")

        try:
            # Perform OCR with configured languages
            image = Image.open(screenshot_path)
            text = pytesseract.image_to_string(image, lang=self._languages)

            text_preview = text[:100].replace('\n', ' ') if text else "(empty)"
            self.logger.debug(f"OCR result preview: {text_preview}...")

            # Save OCR result
            ocr_path = self.config.ocr_dir / ocr_filename
            ocr_path.write_text(text, encoding="utf-8")

            # Update database
            self.db.update_ocr(timestamp, str(ocr_path), text)
            self.logger.info(f"Processed: {ocr_filename} ({len(text)} chars)")

        except Exception as e:
            self.logger.exception(f"Failed to process {screenshot_path}: {e}")
