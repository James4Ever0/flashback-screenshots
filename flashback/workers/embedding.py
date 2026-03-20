"""Embedding worker for flashback."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from flashback.core.database import ScreenshotRecord
from flashback.core.embedding_client import EmbeddingAPIClient
from flashback.core.logger import get_logger, trace_entry_exit, timed
from flashback.workers.base import QueueWorker

logger = get_logger("workers.embedding")


class EmbeddingWorker(QueueWorker):
    """Generates embeddings using OpenAI-compatible API."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.poll_interval = self.config.get(
            "workers.embedding.work_interval_seconds", 1
        )
        self.batch_size = self.config.get("workers.embedding.batch_size", 3)

        # Get embedding mode
        self.mode = self.config.get_embedding_mode()
        logger.info(f"Initializing embedding worker (mode: {self.mode})")

        # Initialize clients based on mode
        self.text_client: Optional[EmbeddingAPIClient] = None
        self.image_client: Optional[EmbeddingAPIClient] = None

        if self.mode in ("text-only", "text-image-hybrid"):
            text_config = self.config.get_text_embedding_config()
            if text_config.get("model"):
                logger.info(f"Setting up text embedding client (model: {text_config['model']})")
                self.text_client = EmbeddingAPIClient(
                    base_url=text_config.get("base_url", "https://api.openai.com/v1"),
                    api_key=text_config.get("api_key", ""),
                    model=text_config["model"],
                    dimension=text_config.get("dimension"),
                    extra_headers=text_config.get("extra_headers", {}),
                    name="text",
                )
                logger.debug(f"Text client configured: {self.text_client.base_url}")

        if self.mode in ("image-only", "text-image-hybrid"):
            image_config = self.config.get_image_embedding_config()
            if image_config.get("model"):
                logger.info(f"Setting up image embedding client (model: {image_config['model']})")
                self.image_client = EmbeddingAPIClient(
                    base_url=image_config.get("base_url", "http://localhost:11434/v1"),
                    api_key=image_config.get("api_key", ""),
                    model=image_config["model"],
                    dimension=image_config.get("dimension"),
                    extra_headers=image_config.get("extra_headers", {}),
                    name="image",
                )
                logger.debug(f"Image client configured: {self.image_client.base_url}")

        if self.text_client:
            logger.info(f"Text API: {self.text_client.base_url} (model: {self.text_client.model})")
        if self.image_client:
            logger.info(f"Image API: {self.image_client.base_url} (model: {self.image_client.model})")

    def run(self):
        """Validate configuration before starting."""
        # Validate dimensions are set
        if self.text_client and self.text_client.dimension is None:
            print(f"[{self.name}] WARNING: Text embedding dimension not configured.")
            print(f"[{self.name}] Run: flashback config test-embedding --type text --write")

        if self.image_client and self.image_client.dimension is None:
            print(f"[{self.name}] WARNING: Image embedding dimension not configured.")
            print(f"[{self.name}] Run: flashback config test-embedding --type image --write")

        super().run()

    def get_items(self) -> list:
        """Get screenshots without embeddings."""
        logger.debug(f"Fetching items (mode: {self.mode})")
        # Get mode-specific unprocessed items
        if self.mode == "text-only":
            items = self.db.get_unprocessed_text_embeddings(limit=self.batch_size * 2)
        elif self.mode == "image-only":
            items = self.db.get_unprocessed_image_embeddings(limit=self.batch_size * 2)
        else:  # text-image-hybrid
            items = self.db.get_unprocessed_embeddings(limit=self.batch_size * 2)

        logger.debug(f"Found {len(items)} items to process")
        return items

    @timed("workers.embedding")
    def process_item(self, item: ScreenshotRecord):
        """Generate embedding(s) for a screenshot."""
        screenshot_path = item.screenshot_path
        timestamp = item.timestamp
        logger.info(f"Processing: {Path(screenshot_path).stem}")
        logger.debug(f"Timestamp: {timestamp}, Has OCR: {bool(item.ocr_text)}")

        try:
            if self.mode == "text-only":
                self._process_text_only(item)
            elif self.mode == "image-only":
                self._process_image_only(item)
            else:  # text-image-hybrid
                self._process_hybrid(item)

            logger.info(f"Successfully processed: {Path(screenshot_path).stem}")

        except Exception as e:
            logger.exception(f"Failed to process {screenshot_path}: {e}")

    def _process_text_only(self, item: ScreenshotRecord):
        """Generate text embedding from OCR content."""
        if not self.text_client:
            logger.debug("No text client configured, skipping")
            return

        if not item.ocr_text:
            # No OCR text available, mark as processed with empty embedding
            logger.debug(f"No OCR text for {item.timestamp}, marking as processed")
            self.db.update_text_embedding(item.timestamp, None)
            return

        logger.debug(f"Generating text embedding for {item.timestamp}")
        # Generate text embedding from OCR
        embedding = self.text_client.get_text_embedding(item.ocr_text)
        logger.debug(f"Generated embedding shape: {embedding.shape}")

        # Save embedding
        emb_filename = f"{item.timestamp}_text.npy"
        emb_path = self.config.embedding_dir / emb_filename
        np.save(emb_path, embedding)
        logger.debug(f"Saved text embedding to {emb_path}")

        # Update database
        self.db.update_text_embedding(item.timestamp, str(emb_path))

    def _process_image_only(self, item: ScreenshotRecord):
        """Generate image embedding from screenshot pixels."""
        if not self.image_client:
            logger.debug("No image client configured, skipping")
            return

        logger.debug(f"Generating image embedding for {item.timestamp}")
        # Generate image embedding
        image = Image.open(item.screenshot_path)
        embedding = self.image_client.get_image_embedding(image)
        logger.debug(f"Generated embedding shape: {embedding.shape}")

        # Save embedding
        emb_filename = f"{item.timestamp}_image.npy"
        emb_path = self.config.embedding_dir / emb_filename
        np.save(emb_path, embedding)
        logger.debug(f"Saved image embedding to {emb_path}")

        # Update database
        self.db.update_image_embedding(item.timestamp, str(emb_path))

    def _process_hybrid(self, item: ScreenshotRecord):
        """Generate both text and image embeddings."""
        # Process text embedding
        if self.text_client and item.ocr_text:
            try:
                logger.debug(f"Generating text embedding for {item.timestamp}")
                text_embedding = self.text_client.get_text_embedding(item.ocr_text)
                text_emb_filename = f"{item.timestamp}_text.npy"
                text_emb_path = self.config.embedding_dir / text_emb_filename
                np.save(text_emb_path, text_embedding)
                self.db.update_text_embedding(item.timestamp, str(text_emb_path))
                logger.debug(f"Text embedding saved: {text_emb_path}")
            except Exception as e:
                logger.exception(f"Text embedding failed: {e}")

        # Process image embedding
        if self.image_client:
            try:
                logger.debug(f"Generating image embedding for {item.timestamp}")
                image = Image.open(item.screenshot_path)
                image_embedding = self.image_client.get_image_embedding(image)
                image_emb_filename = f"{item.timestamp}_image.npy"
                image_emb_path = self.config.embedding_dir / image_emb_filename
                np.save(image_emb_path, image_embedding)
                self.db.update_image_embedding(item.timestamp, str(image_emb_path))
                logger.debug(f"Image embedding saved: {image_emb_path}")
            except Exception as e:
                logger.exception(f"Image embedding failed: {e}")
