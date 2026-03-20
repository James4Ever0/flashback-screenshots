"""Base worker class for flashback."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from flashback.core.config import Config
from flashback.core.database import Database
from flashback.core.logger import get_logger, trace_entry_exit


class BaseWorker(threading.Thread, ABC):
    """Base class for all background workers."""

    def __init__(self, config: Optional[Config] = None, db: Optional[Database] = None):
        super().__init__(daemon=True)
        self.config = config or Config()
        self.db = db or Database(self.config.db_path)
        self.running = False
        self._stop_event = threading.Event()
        self.name = self.__class__.__name__
        self.logger = get_logger(f"workers.{self.__class__.__name__.lower()}")

    def stop(self):
        """Signal the worker to stop."""
        self.running = False
        self._stop_event.set()
        self.logger.debug("Stop requested")

    def should_stop(self, timeout: Optional[float] = None) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.wait(timeout=timeout)

    @abstractmethod
    def run_iteration(self):
        """Perform one iteration of work. Override in subclasses."""
        pass

    def run(self):
        """Main worker loop."""
        self.running = True
        self.logger.info("Started")
        self.logger.debug(f"Worker config: {self.config.to_dict()}")

        iteration = 0
        while self.running:
            iteration += 1
            if iteration % 10 == 0:
                self.logger.debug(f"Running iteration {iteration}")

            try:
                self.run_iteration()
            except Exception as e:
                self.logger.exception(f"Error in iteration {iteration}: {e}")
                time.sleep(5)  # Back off on error

        self.logger.info("Stopped")

    def get_sleep_interval(self) -> float:
        """Get the sleep interval between iterations. Override in subclasses."""
        return 1.0


class IntervalWorker(BaseWorker, ABC):
    """Worker that runs at fixed intervals."""

    def __init__(self, interval_seconds: float, **kwargs):
        super().__init__(**kwargs)
        self.interval_seconds = interval_seconds

    def run(self):
        """Main worker loop with interval timing."""
        self.running = True
        self.logger.info(f"Started (interval: {self.interval_seconds}s)")

        iteration = 0
        while self.running:
            iteration += 1
            start_time = time.time()

            try:
                if self.logger.isEnabledFor(logging.DEBUG) and iteration % 10 == 0:
                    self.logger.debug(f"Iteration {iteration}")
                self.run_iteration()
            except Exception as e:
                self.logger.exception(f"Error in iteration {iteration}: {e}")

            # Sleep until next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval_seconds - elapsed)

            if sleep_time > 0 and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Sleeping for {sleep_time:.2f}s")

            if self._stop_event.wait(timeout=sleep_time):
                break

        self.logger.info("Stopped")


class QueueWorker(BaseWorker, ABC):
    """Worker that processes items from a queue."""

    def __init__(self, poll_interval: float = 1.0, batch_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.poll_interval = poll_interval
        self.batch_size = batch_size

    def run(self):
        """Main worker loop for queue processing."""
        self.running = True
        self.logger.info(f"Started (poll: {self.poll_interval}s, batch: {self.batch_size})")

        iteration = 0
        while self.running:
            iteration += 1
            try:
                self.logger.debug(f"Fetching items (iteration {iteration})")
                items = self.get_items()

                if not items:
                    self.logger.debug("No items, waiting...")
                    if self._stop_event.wait(timeout=self.poll_interval):
                        break
                    continue

                self.logger.debug(f"Processing {len(items)} items")
                for i, item in enumerate(items[: self.batch_size]):
                    if not self.running:
                        break
                    try:
                        self.logger.debug(f"Processing item {i+1}/{min(len(items), self.batch_size)}")
                        self.process_item(item)
                    except Exception as e:
                        self.logger.exception(f"Failed to process item {i}: {e}")

            except Exception as e:
                self.logger.exception(f"Error in iteration {iteration}: {e}")
                time.sleep(5)

        self.logger.info("Stopped")

    @abstractmethod
    def get_items(self) -> list:
        """Get items to process. Override in subclasses."""
        return []

    @abstractmethod
    def process_item(self, item: Any):
        """Process a single item. Override in subclasses."""
        pass
