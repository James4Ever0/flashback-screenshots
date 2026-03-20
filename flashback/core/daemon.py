"""Daemon process management for flashback."""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

import psutil

from flashback.core.paths import get_log_dir


class DaemonError(Exception):
    """Error managing daemon process."""

    pass


class DaemonManager:
    """Manages background daemon processes."""

    def __init__(self, name: str, pid_dir: Optional[Path] = None):
        self.name = name
        self.pid_dir = pid_dir or (Path.home() / ".local" / "state" / "flashback")
        self.pid_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.pid_dir / f"{name}.pid"
        self.log_file = get_log_dir() / f"{name}.log"

    def get_pid(self) -> Optional[int]:
        """Get PID from pidfile if process is running."""
        if not self.pid_file.exists():
            return None

        try:
            with open(self.pid_file, "r") as f:
                pid = int(f.read().strip())

            # Check if process exists
            if psutil.pid_exists(pid):
                return pid
            else:
                # Stale pidfile
                self.pid_file.unlink()
                return None
        except (ValueError, FileNotFoundError):
            return None

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self.get_pid() is not None

    def write_pid(self, pid: int):
        """Write PID to pidfile."""
        self.pid_file.write_text(str(pid))

    def remove_pid(self):
        """Remove pidfile."""
        if self.pid_file.exists():
            self.pid_file.unlink()

    def stop(self, force: bool = False) -> bool:
        """Stop the daemon process."""
        pid = self.get_pid()
        if pid is None:
            return False

        try:
            process = psutil.Process(pid)

            if force:
                process.kill()
            else:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait()

            self.remove_pid()
            return True
        except psutil.NoSuchProcess:
            self.remove_pid()
            return False

    def get_log_path(self) -> Path:
        """Get path to log file."""
        return self.log_file

    def read_logs(self, lines: int = 50) -> str:
        """Read recent log entries."""
        if not self.log_file.exists():
            return ""

        try:
            with open(self.log_file, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading logs: {e}"


def daemonize(
    target,
    daemon_manager: DaemonManager,
    *args,
    **kwargs,
) -> int:
    """
    Daemonize a function.

    Returns the PID of the daemon process.
    """
    # Double fork to detach from terminal
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - wait for child to fork again
            _, status = os.waitpid(pid, 0)
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
                # Read the actual daemon PID
                import time

                time.sleep(0.5)
                daemon_pid = daemon_manager.get_pid()
                return daemon_pid or 0
            return 0
    except OSError as e:
        raise DaemonError(f"Fork failed: {e}")

    # First child
    os.chdir("/")
    os.setsid()
    os.umask(0)

    try:
        pid = os.fork()
        if pid > 0:
            # First child exits
            sys.exit(0)
    except OSError as e:
        raise DaemonError(f"Second fork failed: {e}")

    # Grandchild (daemon)
    # Redirect stdout/stderr to log file
    log_file = daemon_manager.get_log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Open log file
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)

    # Close stdin
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, sys.stdin.fileno())
    os.close(devnull)

    # Write PID file
    daemon_manager.write_pid(os.getpid())

    # Run the target function
    try:
        target(*args, **kwargs)
    finally:
        daemon_manager.remove_pid()

    return 0
