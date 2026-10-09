"""
Single-instance guard for the pipeline.

The scheduled task and a manual `python run.py` must never run together: both write the same shadow
file (one deletes it while the other has it open). The lock is an OS-level file lock, so it disappears
by itself when the process dies — no stale lock files to clean up.
"""
import os
import sys
from pathlib import Path


class AlreadyRunning(RuntimeError):
    """Another pipeline run holds the lock."""


class RunLock:
    def __init__(self, path: str):
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            fh.close()
            raise AlreadyRunning(f"another ETL run holds {self.path.name}") from e
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
