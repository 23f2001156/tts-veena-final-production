


import logging
import os
import sys
import time
from contextlib import contextmanager
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

# Lock name shared between STT and TTS processes
DEFAULT_LOCK_NAME = "gpu_inference_lock"
DEFAULT_MAX_SLOTS = 2  # Total GPU slots across all processes
DEFAULT_TIMEOUT = 30.0  # seconds


class CrossProcessGPUSemaphore:
    """
    Cross-process GPU semaphore using OS-level primitives.
    
    On Windows: Named Mutex
    On Linux: fcntl file locks (for deployment compatibility)
    
    Args:
        max_slots: Maximum concurrent GPU jobs for THIS process (default: 1)
        lock_name: Shared lock name across processes (default: "gpu_inference_lock")
        timeout: Maximum wait time in seconds (default: 30.0)
    """
    
    def __init__(
        self,
        max_slots: int = 1,
        lock_name: str = DEFAULT_LOCK_NAME,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.max_slots = max_slots
        self.lock_name = lock_name
        self.timeout = timeout
        self._local_lock = Lock()  # Thread-local protection
        self._acquired_count = 0
        
        # Platform-specific initialization
        if sys.platform == "win32":
            self._init_windows()
        else:
            self._init_posix()
        
        logger.info(
            f"Cross-process GPU semaphore initialized: "
            f"max {max_slots} concurrent GPU jobs, lock='{lock_name}'"
        )
    
    def _init_windows(self):
        """Initialize Windows Named Mutex."""
        import ctypes
        from ctypes import wintypes
        
        self._kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        
        # CreateMutexW
        self._kernel32.CreateMutexW.argtypes = [
            wintypes.LPVOID,  # lpMutexAttributes
            wintypes.BOOL,    # bInitialOwner
            wintypes.LPCWSTR  # lpName
        ]
        self._kernel32.CreateMutexW.restype = wintypes.HANDLE
        
        # WaitForSingleObject
        self._kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,  # hHandle
            wintypes.DWORD    # dwMilliseconds
        ]
        self._kernel32.WaitForSingleObject.restype = wintypes.DWORD
        
        # ReleaseMutex
        self._kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self._kernel32.ReleaseMutex.restype = wintypes.BOOL
        
        # CloseHandle
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        
        # Create the mutex (Global\ prefix for cross-session visibility)
        mutex_name = f"Global\\{self.lock_name}"
        self._mutex = self._kernel32.CreateMutexW(None, False, mutex_name)
        
        if not self._mutex:
            error = ctypes.get_last_error()
            raise OSError(f"Failed to create mutex: error code {error}")
        
        logger.debug(f"Windows mutex created: {mutex_name}")
    
    def _init_posix(self):
        """Initialize POSIX file lock (for Linux deployment)."""
        import fcntl
        
        # Use /tmp for lock file
        self._lock_path = f"/tmp/{self.lock_name}.lock"
        self._lock_fd: Optional[int] = None
        self._fcntl = fcntl
        
        # Create lock file if it doesn't exist
        if not os.path.exists(self._lock_path):
            with open(self._lock_path, "w") as f:
                f.write("")
        
        logger.debug(f"POSIX lock file: {self._lock_path}")
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the GPU lock (blocking).
        
        Args:
            timeout: Override default timeout (None = use default)
            
        Returns:
            True if acquired, False if timeout
        """
        timeout = timeout if timeout is not None else self.timeout
        
        # First, acquire thread-local lock
        if not self._local_lock.acquire(timeout=timeout):
            logger.warning("Failed to acquire thread-local lock (timeout)")
            return False
        
        try:
            if sys.platform == "win32":
                return self._acquire_windows(timeout)
            else:
                return self._acquire_posix(timeout)
        except Exception:
            self._local_lock.release()
            raise
    
    def _acquire_windows(self, timeout: float) -> bool:
        """Acquire Windows mutex."""
        WAIT_OBJECT_0 = 0x00000000
        WAIT_TIMEOUT = 0x00000102
        WAIT_ABANDONED = 0x00000080
        
        timeout_ms = int(timeout * 1000)
        result = self._kernel32.WaitForSingleObject(self._mutex, timeout_ms)
        
        if result == WAIT_OBJECT_0 or result == WAIT_ABANDONED:
            self._acquired_count += 1
            logger.debug(f"GPU lock acquired (count={self._acquired_count})")
            return True
        elif result == WAIT_TIMEOUT:
            self._local_lock.release()
            logger.warning(f"GPU lock timeout after {timeout}s")
            return False
        else:
            self._local_lock.release()
            raise OSError(f"WaitForSingleObject failed: {result}")
    
    def _acquire_posix(self, timeout: float) -> bool:
        """Acquire POSIX file lock."""
        start_time = time.time()
        
        self._lock_fd = os.open(self._lock_path, os.O_RDWR)
        
        while True:
            try:
                self._fcntl.flock(self._lock_fd, self._fcntl.LOCK_EX | self._fcntl.LOCK_NB)
                self._acquired_count += 1
                logger.debug(f"GPU lock acquired (count={self._acquired_count})")
                return True
            except (IOError, OSError):
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    os.close(self._lock_fd)
                    self._lock_fd = None
                    self._local_lock.release()
                    logger.warning(f"GPU lock timeout after {timeout}s")
                    return False
                time.sleep(0.01)  # 10ms retry interval
    
    def release(self):
        """Release the GPU lock."""
        try:
            if sys.platform == "win32":
                self._release_windows()
            else:
                self._release_posix()
        finally:
            self._local_lock.release()
    
    def _release_windows(self):
        """Release Windows mutex."""
        if self._acquired_count > 0:
            self._kernel32.ReleaseMutex(self._mutex)
            self._acquired_count -= 1
            logger.debug(f"GPU lock released (count={self._acquired_count})")
    
    def _release_posix(self):
        """Release POSIX file lock."""
        if self._lock_fd is not None:
            self._fcntl.flock(self._lock_fd, self._fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None
            self._acquired_count -= 1
            logger.debug(f"GPU lock released (count={self._acquired_count})")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire GPU lock within {self.timeout}s")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
    
    def __del__(self):
        """Cleanup on destruction."""
        if sys.platform == "win32" and hasattr(self, '_mutex') and self._mutex:
            self._kernel32.CloseHandle(self._mutex)


# Convenience function for creating a singleton instance
_global_gpu_lock: Optional[CrossProcessGPUSemaphore] = None


def get_gpu_lock(max_slots: int = 1) -> CrossProcessGPUSemaphore:
    """
    Get or create a global GPU lock instance.
    
    Args:
        max_slots: Maximum concurrent GPU jobs for this process
        
    Returns:
        CrossProcessGPUSemaphore instance
    """
    global _global_gpu_lock
    if _global_gpu_lock is None:
        _global_gpu_lock = CrossProcessGPUSemaphore(max_slots=max_slots)
    return _global_gpu_lock
