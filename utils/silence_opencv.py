"""Utilities to suppress OpenCV and libjpeg warnings (cross-version safe)."""

from __future__ import annotations

import os
import sys
import contextlib
import warnings
from typing import Generator, Optional


@contextlib.contextmanager
def suppress_stderr() -> Generator[None, None, None]:
    """Temporarily redirect stderr to suppress libjpeg/OpenCV warnings."""
    save_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = save_stderr


@contextlib.contextmanager
def suppress_all_warnings() -> Generator[None, None, None]:
    """Temporarily suppress both stderr and Python warnings."""
    save_stderr = sys.stderr
    save_stdout = sys.stdout
    
    # Also suppress Python warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with open(os.devnull, "w") as devnull:
                sys.stderr = devnull
                # Don't suppress stdout as it may contain important output
                yield
        finally:
            sys.stderr = save_stderr
            sys.stdout = save_stdout


def _detect_opencv_logging_api():
    """
    Return a tuple of (get_level, set_level, silent_level) callables/values if available.
    Works across different OpenCV builds where either:
      - cv2.utils.logging.{getLogLevel,setLogLevel,LOG_LEVEL_*} exists, or
      - cv2.setLogLevel / cv2.getLogLevel / cv2.LOG_LEVEL_* exist.
    Otherwise returns (None, None, None).
    """
    try:
        import cv2  # type: ignore

        # Newer style: cv2.utils.logging
        utils = getattr(cv2, "utils", None)
        logging_mod = getattr(utils, "logging", None)

        if logging_mod is not None:
            get_level = getattr(logging_mod, "getLogLevel", None)
            set_level = getattr(logging_mod, "setLogLevel", None)
            silent = getattr(logging_mod, "LOG_LEVEL_SILENT", None)
            if callable(set_level) and silent is not None:
                return get_level, set_level, silent

        # Older / alternate style: top-level functions/constants
        get_level = getattr(cv2, "getLogLevel", None)
        set_level = getattr(cv2, "setLogLevel", None)
        silent = getattr(cv2, "LOG_LEVEL_SILENT", None)
        if callable(set_level) and silent is not None:
            return get_level, set_level, silent

    except Exception:
        pass

    return None, None, None


@contextlib.contextmanager
def suppress_opencv_warnings() -> Generator[None, None, None]:
    """
    Temporarily suppress OpenCV warnings/errors in a cross-version way.
    Also suppresses libjpeg warnings like "Invalid SOS parameters for sequential JPEG".
    Falls back to redirecting stderr if logging API is unavailable.
    """
    get_level, set_level, silent_level = _detect_opencv_logging_api()
    previous_level: Optional[int] = None
    used_logging_api = False

    try:
        if set_level is not None and silent_level is not None:
            # Save current level if possible
            if callable(get_level):
                try:
                    previous_level = get_level()  # type: ignore[misc]
                except Exception:
                    previous_level = None
            # Set to silent
            try:
                set_level(silent_level)  # type: ignore[misc]
                used_logging_api = True
            except Exception:
                used_logging_api = False

        # ALWAYS use stderr suppression for libjpeg warnings
        # Even if OpenCV logging API worked, libjpeg warnings go directly to stderr
        with suppress_stderr():
            yield

    finally:
        # Restore previous logging level if we changed it
        if used_logging_api and previous_level is not None and callable(set_level):
            try:
                set_level(previous_level)  # type: ignore[misc]
            except Exception:
                pass


def setup_opencv_silence() -> None:
    """
    Set up global OpenCV silence (call once at startup).
    Uses logging API if available, otherwise sets env var and relies on stderr redirection later.

    Also disables OpenCV's internal thread scheduling and OpenCL to avoid extra noise/overhead.
    """
    # Hint OpenCV (>=4.5) via env var regardless of runtime API availability
    # Accepted values include: "SILENT", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    
    # Additional environment variables to suppress libjpeg warnings
    os.environ.setdefault("OPENCV_IO_ENABLE_JASPER", "0")  # Disable Jasper codec warnings
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")  # Disable OpenEXR warnings
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")  # Disable video warnings on Windows

    try:
        import cv2  # type: ignore

        # Prefer real logging API if present
        _get, _set, _silent = _detect_opencv_logging_api()
        if _set is not None and _silent is not None:
            try:
                _set(_silent)  # type: ignore[misc]
            except Exception:
                pass

        # Optional: reduce OpenCV internal threading / OpenCL side-effects
        try:
            # Let your own executors control parallelism
            if hasattr(cv2, "setNumThreads"):
                cv2.setNumThreads(0)
        except Exception:
            pass

        try:
            # Avoid OpenCL scheduling overhead if you don't rely on it
            if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
                cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

    except Exception:
        # cv2 not importable — nothing else to do
        pass
