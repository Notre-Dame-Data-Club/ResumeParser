#!/usr/bin/env python3
"""
check_requirements.py — Diagnose Python / GPU / dependency status.
Invoked by run_processor.bat option 4.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

# processing.log lives at the repo root (one level up from scripts/).
LOG_FILE = Path(__file__).resolve().parent.parent / "processing.log"

_file_logger = logging.getLogger("check_requirements")
_file_logger.setLevel(logging.INFO)
_file_logger.propagate = False
_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [check] %(message)s"))
_file_logger.addHandler(_handler)


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def check_python_and_gpu() -> None:
    _print_section("Python & GPU")
    py_ver = sys.version.split()[0]
    print(f"Python Version:    {py_ver}")
    print(f"Python Executable: {sys.executable}")
    _file_logger.info("Python %s at %s", py_ver, sys.executable)

    try:
        import torch
    except ImportError:
        print("PyTorch:           NOT INSTALLED")
        _file_logger.warning("PyTorch not installed")
        return

    print(f"PyTorch Version:   {torch.__version__}")
    print(f"CUDA Available:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"CUDA Device:       {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability:   {cap[0]}.{cap[1]}")
        print(f"Total GPU Memory:  {props.total_memory / 1024 ** 3:.1f} GB")
        _file_logger.info(
            "PyTorch %s, GPU=%s (%.1f GB, cap %d.%d)",
            torch.__version__,
            torch.cuda.get_device_name(0),
            props.total_memory / 1024 ** 3,
            cap[0], cap[1],
        )
    else:
        print("Note: No CUDA GPU detected - will use CPU (much slower)")
        _file_logger.warning("PyTorch %s, no CUDA GPU detected", torch.__version__)


# (import_name, display_name)
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("transformers",                         "Transformers"),
    ("bitsandbytes",                         "bitsandbytes (NF4)"),
    ("pdfplumber",                           "pdfplumber"),
    ("easyocr",                              "easyocr"),
    ("pandas",                               "pandas"),
    ("smartresume.backend.resume_analyzer",  "SmartResume"),
]


def check_packages() -> int:
    """Return count of missing packages."""
    _print_section("Dependencies")
    missing = 0
    for module_name, display in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            print(f"{display:<16} INSTALLED")
            _file_logger.info("%s: INSTALLED", display)
        except ImportError:
            print(f"{display:<16} NOT INSTALLED")
            _file_logger.warning("%s: NOT INSTALLED", display)
            missing += 1
    return missing


def main() -> int:
    _file_logger.info("--- requirements check started ---")
    check_python_and_gpu()
    missing = check_packages()

    print()
    if missing:
        msg = f"{missing} missing package(s). Run setup.bat to install."
        print(f"Result: {msg}")
        _file_logger.warning("Result: %s", msg)
        return 1
    print("Result: All required packages are installed.")
    _file_logger.info("Result: all required packages installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
