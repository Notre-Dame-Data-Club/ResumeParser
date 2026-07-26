#!/usr/bin/env python3
"""check_deps.py -- Is this environment already set up? Lets setup.bat skip work.

Reads requirements.txt and compares each declared spec against what is actually
installed, then reports whether torch is a CUDA build. Nothing is installed here.

Exit code is a bitmask so setup.bat can decide what to install:
    0  nothing to do -- everything satisfied
    1  one or more declared requirements missing/outdated -> pip install -e .
    2  torch/torchvision missing or CPU-only            -> install from CUDA index
    3  both of the above

With --gpu, additionally imports torch and prints the live CUDA device (used by
setup.bat's verify step; slower, since importing torch takes a few seconds).
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"

# Packages that must come from the PyTorch CUDA index, not PyPI. On Windows the
# PyPI wheels are CPU-only and carry no local version, so a "+cu" tag is what
# distinguishes a CUDA build.
CUDA_PACKAGES = ("torch", "torchvision")

NEED_DEPS = 1
NEED_TORCH = 2


def _load_requirement_parser():
    """packaging isn't guaranteed on a fresh machine; fall back to pip's copy."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
        except ImportError:
            return None
    return Requirement


def declared_specs() -> list[str]:
    if not REQUIREMENTS.is_file():
        print(f"  requirements.txt not found at {REQUIREMENTS}")
        return []
    specs = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            specs.append(line)
    return specs


def check_declared(specs: list[str]) -> list[str]:
    """Return a human-readable problem per unsatisfied requirement."""
    Requirement = _load_requirement_parser()
    problems = []

    for spec in specs:
        if Requirement is None:
            # Presence-only fallback: strip any version operator.
            name = spec
            for op in (">=", "<=", "==", "~=", "!=", ">", "<"):
                name = name.split(op, 1)[0]
            name = name.strip()
            try:
                version(name)
            except PackageNotFoundError:
                problems.append(f"  {name:<16} MISSING")
            continue

        req = Requirement(spec)
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            problems.append(f"  {req.name:<16} MISSING          (need {req.specifier or 'any'})")
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            problems.append(f"  {req.name:<16} {installed:<16} (need {req.specifier})")

    return problems


def check_cuda_build() -> list[str]:
    """Return a problem per CUDA package that is absent or not a +cu build."""
    problems = []
    for name in CUDA_PACKAGES:
        try:
            installed = version(name)
        except PackageNotFoundError:
            problems.append(f"  {name:<16} MISSING          (need a CUDA build)")
            continue
        if "+cu" not in installed:
            problems.append(f"  {name:<16} {installed:<16} (CPU-only build)")
    return problems


def report_gpu() -> None:
    try:
        import torch
    except ImportError:
        print("  GPU              torch not importable")
        return
    if torch.cuda.is_available():
        print(f"  GPU              {torch.cuda.get_device_name(0)}")
    else:
        print(f"  GPU              none detected - torch {torch.__version__} will run on CPU (slower)")


def main() -> int:
    specs = declared_specs()
    dep_problems = check_declared(specs)
    cuda_problems = check_cuda_build()

    status = 0
    if dep_problems:
        status |= NEED_DEPS
    if cuda_problems:
        status |= NEED_TORCH

    if status == 0:
        torch_ver = version("torch")
        print(f"  {len(specs)} requirements satisfied, torch {torch_ver}")
    else:
        for problem in cuda_problems + dep_problems:
            print(problem)

    if "--gpu" in sys.argv:
        report_gpu()

    return status


if __name__ == "__main__":
    raise SystemExit(main())
