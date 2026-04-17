"""Canonical runtime stack used by launchers and docs.

Keep all runtime-sensitive pins in one place to avoid drift between scripts.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Tuple

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except Exception as exc:  # pragma: no cover - packaging is expected in runtime envs
    raise RuntimeError("Missing required dependency 'packaging' for runtime validation") from exc


# Canonical stack for the currently supported and validated training path.
CANONICAL_RUNTIME: Dict[str, str] = {
    "python": "3.11",
    "verl": "0.6.1",
    "vllm": "0.8.5",
    "ray": ">=2.40.0",
    "transformers": ">=4.51.0,<5.0.0",
    "tensordict": ">=0.8.0,<=0.10.0,!=0.9.0",
}

# Dependency pins consumed by Modal image/local fallback checks.
CANONICAL_PIP_PACKAGES: Tuple[str, ...] = (
    f"verl=={CANONICAL_RUNTIME['verl']}",
    f"ray[default]{CANONICAL_RUNTIME['ray']}",
    f"vllm=={CANONICAL_RUNTIME['vllm']}",
    f"transformers{CANONICAL_RUNTIME['transformers']}",
    f"tensordict{CANONICAL_RUNTIME['tensordict']}",
)


def runtime_summary_lines() -> list[str]:
    """Return a human-readable summary for logs/docs."""
    return [
        f"python={CANONICAL_RUNTIME['python']}",
        f"verl={CANONICAL_RUNTIME['verl']}",
        f"vllm={CANONICAL_RUNTIME['vllm']}",
        f"ray{CANONICAL_RUNTIME['ray']}",
        f"transformers{CANONICAL_RUNTIME['transformers']}",
        f"tensordict{CANONICAL_RUNTIME['tensordict']}",
    ]


def runtime_summary_text() -> str:
    """Single-line canonical stack text."""
    return ", ".join(runtime_summary_lines())


def _matches_python(expected: str) -> bool:
    major_minor = ".".join(str(v) for v in sys.version_info[:2])
    return major_minor == expected


def _get_pkg_version(pkg_name: str) -> str | None:
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return None


def validate_runtime_stack() -> None:
    """Fail fast when local/container runtime drifts from canonical stack."""
    errors: list[str] = []

    expected_python = CANONICAL_RUNTIME["python"]
    if not _matches_python(expected_python):
        current = ".".join(str(v) for v in sys.version_info[:3])
        errors.append(f"python expected {expected_python}.x, got {current}")

    for package in ("verl", "vllm"):
        expected = CANONICAL_RUNTIME[package]
        current = _get_pkg_version(package)
        if current is None:
            errors.append(f"{package} expected {expected}, got not-installed")
            continue
        if Version(current) != Version(expected):
            errors.append(f"{package} expected {expected}, got {current}")

    for package in ("ray", "transformers", "tensordict"):
        spec_text = CANONICAL_RUNTIME[package]
        current = _get_pkg_version(package)
        if current is None:
            errors.append(f"{package} expected {spec_text}, got not-installed")
            continue
        if Version(current) not in SpecifierSet(spec_text):
            errors.append(f"{package} expected {spec_text}, got {current}")

    if errors:
        details = "\n".join(f"- {item}" for item in errors)
        raise RuntimeError(f"Runtime stack validation failed:\n{details}")