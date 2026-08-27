#!/usr/bin/env python3
"""Check a built Lambda artifact before it is uploaded.

    scripts/verify_artifact.py <artifacts-dir> [--runtime 3.11] [--arch x86_64]

A package that is wrong in the ways this catches does not fail at deploy time.
It deploys cleanly and then every function crashes on import, several minutes
later, with the stack looking healthy in CloudFormation. This turns that into a
build-time error.

Checked, in order of how quietly each one fails in production:

  1. native extensions carry the *target runtime's* ABI tag — the failure mode
     when a build host's Python differs from the Lambda runtime, and invisible
     until the first invocation;
  2. the package and its config actually made it into the artifact;
  3. the imports the handlers need are present;
  4. every declared handler module exists at the path Lambda will look for.

It never imports the artifact: on a host whose Python differs from the runtime,
importing the (correct) extensions would fail and reject a good build. The ABI
tag is the evidence, and it is readable from any host.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HANDLERS = [
    "trade_agent/handlers/tick.py",
    "trade_agent/handlers/screen.py",
    "trade_agent/handlers/decide.py",
    "trade_agent/handlers/reflect.py",
    "trade_agent/handlers/mcp_handler.py",
]
REQUIRED_PACKAGES = ["pydantic", "pydantic_core", "anthropic", "yaml", "requests"]
ELF_X86_64 = 0x3E
ELF_AARCH64 = 0xB7
MACHINES = {"x86_64": ELF_X86_64, "arm64": ELF_AARCH64}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("--runtime", default="3.11",
                        help="target Lambda Python version (default 3.11)")
    parser.add_argument("--arch", default="x86_64", choices=sorted(MACHINES))
    args = parser.parse_args(argv)

    root: Path = args.artifacts_dir
    problems: list[str] = []

    if not root.is_dir():
        print(f"artifact directory not found: {root}", file=sys.stderr)
        return 2

    problems += _check_abi(root, args.runtime, args.arch)
    problems += _check_layout(root)
    problems += _check_packages(root)

    if problems:
        print(f"artifact check FAILED ({len(problems)} problem(s)):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    extensions = list(root.rglob("*.so"))
    print(f"artifact OK: python{args.runtime}/{args.arch}, "
          f"{len(extensions)} native extension(s), "
          f"{len(HANDLERS)} handler(s), config present")
    return 0


def _check_abi(root: Path, runtime: str, arch: str) -> list[str]:
    """Native extensions must match the runtime the function will run on."""
    expected_tag = "cpython-" + runtime.replace(".", "")
    expected_machine = MACHINES[arch]
    problems: list[str] = []

    for path in root.rglob("*.so"):
        name = path.name
        tag = re.search(r"cpython-(\d+)", name)
        if tag and f"cpython-{tag.group(1)}" != expected_tag:
            problems.append(
                f"{path.relative_to(root)} is built for cpython-{tag.group(1)}, "
                f"not {expected_tag}. The build host's Python leaked into the "
                "artifact; it will fail to import on Lambda.")
            continue
        machine = _elf_machine(path)
        if machine is not None and machine != expected_machine:
            problems.append(
                f"{path.relative_to(root)} is compiled for machine "
                f"{machine:#x}, not {arch}.")
    return problems


def _elf_machine(path: Path) -> int | None:
    """e_machine from the ELF header, or None if it is not an ELF file."""
    try:
        header = path.read_bytes()[:20]
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    return int.from_bytes(header[18:20], "little")


def _check_layout(root: Path) -> list[str]:
    problems: list[str] = []
    if not (root / "config" / "default.yaml").is_file():
        problems.append(
            "config/default.yaml is missing; every function fails at start-up "
            "because it cannot load its configuration")
    for handler in HANDLERS:
        if not (root / "src" / handler).is_file():
            problems.append(f"src/{handler} is missing from the artifact")
    return problems


def _check_packages(root: Path) -> list[str]:
    """Presence only — importing would need the target interpreter."""
    problems = []
    for package in REQUIRED_PACKAGES:
        if not ((root / package).is_dir() or (root / f"{package}.py").is_file()):
            problems.append(f"dependency {package!r} is not in the artifact")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
