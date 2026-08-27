"""The pre-upload artifact check (scripts/verify_artifact.py).

Everything it catches deploys cleanly and then crash-loops every function on
import, minutes later, with CloudFormation reporting success. These tests build
a miniature artifact and break it one way at a time.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "verify_artifact", ROOT / "scripts" / "verify_artifact.py")
verify_artifact = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_artifact)

# x86_64 ELF64 header: magic, class=2, then e_machine 0x3e at offset 18.
ELF_X86_64 = b"\x7fELF\x02\x01\x01" + b"\x00" * 11 + b"\x3e\x00"
ELF_AARCH64 = b"\x7fELF\x02\x01\x01" + b"\x00" * 11 + b"\xb7\x00"


@pytest.fixture
def artifact(tmp_path) -> Path:
    """A minimal but structurally complete build."""
    root = tmp_path / "build"
    (root / "config").mkdir(parents=True)
    (root / "config" / "default.yaml").write_text("system: {}\n")

    handlers = root / "src" / "trade_agent" / "handlers"
    handlers.mkdir(parents=True)
    for name in ("tick", "screen", "decide", "reflect", "mcp_handler"):
        (handlers / f"{name}.py").write_text("def handler(event, context): ...\n")

    for package in ("pydantic", "pydantic_core", "anthropic", "yaml", "requests"):
        (root / package).mkdir()
    (root / "pydantic_core"
     / "_pydantic_core.cpython-311-x86_64-linux-gnu.so").write_bytes(ELF_X86_64)
    return root


def test_a_good_artifact_passes(artifact, capsys):
    assert verify_artifact.main([str(artifact)]) == 0
    assert "artifact OK" in capsys.readouterr().out


def test_a_missing_directory_is_reported(tmp_path):
    assert verify_artifact.main([str(tmp_path / "nope")]) == 2


# -- the failure that is invisible until the first invocation -------------

def test_the_build_hosts_python_leaking_in_is_caught(artifact, capsys):
    """A cp313 extension on a python3.11 runtime: imports fine on the build
    host, fails on Lambda."""
    old = artifact / "pydantic_core" / "_pydantic_core.cpython-311-x86_64-linux-gnu.so"
    old.rename(old.with_name("_pydantic_core.cpython-313-x86_64-linux-gnu.so"))

    assert verify_artifact.main([str(artifact)]) == 1
    out = capsys.readouterr().out
    assert "cpython-313" in out
    assert "build host's Python leaked" in out


def test_the_runtime_to_check_against_is_selectable(artifact):
    """Building for a 3.13 runtime makes the 3.11 extension the wrong one."""
    assert verify_artifact.main([str(artifact), "--runtime", "3.13"]) == 1
    assert verify_artifact.main([str(artifact), "--runtime", "3.11"]) == 0


def test_a_wrong_cpu_architecture_is_caught(artifact, capsys):
    (artifact / "pydantic_core"
     / "_pydantic_core.cpython-311-x86_64-linux-gnu.so").write_bytes(ELF_AARCH64)
    assert verify_artifact.main([str(artifact)]) == 1
    assert "not x86_64" in capsys.readouterr().out


def test_an_arm64_target_accepts_an_arm64_extension(artifact):
    (artifact / "pydantic_core"
     / "_pydantic_core.cpython-311-x86_64-linux-gnu.so").write_bytes(ELF_AARCH64)
    assert verify_artifact.main([str(artifact), "--arch", "arm64"]) == 0


def test_a_non_elf_file_is_not_mistaken_for_an_extension(artifact):
    (artifact / "pydantic_core" / "stub.cpython-311-x86_64-linux-gnu.so").write_text(
        "not an ELF file")
    assert verify_artifact.main([str(artifact)]) == 0


# -- the failures that stop start-up outright -----------------------------

def test_a_missing_config_is_caught(artifact, capsys):
    (artifact / "config" / "default.yaml").unlink()
    assert verify_artifact.main([str(artifact)]) == 1
    assert "config/default.yaml is missing" in capsys.readouterr().out


def test_a_missing_handler_is_caught(artifact, capsys):
    (artifact / "src" / "trade_agent" / "handlers" / "mcp_handler.py").unlink()
    assert verify_artifact.main([str(artifact)]) == 1
    assert "mcp_handler.py is missing" in capsys.readouterr().out


def test_a_missing_dependency_is_caught(artifact, capsys):
    (artifact / "anthropic").rmdir()
    assert verify_artifact.main([str(artifact)]) == 1
    assert "'anthropic'" in capsys.readouterr().out


def test_every_problem_is_reported_not_just_the_first(artifact, capsys):
    (artifact / "config" / "default.yaml").unlink()
    (artifact / "src" / "trade_agent" / "handlers" / "tick.py").unlink()
    (artifact / "requests").rmdir()

    assert verify_artifact.main([str(artifact)]) == 1
    assert "3 problem(s)" in capsys.readouterr().out


def test_the_checked_handlers_match_the_template():
    """The verifier's handler list has to track template.yaml, or it would
    wave through an artifact missing the very function that was added."""
    import re

    template = (ROOT / "template.yaml").read_text()
    declared = set(re.findall(r"Handler: trade_agent\.handlers\.(\w+)\.handler",
                              template))
    checked = {Path(h).stem for h in verify_artifact.HANDLERS}
    assert declared == checked


# -- the failure these were written after ---------------------------------
#
# .gitignore carried an unanchored `data/`. It matched src/trade_agent/data/
# exactly as well as the data directory it was meant for, so four source files
# were never committed. Every clone built an artifact without them, and every
# Lambda died with Runtime.ImportModuleError. Nothing caught it: the tests pass
# locally because the files are on disk, and the artifact checks looked only at
# named handlers and third-party packages.

def test_every_source_file_is_tracked_by_git():
    """A source file git does not know about does not exist for anyone who
    clones — which is how the artifact is built."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "src", "config"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    tracked = {ROOT / path for path in tracked}

    on_disk = {path for pattern in ("src/**/*.py", "config/**/*.yaml")
               for path in ROOT.glob(pattern)
               if "__pycache__" not in path.parts}

    missing = sorted(str(p.relative_to(ROOT)) for p in on_disk - tracked)
    assert not missing, (
        "these files exist locally but are not in git, so a fresh clone — and "
        f"the Lambda artifact built from it — will not have them: {missing}")


def test_the_verifier_catches_a_missing_internal_package(tmp_path):
    """The check that the shipped artifact needed and did not have."""
    artifact = _minimal_artifact(tmp_path)
    package = artifact / "src" / "trade_agent"
    (package / "handlers" / "tick.py").write_text(
        "from ..data.indicators import rsi\n")

    problems = verify_artifact._check_internal_imports(artifact)
    assert any("trade_agent.data" in p for p in problems), problems

    (package / "data").mkdir()
    (package / "data" / "__init__.py").write_text("")
    (package / "data" / "indicators.py").write_text("def rsi(): ...\n")
    assert not verify_artifact._check_internal_imports(artifact)


def test_an_attribute_import_is_not_mistaken_for_a_module(tmp_path):
    """`from ..money import dec` imports a name, not trade_agent.money.dec."""
    artifact = _minimal_artifact(tmp_path)
    package = artifact / "src" / "trade_agent"
    (package / "money.py").write_text("def dec(x): return x\n")
    (package / "handlers" / "tick.py").write_text("from ...money import dec\n")

    assert not verify_artifact._check_internal_imports(artifact)


def _minimal_artifact(tmp_path):
    artifact = tmp_path / "artifact"
    package = artifact / "src" / "trade_agent"
    (package / "handlers").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "handlers" / "__init__.py").write_text("")
    return artifact
