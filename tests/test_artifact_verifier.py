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
