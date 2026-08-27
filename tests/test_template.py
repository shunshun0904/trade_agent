"""Checks on template.yaml that CloudFormation only makes at deploy time.

A template can be perfectly valid YAML, pass `cfn-lint`, and still be rejected
several minutes into a deploy by the service that actually owns the resource.
When that happens on a *create*, the stack rolls back to an empty
ROLLBACK_COMPLETE shell that cannot be updated, so the next attempt fails on
the shell rather than the cause and the original error disappears.

These tests move the cheap half of that feedback to the test suite.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

# IAM validates several free-text fields against
# [\p{L}\p{M}\p{Z}\p{S}\p{N}\p{P}]* — letters, marks, separators, symbols,
# numbers and punctuation. Notably absent: \p{C}, the control characters. A
# newline is Cc, so a YAML folded block (`>`, which keeps a trailing newline)
# in one of these fields is rejected, while `>-` is fine. Em dashes and other
# typography are Pd/Sk and pass.
IAM_TEXT_CATEGORIES = ("L", "M", "Z", "S", "N", "P")

IAM_TYPES = {
    "AWS::IAM::ManagedPolicy",
    "AWS::IAM::Role",
    "AWS::IAM::InstanceProfile",
}


class _Loader(yaml.SafeLoader):
    """SafeLoader that keeps CloudFormation's short-form tags readable.

    `!Sub 'x'` becomes `{"Fn::Sub": "x"}`, the same shape the long form
    parses to, so a test can look at what a property actually says instead of
    tripping over the tag.
    """


def _short_form(loader, suffix, node):
    key = "Ref" if suffix == "Ref" else f"Fn::{suffix}"
    if isinstance(node, yaml.ScalarNode):
        return {key: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {key: loader.construct_sequence(node, deep=True)}
    return {key: loader.construct_mapping(node, deep=True)}


_Loader.add_multi_constructor("!", _short_form)


@pytest.fixture(scope="module")
def template() -> dict:
    return yaml.load((ROOT / "template.yaml").read_text(encoding="utf-8"),
                     Loader=_Loader)


def _rejected_by_iam(text: str) -> list[tuple[int, str, str]]:
    return [(i, repr(ch), unicodedata.category(ch))
            for i, ch in enumerate(text)
            if not unicodedata.category(ch).startswith(IAM_TEXT_CATEGORIES)]


def test_iam_descriptions_contain_no_control_characters(template):
    """The failure this catches cost a full deploy cycle.

    `Description: >` on SecretsReadPolicy folded to a single line with a
    trailing newline, and IAM rejected it with a 400 after CloudFormation had
    already created the DynamoDB tables and the bucket.
    """
    offenders = {}
    for name, resource in template["Resources"].items():
        if resource.get("Type") not in IAM_TYPES:
            continue
        description = resource.get("Properties", {}).get("Description")
        if not isinstance(description, str):
            continue
        bad = _rejected_by_iam(description)
        if bad:
            offenders[name] = bad

    assert not offenders, (
        "IAM rejects these descriptions at deploy time "
        f"(use `>-` rather than `>`): {offenders}")


def test_every_function_is_ordered_after_its_log_group(template):
    """Lambda creates /aws/lambda/<name> on a function's first invocation, and
    CloudFormation cannot create a log group that already exists. Because the
    schedules can fire while the stack is still building, a function that is
    not ordered after its log group can lose a race against its own first run
    — and the orphaned group then fails every retry the same way."""
    resources = template["Resources"]
    functions = {name for name, r in resources.items()
                 if r.get("Type") == "AWS::Serverless::Function"}
    assert functions, "no functions found; the template parse is wrong"

    for name in sorted(functions):
        log_group = name.replace("Function", "LogGroup")
        assert log_group in resources, (
            f"{name} has no matching {log_group}; its log group would be "
            "created by Lambda with no retention policy")
        depends = resources[name].get("DependsOn")
        depends = [depends] if isinstance(depends, str) else (depends or [])
        assert log_group in depends, (
            f"{name} must declare DependsOn: {log_group}")


def test_log_groups_match_the_function_names(template):
    """A log group whose name does not match its function is retained forever
    while the real one, created by Lambda, is not retained at all."""
    resources = template["Resources"]
    for name, resource in resources.items():
        if resource.get("Type") != "AWS::Serverless::Function":
            continue
        function_name = resource["Properties"]["FunctionName"]["Fn::Sub"]
        log_group = resources[name.replace("Function", "LogGroup")]
        declared = log_group["Properties"]["LogGroupName"]["Fn::Sub"]
        assert declared == f"/aws/lambda/{function_name}", (
            f"{name}: log group {declared!r} does not match "
            f"/aws/lambda/{function_name}")
