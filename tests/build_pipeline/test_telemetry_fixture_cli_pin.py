"""The checked-in OTLP fixture was captured from the Claude Code CLI the deployment pins.

The fixture ``tests/fixtures/telemetry/claude_cli_session.otlp.json`` records
the event names and attribute keys the CLI emits; the gateway, the store's
columns and the "reconstruct a session" SQL are tested against it. It is
recorded by ``tests/e2e/test_telemetry_contract_live.py``, which needs a model
and runs only under ``full-ci``. This check needs neither, so it runs on every
pull request: a change of the pinned version that is not accompanied by a new
capture fails here, on the pull request that makes it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "telemetry" / "claude_cli_session.otlp.json"
)
RECAPTURE = "re-capture with OSPREY_CAPTURE_TELEMETRY_FIXTURE=1 under a provider key"
#: The longest attribute value the fixture may keep; mirrors ``MAX_KEPT_VALUE``
#: of the capturing test, which drops every longer value.
MAX_KEPT_VALUE = 64
#: Attribute keys that carry session content; the capture keeps the key only.
CONTENT_KEYS = frozenset(
    {
        "body",
        "prompt",
        "user_prompt",
        "response",
        "tool_input",
        "tool_parameters",
        "output",
        "full_command",
        "bash_command",
    }
)


def test_the_otlp_fixture_was_captured_with_the_pinned_cli_version() -> None:
    assert FIXTURE.is_file(), f"no OTLP fixture at {FIXTURE}; {RECAPTURE}"
    recorded = json.loads(FIXTURE.read_text())
    assert recorded.get("cli_version") == _DEFAULT_CLAUDE_CLI_VERSION, (
        f"the OTLP fixture was captured with Claude Code CLI {recorded.get('cli_version')!r}, "
        f"the pinned version is {_DEFAULT_CLAUDE_CLI_VERSION!r}; {RECAPTURE}"
    )
    assert recorded.get("captured_at"), f"the OTLP fixture records no capture time; {RECAPTURE}"
    assert recorded.get("requests"), f"the OTLP fixture holds no requests; {RECAPTURE}"


def _attributes(node: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(node, list):
        for item in node:
            yield from _attributes(item)
    elif isinstance(node, dict):
        if set(node) == {"key", "value"}:
            yield str(node["key"]), node["value"]
        for value in node.values():
            yield from _attributes(value)


def test_the_otlp_fixture_holds_structure_not_session_content() -> None:
    recorded = json.loads(FIXTURE.read_text())
    too_long = sorted(
        {
            key
            for key, value in _attributes(recorded.get("requests"))
            if isinstance(value, dict)
            and key != "event.name"
            and len(str(value.get("stringValue", ""))) > MAX_KEPT_VALUE
        }
    )
    assert not too_long, f"the OTLP fixture keeps long values under {too_long}; {RECAPTURE}"
    kept = sorted(
        {
            key
            for key, value in _attributes(recorded.get("requests"))
            if key in CONTENT_KEYS and value != {"stringValue": "<redacted>"}
        }
    )
    assert not kept, f"the OTLP fixture keeps session content under {kept}; {RECAPTURE}"
