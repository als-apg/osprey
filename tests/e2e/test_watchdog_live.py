"""Live watchdog checks against real services: opt-in, never part of CI.

Two proofs that a stubbed model cannot give:

* the decision engine answers within its latency budget, over 50 prompts on the
  direct Jev route (``TYPESAFE_API_KEY``, spends real money per call);
* an agent watchdog, on the provider the run names, reads the machine state its
  answer turns on: it flags the risky fixture citing the channel and the value it
  read, and does not flag the benign one, whose text is the same.

Run with both keys exported into the shell that starts pytest::

    export TYPESAFE_API_KEY=...
    OSPREY_E2E_PROVIDER=<provider> uv run pytest tests/e2e/test_watchdog_live.py -q

A run that names a provider has asked for this proof, so a key it lacks fails the
test by name instead of skipping it: a skip would read as a proof that passed. No
test reads or prints a key, and a failure names an error's type, never its text.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.services.watchdog.engines.agent as agent_engine
from osprey.agent_runner import ToolUseEvent
from osprey.cli.watchdog_cmd import _percentile
from osprey.services.watchdog.config import WatchdogDefinition, parse_watchdogs_block
from osprey.services.watchdog.engines.agent import AgentEngine
from osprey.services.watchdog.history import SessionNodes
from osprey.services.watchdog.runner import WatchdogRunner
from osprey.services.watchdog.verdict import Verdict
from tests.conftest import _e2e_provider_reason, _has_e2e_provider_key, _has_typesafe_api_key
from tests.e2e.provider import build_model, build_provider, e2e_provider
from tests.e2e.test_watchdog_agent_scripted import (
    FIXTURES,
    PRESSURE,
    READ,
    build_deployment,
    read_pair_fixture,
    reader_definition,
    seeded,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"
PRESET_WATCHDOG = "guard-circumvention"

MEASUREMENTS = 50
P95_LIMIT_S = 3.0
TYPESAFE_CONFIG: Mapping[str, Any] = {"api": {"jev": {"route": "typesafe"}}}

#: Superscripts and the unicode minus, as plain text for :data:`RISKY_VALUE`.
PLAIN_NUMBERS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻−", "0123456789--")
#: How the model may spell the risky seed's value when it cites it.
RISKY_VALUE = re.compile(
    r"(?<![\d.])(?:"
    r"5(?:\.0+)?\s*[eE]\s*-0?5"
    r"|5(?:\.0+)?\s*(?:[x×*·]|\\times)\s*10\s*\^?\s*\{?\s*-0?5"
    r"|0\.00005"
    r")(?!\d)"
)


def _cites_the_risky_value(reason: str) -> bool:
    return RISKY_VALUE.search(reason.translate(PLAIN_NUMBERS)) is not None


def _run_names_a_provider() -> bool:
    with contextlib.suppress(RuntimeError):
        return bool(e2e_provider())
    return False


def _gated(marker: str, key_present: bool, failing_fixture: str) -> pytest.MarkDecorator:
    """The test's resource gate: its marker, or a failure when the run asked and lacks the key.

    The marker alone turns an absent key into a skip before any fixture runs, so
    in the one case that must fail the test does not carry it.
    """
    if key_present or not _run_names_a_provider():
        return getattr(pytest.mark, marker)
    return pytest.mark.usefixtures(failing_fixture)


@pytest.fixture
def _typesafe_key_missing() -> None:
    pytest.fail("OSPREY_E2E_PROVIDER is set and TYPESAFE_API_KEY is not exported", pytrace=False)


@pytest.fixture
def _provider_key_missing() -> None:
    pytest.fail(_e2e_provider_reason(), pytrace=False)


requires_typesafe_or_fail = _gated(
    "requires_typesafe", _has_typesafe_api_key(), "_typesafe_key_missing"
)
requires_provider_or_fail = _gated(
    "requires_e2e_provider", _has_e2e_provider_key(), "_provider_key_missing"
)


def _preset_decision_definition() -> WatchdogDefinition:
    """The decision watchdog the control-assistant preset ships, switched on."""
    document = yaml.safe_load((PRESET_DIR / "control-assistant.yml").read_text(encoding="utf-8"))
    definitions = parse_watchdogs_block(document["watchdogs"], base_dir=PRESET_DIR)
    by_name = {defn.name: defn for defn in definitions}
    return replace(by_name[PRESET_WATCHDOG], enabled=True)


def _decision_fixtures() -> list[tuple[str, list[dict[str, Any]]]]:
    fixtures = []
    for path in sorted((FIXTURES / "decision").glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        fixtures.append((json.loads(lines[0])["prompt"], [json.loads(line) for line in lines[1:]]))
    return fixtures


@requires_typesafe_or_fail
async def test_decision_verdict_latency_p95_on_the_direct_route() -> None:
    """Criterion: p95 from prompt submission to verdict is at most 3 s over 50 live calls."""
    fixtures = _decision_fixtures()
    verdicts: list[Verdict] = []
    latencies: list[float] = []
    runner = WatchdogRunner(
        [_preset_decision_definition()],
        recorder=lambda _verdict: None,
        hold_root=None,
        notify=lambda _frame: None,
        config=TYPESAFE_CONFIG,
    )
    try:
        for index in range(MEASUREMENTS):
            prompt, nodes = fixtures[index % len(fixtures)]
            # Distinct text, so no call is answered from a provider's cache.
            prompt = f"{prompt}\n\nRequest {index + 1} of this measurement."
            submitted = time.monotonic()
            answered = await runner.on_nodes(
                session_key=None,
                record_session_id=f"live-latency-{index}",
                submission_id=f"live-latency-{index}",
                nodes=SessionNodes(nodes=nodes),
                prompt=prompt,
                deadline=submitted + 30,
                can_hold=False,
            )
            latencies.append(time.monotonic() - submitted)
            verdicts.extend(answered)
    finally:
        await runner.shutdown()

    failed = [(v.status, v.error_type) for v in verdicts if v.status != "ok"]
    assert not failed, f"{len(failed)} of {MEASUREMENTS} calls gave no verdict: {failed[:5]}"
    assert len(verdicts) == MEASUREMENTS
    p95 = _percentile(latencies, 95)
    print(f"\nwatchdog decision latency: p50={_percentile(latencies, 50):.2f}s p95={p95:.2f}s")
    assert p95 <= P95_LIMIT_S, f"p95 {p95:.2f} s over {MEASUREMENTS} calls"


@pytest.fixture(scope="module")
def live_deployment(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The ``agent-profile`` fixture built with the provider this run names."""
    return build_deployment(
        tmp_path_factory.mktemp("watchdog-live"),
        provider=build_provider(e2e_provider()),
        model=build_model(None),
    )


@pytest.fixture
def traced_tools(monkeypatch: pytest.MonkeyPatch) -> list[ToolUseEvent]:
    """Every tool call the judge's real run makes; the run itself is left as it is."""
    calls: list[ToolUseEvent] = []
    real = agent_engine.stream_query

    async def tee(*args: Any, **kwargs: Any):
        async with contextlib.aclosing(real(*args, **kwargs)) as events:
            async for event in events:
                if isinstance(event, ToolUseEvent):
                    calls.append(event)
                yield event

    monkeypatch.setattr(agent_engine, "stream_query", tee)
    return calls


@pytest.fixture
def judge_pair_fixture(
    live_deployment: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], Any]:
    repo, build = live_deployment
    monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))
    monkeypatch.setenv("CONFIG_FILE", str(build / "config.yml"))

    async def _judge(label: str) -> Verdict:
        prompt, nodes, machine = read_pair_fixture(label)
        async with seeded(build, machine):
            return await AgentEngine(build).judge(
                reader_definition(repo),
                nodes,
                prompt,
                deadline=time.monotonic() + 120,
                session_key="live-session",
                record_session_id="live-record",
                submission_id=f"live-{label}",
            )

    return _judge


def _assert_read_the_channel(calls: list[ToolUseEvent]) -> None:
    reads = [call for call in calls if call.name == READ]
    assert reads, f"the judge never read a channel: {[call.name for call in calls]}"
    assert any(PRESSURE in json.dumps(call.input) for call in reads), "it read another channel"


@requires_provider_or_fail
async def test_the_risky_seed_is_flagged_citing_what_the_judge_read(
    judge_pair_fixture: Callable[[str], Any], traced_tools: list[ToolUseEvent]
) -> None:
    verdict = await judge_pair_fixture("risky")

    assert verdict.status == "ok", (verdict.status, verdict.error_type)
    assert verdict.flagged is True, verdict.reason
    assert PRESSURE in (verdict.reason or ""), verdict.reason
    assert _cites_the_risky_value(verdict.reason or ""), verdict.reason
    _assert_read_the_channel(traced_tools)


@requires_provider_or_fail
async def test_the_benign_seed_with_the_same_text_is_not_flagged(
    judge_pair_fixture: Callable[[str], Any], traced_tools: list[ToolUseEvent]
) -> None:
    verdict = await judge_pair_fixture("benign")

    assert verdict.status == "ok", (verdict.status, verdict.error_type)
    assert verdict.flagged is False, verdict.reason
    _assert_read_the_channel(traced_tools)


@pytest.mark.parametrize(
    "reason",
    ["reads 5e-05 torr", "at 5.0E-5", "at 5 × 10⁻⁵ torr", "5 x 10^-5", "5 × 10−5", "0.00005 torr"],
)
def test_the_citation_check_takes_the_spellings_of_the_risky_value(reason: str) -> None:
    assert _cites_the_risky_value(reason)


@pytest.mark.parametrize("reason", ["reads 2.5e-5", "15e-5", "5e-50", "5 x 10^5", "1e-07"])
def test_the_citation_check_refuses_other_values(reason: str) -> None:
    assert not _cites_the_risky_value(reason)


def test_the_pair_differs_only_in_its_machine_values() -> None:
    """Only the read can tell them apart, or the two verdicts above prove nothing."""
    risky, benign = read_pair_fixture("risky"), read_pair_fixture("benign")

    assert risky[:2] == benign[:2]
    assert risky[2] != benign[2]
    assert set(risky[2]) == {PRESSURE}
