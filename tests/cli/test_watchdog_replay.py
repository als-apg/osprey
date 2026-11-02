"""``osprey watchdog replay``: the report over labelled prompts, and what it leaves alone."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_runner import ResultEvent
from osprey.agent_runner import launcher as agent_launcher
from osprey.cli import watchdog_cmd
from osprey.cli.main import cli
from osprey.cli.watchdog_cmd import _MachineSeed
from osprey.services.watchdog.definitions import load_profile_watchdogs
from osprey.services.watchdog.engines import agent as agent_module
from osprey.services.watchdog.engines.decision import QUESTION_ID, DecisionEngine
from osprey.services.watchdog.history import build_session_view_from_nodes, rough_tokens
from osprey.services.watchdog.runner import WatchdogRunner
from osprey.services.watchdog.verdict import Verdict
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests._preset_data import copy_bundle_data
from tests.cli._lifecycle_build import invoke_build
from tests.models.jev._jev_fake_transport import FakeJev

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "watchdog"
PRESET_DIR = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"
TWIN = "guard-circumvention"


def _invoke(*args: str):
    return CliRunner().invoke(cli, ["watchdog", "replay", *args])


def _write_fixture(directory: Path, name: str, label: str, **extra) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    header = {"label": label, "prompt": f"prompt {name}", **extra}
    (directory / f"{name}.jsonl").write_text(json.dumps(header) + "\n", encoding="utf-8")


def _labelled_set(root: Path) -> Path:
    for index in range(4):
        _write_fixture(root / "known", f"risky-{index}", "risky")
        _write_fixture(root / "known", f"benign-{index}", "benign")
    return root


def _decision_profile(root: Path, *, enabled: str = "false") -> Path:
    (root / "watchdogs" / "stub").mkdir(parents=True)
    (root / "watchdogs" / "stub" / "instructions.md").write_text("Flag it.\n", encoding="utf-8")
    (root / "profile.yml").write_text(
        "name: Replay\nprovider: anthropic\nmodel: m\nwatchdogs:\n  stub:\n"
        f"    engine: decision\n    enabled: {enabled}\n    threshold: 0.5\n    options:\n"
        "      - {id: fine, text: Fine, flag: false}\n"
        "      - {id: bad, text: Bad, flag: true}\n",
        encoding="utf-8",
    )
    return root


class _Stub:
    """An engine answering by prompt, at a fixed latency and cost."""

    def __init__(self, flagged=(), unavailable=False):
        self.flagged, self.unavailable = set(flagged), unavailable
        self.calls = []

    async def judge(self, defn, _nodes, prompt, *, session_key, **ids):
        self.calls.append((defn.name, prompt, session_key))
        common = {
            "watchdog": defn.name,
            "engine": defn.engine,
            "record_session_id": ids["record_session_id"],
            "submission_id": ids["submission_id"],
            "threshold": 0.5 if defn.engine == "decision" else None,
            "latency_s": 2.0 if prompt.endswith("3") else 1.0,
            "cost_usd": 0.01,
        }
        if self.unavailable:
            return Verdict(status="unavailable", error_type="missing_key", **common)
        return Verdict(status="ok", flagged=prompt in self.flagged, **common)


def _stub_engines(monkeypatch, engine) -> None:
    monkeypatch.setattr(WatchdogRunner, "_build_engine", lambda self, _name: engine)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """The command sets ``OSPREY_CONFIG``; hand it back, and stand outside any repo."""
    monkeypatch.setenv("OSPREY_CONFIG", "")
    monkeypatch.chdir(tmp_path)


class TestReport:
    def test_known_labels_print_the_rates_latency_and_cost(self, tmp_path, monkeypatch):
        fixtures = _labelled_set(tmp_path / "fixtures")
        profile = _decision_profile(tmp_path / "profile")
        flagged = {"prompt risky-0", "prompt risky-1", "prompt risky-2", "prompt benign-0"}
        _stub_engines(monkeypatch, _Stub(flagged))

        result = _invoke(str(fixtures), "--profile", str(profile))

        assert result.exit_code == 0, result.output
        assert re.search(r"Hit rate on risky\s+75 % \(3 of 4\)", result.output)
        assert re.search(r"False flags on benign\s+25 % \(1 of 4\)", result.output)
        assert re.search(r"Latency p50\s+1\.00 s", result.output)
        assert re.search(r"Latency p95\s+2\.00 s", result.output)
        assert re.search(r"Cost\s+\$0\.0800", result.output)
        assert "engine: decision" in result.output
        assert "set: known" in result.output

    def test_an_engine_silent_on_every_fixture_exits_non_zero(self, tmp_path, monkeypatch):
        fixtures = _labelled_set(tmp_path / "fixtures")
        profile = _decision_profile(tmp_path / "profile")
        _stub_engines(monkeypatch, _Stub(unavailable=True))

        result = _invoke(str(fixtures), "--profile", str(profile))

        assert result.exit_code == 1
        assert "8 of 8 (missing_key)" in result.output

    def test_poor_rates_alone_exit_zero(self, tmp_path, monkeypatch):
        fixtures = _labelled_set(tmp_path / "fixtures")
        profile = _decision_profile(tmp_path / "profile")
        _stub_engines(monkeypatch, _Stub())

        result = _invoke(str(fixtures), "--profile", str(profile))

        assert result.exit_code == 0
        assert "0 % (0 of 4)" in result.output

    def test_an_engine_answering_once_exits_zero(self, tmp_path, monkeypatch):
        fixtures = _labelled_set(tmp_path / "fixtures")
        profile = _decision_profile(tmp_path / "profile")
        answering = _Stub()
        silent = _Stub(unavailable=True)

        class Mixed:
            async def judge(self, defn, nodes, prompt, **kwargs):
                chosen = answering if prompt == "prompt risky-0" else silent
                return await chosen.judge(defn, nodes, prompt, **kwargs)

        _stub_engines(monkeypatch, Mixed())

        assert _invoke(str(fixtures), "--profile", str(profile)).exit_code == 0

    def test_nothing_is_recorded_and_nothing_can_hold(
        self, tmp_path, monkeypatch, recorded_entries
    ):
        fixtures = _labelled_set(tmp_path / "fixtures")
        profile = _decision_profile(tmp_path / "profile")
        engine = _Stub({"prompt risky-0"})
        _stub_engines(monkeypatch, engine)

        assert _invoke(str(fixtures), "--profile", str(profile)).exit_code == 0

        assert recorded_entries() == []
        assert {session_key for _, _, session_key in engine.calls} == {None}

    def test_a_disabled_preset_watchdog_replays(self, tmp_path, monkeypatch):
        assert not load_profile_watchdogs(self._preset(tmp_path))[0].enabled
        fixtures = _labelled_set(tmp_path / "fixtures")
        _stub_engines(monkeypatch, _Stub())

        result = _invoke(str(fixtures), "--profile", str(self._preset(tmp_path)))

        assert result.exit_code == 0, result.output
        assert TWIN in result.output

    @staticmethod
    def _preset(tmp_path: Path) -> Path:
        directory = tmp_path / "preset"
        if not directory.exists():
            directory.mkdir()
            shutil.copytree(PRESET_DIR / "watchdogs" / TWIN, directory / "watchdogs" / TWIN)
            (directory / "profile.yml").write_text(
                (PRESET_DIR / "control-assistant.yml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        return directory

    def test_the_default_profile_is_the_projects(self, tmp_path, monkeypatch):
        fixtures = _labelled_set(tmp_path / "fixtures")
        project = _decision_profile(tmp_path / "repo")
        _stub_engines(monkeypatch, _Stub())
        monkeypatch.chdir(project)

        result = _invoke(str(fixtures))

        assert result.exit_code == 0, result.output
        assert "stub" in result.output


class TestRefusals:
    def test_a_watchdog_no_profile_defines(self, tmp_path):
        fixtures = _labelled_set(tmp_path / "fixtures")
        profile = _decision_profile(tmp_path / "profile")

        result = _invoke(str(fixtures), "--profile", str(profile), "--watchdog", "nope")

        assert result.exit_code == 2
        assert "nope" in result.output

    def test_a_malformed_fixture_is_named(self, tmp_path):
        (tmp_path / "fixtures").mkdir()
        bad = tmp_path / "fixtures" / "bad.jsonl"
        bad.write_text('{"label": "maybe", "prompt": "p"}\n', encoding="utf-8")
        profile = _decision_profile(tmp_path / "profile")

        result = _invoke(str(tmp_path / "fixtures"), "--profile", str(profile))

        assert result.exit_code == 2
        assert "bad.jsonl" in result.output

    def test_an_empty_directory_is_refused(self, tmp_path):
        (tmp_path / "fixtures").mkdir()
        profile = _decision_profile(tmp_path / "profile")

        assert _invoke(str(tmp_path / "fixtures"), "--profile", str(profile)).exit_code == 2

    def test_no_repo_and_no_profile(self, tmp_path):
        fixtures = _labelled_set(tmp_path / "fixtures")

        assert _invoke(str(fixtures)).exit_code == 2


class TestAgentDefinitions:
    def test_without_a_project_they_are_reported_not_run(self, tmp_path, monkeypatch):
        fixtures = _labelled_set(tmp_path / "fixtures")
        _stub_engines(monkeypatch, _Stub())

        result = _invoke(str(fixtures), "--profile", str(FIXTURES / "agent-profile"))

        assert result.exit_code == 0, result.output
        assert re.search(r"Not run\s+8 \(needs a built project", result.output)
        assert "agent-reader" in result.output

    def test_a_watchdog_can_be_selected(self, tmp_path, monkeypatch):
        fixtures = _labelled_set(tmp_path / "fixtures")
        _stub_engines(monkeypatch, _Stub())

        result = _invoke(
            str(fixtures),
            "--profile",
            str(FIXTURES / "agent-profile"),
            "--watchdog",
            "agent-reader",
        )

        assert "agent-reader" in result.output
        assert f"{TWIN}-agent" not in result.output


class TestShippedFixtures:
    @pytest.mark.parametrize("path", sorted(FIXTURES.glob("*/*.jsonl")), ids=lambda path: path.stem)
    def test_every_fixture_parses_into_a_session_view(self, path):
        fixture = watchdog_cmd._read_fixtures(path.parent)
        assert fixture
        lines = path.read_text(encoding="utf-8").splitlines()
        nodes = [json.loads(line) for line in lines[1:]]
        view = build_session_view_from_nodes(
            nodes, json.loads(lines[0])["prompt"], max_tokens=4000, count_tokens=rough_tokens
        )
        assert view.text

    def test_the_decision_set_and_the_agent_pair_are_there(self):
        fixtures = watchdog_cmd._read_fixtures(FIXTURES)
        decision = [f.label for f in fixtures if f.set_name == "decision"]
        assert decision.count("risky") >= 10
        assert decision.count("benign") >= 10
        pair = [f for f in fixtures if f.set_name == "agent"]
        assert sorted(f.label for f in pair) == ["benign", "risky"]
        assert all(f.machine for f in pair)

    def test_fake_jev_and_fake_agent_runner_print_both_engine_rows(
        self, tmp_path, monkeypatch, recorded_entries
    ):
        build = tmp_path / "project" / "build"
        build.mkdir(parents=True)
        (build / "config.yml").write_text("control_system:\n  type: x\n", encoding="utf-8")
        (build / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "controls": {"command": "c"},
                        "osprey_facility_knowledge": {"command": "k"},
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(agent_launcher, "project_agent_cli_path", lambda _dir: "/bin/claude")
        monkeypatch.setattr("osprey.agent_runner.primitives.sdk_env", lambda *a, **k: {})
        monkeypatch.setattr(
            "osprey.agent_runner.primitives.resolve_default_model", lambda *a, **k: "m"
        )
        monkeypatch.setattr(
            "osprey.agent_runner.primitives._resolve_project_spec", lambda *a, **k: None
        )
        seeded: list[dict] = []

        async def seed(_self, machine):
            if machine:
                seeded.append(dict(machine))
            return None

        async def close(_self):
            return None

        monkeypatch.setattr(_MachineSeed, "apply", seed)
        monkeypatch.setattr(_MachineSeed, "close", close)

        def stream(_project_dir, _prompt, **_kwargs):
            async def events():
                yield ResultEvent(
                    subtype="success",
                    is_error=False,
                    num_turns=1,
                    duration_ms=1,
                    session_id="judge",
                    total_cost_usd=0.02,
                    usage=None,
                    result=json.dumps({"flagged": False, "explanation": "fine", "sources": []}),
                    api_error_status=None,
                )

            return events()

        monkeypatch.setattr(agent_module, "stream_query", stream)
        fake = FakeJev(
            {
                QUESTION_ID: {
                    "type": "choice",
                    "choice": "k001",
                    "probabilities": {"k001": 1.0},
                }
            }
        )
        real_build = WatchdogRunner._build_engine

        def build_engine(self, name):
            if name == "decision":
                config = {"api": {"jev": {"routes": {"typesafe": {"api_key": "k"}}}}}
                return DecisionEngine(config, transport=fake.transport)
            return real_build(self, name)

        monkeypatch.setattr(WatchdogRunner, "_build_engine", build_engine)
        preset = TestReport._preset(tmp_path)

        result = _invoke(
            str(FIXTURES),
            "--profile",
            str(preset),
            "--profile",
            str(FIXTURES / "agent-profile"),
            "--project-dir",
            str(tmp_path / "project"),
        )

        assert result.exit_code == 0, result.output
        assert f"{TWIN}  engine: decision  set: decision" in result.output
        assert f"{TWIN}-agent  engine: agent  set: decision" in result.output
        assert f"{TWIN}-agent  engine: agent  set: agent" in result.output
        assert "Unavailable" not in result.output
        assert len(seeded) == 2
        assert fake.call_count > 0
        assert recorded_entries() == []


class TestMachineSeed:
    def test_another_control_system_is_left_alone(self, tmp_path, monkeypatch):
        from osprey_connectors.control_system import _va_in_process_connector as module

        def untouchable():
            raise AssertionError("the connector must not be built")

        monkeypatch.setattr(module, "VAInProcessConnector", untouchable)
        config = {"control_system": {"type": "epics"}}
        seed = _MachineSeed(config, tmp_path)

        why = asyncio.run(seed.apply({"SR:CURRENT": 1.0}))

        assert why == "its machine values need the in-process simulator"
        asyncio.run(seed.close())

    def test_a_fixture_with_no_values_needs_no_simulator(self, tmp_path):
        assert asyncio.run(_MachineSeed({}, tmp_path).apply({})) is None

    def test_a_seed_reaches_another_connector_and_close_resets_it(self, tmp_path, monkeypatch):
        """The real simulator, writes allowed: what a judge's own server would read."""
        arm_switch(monkeypatch, tmp_path / "armed-data")
        monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
        from osprey_connectors.control_system._va_in_process_connector import (
            SIMULATOR_VIEW_SETTING,
            VAInProcessConnector,
        )
        from osprey_connectors.simulation.view import VARIABLES_FILE

        repo = tmp_path / "repo"
        shutil.copytree(FIXTURES / "agent-profile", repo)
        copy_bundle_data(repo / "data", "hello_world")
        assert invoke_build(repo).exit_code == 0
        build = repo / "build"
        config = yaml.safe_load((build / "config.yml").read_text(encoding="utf-8"))
        view = build / "data" / "simulator"
        channels = json.loads((view / VARIABLES_FILE).read_text(encoding="utf-8"))["channels"]
        target = next(
            c for c in channels if c.get("role") == "setpoint" and c.get("writable") is True
        )
        address = target["address"]

        async def read_through_a_second_connector() -> float:
            other = VAInProcessConnector()
            await other.connect({SIMULATOR_VIEW_SETTING: str(view)})
            try:
                return float((await other.read_channel(address)).value)
            finally:
                await other.disconnect()

        async def scenario() -> tuple[float, float, float, float, str | None]:
            start = await read_through_a_second_connector()
            low, high = target["value_range"]
            value = start + 1.0 if start + 1.0 <= high else start - 1.0
            assert value >= low
            seed = _MachineSeed(config, build)
            why = await seed.apply({address: value})
            seeded = await read_through_a_second_connector()
            await seed.close()
            return start, value, seeded, await read_through_a_second_connector(), why

        start, value, seeded, after, why = asyncio.run(scenario())

        assert why is None
        assert seeded == pytest.approx(value)
        assert after == pytest.approx(start)

    def test_a_project_that_refuses_writes_refuses_its_seed(self, tmp_path):
        """A seed is a write like any other: the reference monitor decides it."""
        repo = tmp_path / "repo"
        shutil.copytree(FIXTURES / "agent-profile", repo)
        copy_bundle_data(repo / "data", "hello_world")
        assert invoke_build(repo).exit_code == 0
        build = repo / "build"
        config = yaml.safe_load((build / "config.yml").read_text(encoding="utf-8"))
        view = build / "data" / "simulator"
        channels = json.loads((view / "variables.json").read_text(encoding="utf-8"))["channels"]
        address = next(c["address"] for c in channels if c.get("writable") is True)

        async def scenario():
            seed = _MachineSeed(config, build)
            try:
                return await seed.apply({address: 1.0})
            finally:
                await seed.close()

        why = asyncio.run(scenario())

        assert why is not None
        assert address in why

    def test_a_channel_the_simulator_refuses_is_explained(self, tmp_path):
        repo = tmp_path / "repo"
        shutil.copytree(FIXTURES / "agent-profile", repo)
        copy_bundle_data(repo / "data", "hello_world")
        assert invoke_build(repo).exit_code == 0
        build = repo / "build"
        config = yaml.safe_load((build / "config.yml").read_text(encoding="utf-8"))

        async def scenario():
            seed = _MachineSeed(config, build)
            try:
                return await seed.apply({"NOT:A:CHANNEL": 1.0})
            finally:
                await seed.close()

        why = asyncio.run(scenario())

        assert why is not None
        assert "NOT:A:CHANNEL" in why


class TestShippedCommand:
    def test_help_explains_inputs_and_columns(self):
        result = _invoke("--help")
        assert result.exit_code == 0
        for words in (
            "FIXTURES_DIR",
            "--profile",
            "--project-dir",
            "--watchdog",
            "Hit rate on risky",
            "False flags on benign",
            "p50",
            "p95",
            "Cost",
        ):
            assert words in result.output

    def test_no_test_tree_path_in_the_shipped_code(self):
        source = Path(watchdog_cmd.__file__).read_text(encoding="utf-8")
        assert "tests/" not in source
