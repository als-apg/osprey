"""``osprey watchdog``: try a watchdog on labelled prompts before enabling it.

``osprey watchdog replay`` runs every selected watchdog over every fixture, with
``enabled`` ignored, and prints how each one did. It only reports: no flag opens,
no session is held, and no verdict reaches the audit record.

The exit status says whether the replay itself worked, not whether the watchdog
is any good: it is non-zero only when an engine answered on none of its fixtures.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from osprey.cli.output import fail, note, report
from osprey.cli.repo_resolver import RepoNotFoundError, find_repo_root

if TYPE_CHECKING:
    from osprey.services.watchdog.config import WatchdogDefinition
    from osprey.services.watchdog.verdict import Verdict

_RISKY = "risky"
_BENIGN = "benign"

#: Stands in for a figure with nothing behind it.
_NO_FIGURE = "-"

_NO_PROJECT = "needs a built project; name one with --project-dir"


@dataclass(frozen=True)
class _Fixture:
    """One labelled prompt, and the session and machine it arrives in."""

    set_name: str
    label: str
    prompt: str
    nodes: list[dict[str, Any]]
    machine: Mapping[str, Any]


@dataclass
class _Row:
    """What one watchdog did on one fixture set."""

    watchdog: str
    engine: str
    set_name: str
    runs: dict[str, int] = field(default_factory=lambda: {_RISKY: 0, _BENIGN: 0})
    flagged: dict[str, int] = field(default_factory=lambda: {_RISKY: 0, _BENIGN: 0})
    latencies: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    unavailable: Counter[str] = field(default_factory=Counter)
    not_run: Counter[str] = field(default_factory=Counter)

    def add(self, label: str, verdict: Verdict) -> None:
        self.latencies.append(verdict.latency_s)
        if verdict.cost_usd is not None:
            self.costs.append(verdict.cost_usd)
        if verdict.status == "unavailable":
            self.unavailable[str(verdict.error_type)] += 1
            return
        self.runs[label] += 1
        self.flagged[label] += int(verdict.flagged)

    @property
    def attempted(self) -> int:
        return sum(self.runs.values()) + sum(self.unavailable.values())


class _Refusal(Exception):
    """The replay cannot start; carries the parts of its failure message."""

    def __init__(self, summary: str, cause: str | None = None, remedy: str | None = None) -> None:
        super().__init__(summary)
        self.parts = (summary, cause, remedy)


class _MachineSeed:
    """Puts a fixture's channel values on the project's simulator.

    A seed is an ordinary connector write, decided by the reference monitor like
    any other, and the in-process simulator journals it where the agent's
    control-system server reads it, so a judge reads what was seeded. Only that
    simulator is ever written: any other control system answers why it was left
    alone, whatever the fixture asks for.
    """

    def __init__(self, config: Mapping[str, Any], build_dir: Path) -> None:
        self._config = config
        self._build_dir = build_dir
        # Any: ``reset`` belongs to the simulator alone, and _is_simulator admits only it.
        self._connector: Any = None
        self._seeded = False

    def _is_simulator(self) -> bool:
        from osprey_connectors.types import VIRTUAL_ACCELERATOR

        system = self._config.get("control_system")
        if not isinstance(system, Mapping) or system.get("type") != VIRTUAL_ACCELERATOR:
            return False
        connector = system.get("connector")
        settings = connector.get(VIRTUAL_ACCELERATOR) if isinstance(connector, Mapping) else None
        return isinstance(settings, Mapping) and settings.get("serving") == "in_process"

    async def apply(self, machine: Mapping[str, Any]) -> str | None:
        """Start the simulator from *machine*; why it could not, ``None`` when it did."""
        if not machine and not self._seeded:
            return None
        if not self._is_simulator():
            return "its machine values need the in-process simulator"
        try:
            connector = await self._connect()
            self._seeded = True
            await connector.reset()
            for channel, value in machine.items():
                written = await connector.write_channel(channel, value, confirm=False)
                if written.outcome in ("refused", "failed"):
                    return f"the simulator would not take {channel}: {written.error_message}"
        except Exception as exc:
            return f"the simulator could not be seeded: {type(exc).__name__}"
        return None

    async def _connect(self) -> Any:
        if self._connector is None:
            from osprey_connectors.factory import ConnectorFactory
            from osprey_connectors.simulation.view import VIEW_RELPATH
            from osprey_connectors.types import TARGET_VA

            connector, settings = ConnectorFactory.build_control_system_connector(
                dict(self._config["control_system"]), control_target=TARGET_VA
            )
            view = self._build_dir / VIEW_RELPATH
            await connector.connect({**settings, "simulator_view": str(view)})
            self._connector = connector
        return self._connector

    async def close(self) -> None:
        """Hand the simulator back in its start state."""
        connector, self._connector = self._connector, None
        if connector is None:
            return
        try:
            await connector.reset()
        finally:
            await connector.disconnect()


def _read_fixtures(root: Path) -> list[_Fixture]:
    """Every ``*.jsonl`` fixture under *root*, in path order."""
    fixtures = []
    for path in sorted(root.rglob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            header = json.loads(lines[0])
            nodes = [json.loads(line) for line in lines[1:]]
        except (OSError, UnicodeDecodeError, ValueError, IndexError) as exc:
            raise _Refusal(f"Cannot read fixture {path}", type(exc).__name__) from exc
        if (
            not isinstance(header, dict)
            or header.get("label") not in (_RISKY, _BENIGN)
            or not isinstance(header.get("prompt"), str)
        ):
            raise _Refusal(
                f"Cannot read fixture {path}",
                "Its first line must be an object with a risky or benign label and a prompt.",
            )
        relative = path.parent.relative_to(root).as_posix()
        fixtures.append(
            _Fixture(
                set_name=root.name if relative == "." else relative,
                label=header["label"],
                prompt=header["prompt"],
                nodes=nodes,
                machine=header.get("machine") or {},
            )
        )
    if not fixtures:
        raise _Refusal(f"No fixtures under {root}", "A fixture is a *.jsonl file.")
    return fixtures


def _load_definitions(
    profiles: Sequence[Path], project_dir: Path | None, selected: str | None
) -> tuple[WatchdogDefinition, ...]:
    """The watchdogs to replay, each switched on: a replay comes before enabling."""
    from osprey.services.watchdog.definitions import load_profile_watchdogs
    from osprey_connectors.errors import BuildProfileError

    try:
        roots = list(profiles) or [find_repo_root(project_dir)]
    except RepoNotFoundError as exc:
        raise _Refusal("No profile to replay", str(exc), "Name one with --profile DIR.") from exc
    loaded: dict[str, WatchdogDefinition] = {}
    for root in roots:
        try:
            definitions = load_profile_watchdogs(root)
        except BuildProfileError as exc:
            raise _Refusal(f"Cannot load the watchdogs of {root}", str(exc)) from exc
        for defn in definitions:
            if defn.name in loaded:
                raise _Refusal(f"Two profiles define the watchdog {defn.name}")
            loaded[defn.name] = replace(defn, enabled=True)
    if selected is not None and selected not in loaded:
        known = ", ".join(sorted(loaded)) or "none"
        raise _Refusal(f"No watchdog named {selected}", f"These profiles define: {known}.")
    chosen = [loaded[selected]] if selected is not None else list(loaded.values())
    if not chosen:
        raise _Refusal("These profiles define no watchdogs")
    return tuple(chosen)


async def _replay(
    definitions: Iterable[WatchdogDefinition],
    fixtures: Sequence[_Fixture],
    *,
    config: Mapping[str, Any],
    build_dir: Path | None,
    agent_build_dir: Path | None,
) -> dict[tuple[str, str], _Row]:
    """Run the definitions over the fixtures, and tally each pair of watchdog and set."""
    from osprey.services.watchdog.config import ENGINE_AGENT
    from osprey.services.watchdog.history import SessionNodes
    from osprey.services.watchdog.runner import WatchdogRunner, prompt_budget_s

    def runner(selected: list[WatchdogDefinition]) -> WatchdogRunner:
        return WatchdogRunner(
            selected,
            recorder=lambda _verdict: None,
            hold_root=None,
            notify=lambda _frame: None,
            config=config,
            build_dir=agent_build_dir,
        )

    agents = [defn for defn in definitions if defn.engine == ENGINE_AGENT]
    others = [defn for defn in definitions if defn.engine != ENGINE_AGENT]
    decision_runner, agent_runner = runner(others), runner(agents)
    seed = _MachineSeed(config, agent_build_dir or build_dir or Path())
    budget_s = prompt_budget_s(config)
    rows: dict[tuple[str, str], _Row] = {}

    def row_for(defn: WatchdogDefinition, fixture: _Fixture) -> _Row:
        return rows.setdefault(
            (defn.name, fixture.set_name), _Row(defn.name, defn.engine, fixture.set_name)
        )

    async def run(
        chosen: WatchdogRunner, selected: list[WatchdogDefinition], fixture: _Fixture
    ) -> None:
        if not selected:
            return
        task = chosen.on_nodes(
            session_key=None,
            record_session_id=str(uuid.uuid4()),
            submission_id=str(uuid.uuid4()),
            nodes=SessionNodes(nodes=fixture.nodes),
            prompt=fixture.prompt,
            deadline=time.monotonic() + budget_s,
            can_hold=False,
        )
        by_name = {defn.name: defn for defn in selected}
        for verdict in await task:
            row_for(by_name[verdict.watchdog], fixture).add(fixture.label, verdict)

    try:
        for fixture in fixtures:
            await run(decision_runner, others, fixture)
            why_not = _NO_PROJECT if agent_build_dir is None else None
            if agents and why_not is None:
                why_not = await seed.apply(fixture.machine)
            if why_not is None:
                await run(agent_runner, agents, fixture)
            else:
                for defn in agents:
                    row_for(defn, fixture).not_run[why_not] += 1
    finally:
        await decision_runner.shutdown()
        await agent_runner.shutdown()
        await seed.close()
    return rows


def _percentile(values: Sequence[float], percent: int) -> float:
    """The nearest-rank percentile of *values*, which holds at least one."""
    ordered = sorted(values)
    return ordered[math.ceil(percent / 100 * len(ordered)) - 1]


def _rate(flagged: int, runs: int) -> str:
    if not runs:
        return f"{_NO_FIGURE} (no answers)"
    return f"{flagged / runs * 100:.0f} % ({flagged} of {runs})"


def _report_row(row: _Row) -> None:
    latencies = row.latencies
    p50 = f"{_percentile(latencies, 50):.2f} s" if latencies else _NO_FIGURE
    p95 = f"{_percentile(latencies, 95):.2f} s" if latencies else _NO_FIGURE
    cost = f"${sum(row.costs):.4f}" if row.costs else _NO_FIGURE
    report(f"{row.watchdog}  engine: {row.engine}  set: {row.set_name}")
    report(f"  Hit rate on risky      {_rate(row.flagged[_RISKY], row.runs[_RISKY])}")
    report(f"  False flags on benign  {_rate(row.flagged[_BENIGN], row.runs[_BENIGN])}")
    report(f"  Latency p50            {p50}")
    report(f"  Latency p95            {p95}")
    report(f"  Cost                   {cost}")
    unavailable = sum(row.unavailable.values())
    if unavailable:
        why = ", ".join(sorted(row.unavailable))
        report(f"  Unavailable            {unavailable} of {row.attempted} ({why})")
    if row.not_run:
        why = "; ".join(sorted(row.not_run))
        report(f"  Not run                {sum(row.not_run.values())} ({why})")


def _silent_engines(rows: Iterable[_Row]) -> list[str]:
    """The engines that ran and answered on none of their fixtures."""
    answered: Counter[str] = Counter()
    attempted: Counter[str] = Counter()
    for row in rows:
        answered[row.engine] += sum(row.runs.values())
        attempted[row.engine] += row.attempted
    return sorted(name for name, count in attempted.items() if count and not answered[name])


@click.group()
def watchdog() -> None:
    """Try safety watchdogs out before relying on them."""


@watchdog.command("replay")
@click.argument(
    "fixtures_dir",
    metavar="FIXTURES_DIR",
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
)
@click.option("--watchdog", "selected", metavar="NAME", help="Replay this watchdog only.")
@click.option(
    "--profile",
    "profiles",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
    help="Directory holding a profile.yml with a watchdogs: block. Repeatable. "
    "Default: the project's profile.",
)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
    help="Built project agent watchdogs run in. Without one they are reported as not run.",
)
def replay(
    fixtures_dir: Path,
    selected: str | None,
    profiles: tuple[Path, ...],
    project_dir: Path | None,
) -> None:
    """Replay labelled prompts through watchdogs, enabled or not, and report.

    FIXTURES_DIR holds *.jsonl fixtures at any depth; each directory is one set.
    A fixture's first line is {"label": "risky" or "benign", "prompt": ...,
    "machine": {channel: value}}, with machine optional, and every further line
    is one node of the session the prompt arrives in. Agent watchdogs read a
    fixture's machine values from the project's in-process simulator, which is
    written before the run and reset afterwards. Those writes need write access
    to a writable channel; a fixture they cannot apply to, or any other control
    system, is left alone and reported as not run for agent watchdogs.

    Every selected watchdog runs over every fixture. For each watchdog, engine
    and set it prints:

    \b
      Hit rate on risky      share of risky prompts flagged
      False flags on benign  share of benign prompts flagged
      Latency p50, p95       seconds per judgement
      Cost                   total across the set, where known
      Unavailable            runs with no answer, and why (only when any)
      Not run                runs left out, and why (only when any)

    Rates count answered runs. Nothing is flagged, held or recorded. The exit
    status is non-zero only when an engine answered on none of its fixtures.
    """
    try:
        fixtures = _read_fixtures(fixtures_dir)
        definitions = _load_definitions(profiles, project_dir, selected)
    except _Refusal as refusal:
        fail(*refusal.parts)
        raise click.exceptions.Exit(2) from refusal

    root = project_dir
    if root is None:
        try:
            root = find_repo_root()
        except RepoNotFoundError:
            root = None
    from osprey.services.watchdog.definitions import load_rendered_config

    loaded = load_rendered_config(root) if root is not None else None
    config, build_dir = loaded if loaded is not None else ({}, None)
    agent_build_dir = build_dir if project_dir is not None else None

    rows = asyncio.run(
        _replay(
            definitions,
            fixtures,
            config=config,
            build_dir=build_dir,
            agent_build_dir=agent_build_dir,
        )
    )
    for row in rows.values():
        _report_row(row)
    note(f"{len(fixtures)} fixtures. Nothing was flagged, held or recorded.")

    silent = _silent_engines(rows.values())
    if silent:
        fail(f"No answer from the {', '.join(silent)} engine on any fixture")
        raise click.exceptions.Exit(1)
