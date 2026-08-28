"""The honest archiver world, proved end to end against a deployed stack.

Every other test of this feature checks one half of it. This one deploys the
whole thing — store, seeder, recorder, virtual accelerator — and asks the
questions an operator would: is the history there, is it affordable, does what I
just did to the machine show up in it, and does it admit what it does not know.

Eight claims, in the order a deployment makes them true:

* **The seed is affordable.** A first ``osprey up`` seeds the base series inside
  a measured budget and says so while it works. This lane keeps DEFAULT
  ``va_archiver`` knobs precisely so the numbers mean something: the CI lanes
  that shrink retention to run fast cannot own a budget nobody would deploy.
* **The store is affordable.** The seeded archive fits the declared disk budget,
  and the block compressor the manifest claims is the one mongod actually used.
* **The live half reaches the stored half.** A setpoint written now appears in
  an archiver read within the recorder's budget — the whole point of running a
  recorder rather than only a seeder.
* **The archive admits its edges.** A window before coverage begins returns no
  points, and ``get_metadata`` reports the oldest sample it really holds rather
  than a declared window.
* **The seam is invisible.** Where seeded history ends and recorded reality
  begins, the step in every fidelity partition is attributable to noise — the
  threshold quoted from :func:`~osprey.simulation.procedural.deviation_bound`
  per channel, never a constant embedded here.
* **A scenario apply stays bounded.** Activating an eventful scenario rewrites
  event windows, not the archive; the document count is asked to stay inside a
  ratio, which is the budget the disk assertion above rests on.
* **The recorder records a machine, not whatever is plugged in.** Flipping
  ``control_system.type`` to ``mock`` on the mounted config stops recording, and
  flipping back resumes it — with no redeploy and no restart. How many polls
  that takes is deliberately not asserted: the test waits for the recorder's own
  announcement, because a poll landing mid-write keeps its previous answer and
  the service never promised a bound. The claim is that the flip takes effect at
  all, and it matters because synthesized mock values written into a store the
  agent reads as machine history is the same lie this whole feature exists to
  refuse, arriving through a different door.
* **An operator can see all of it.** ``osprey health`` reports the archive fresh
  through a check nobody declared by hand: naming a canary channel in the
  ``va_archiver:`` block derived the category, the probe and a staleness
  threshold from the recorder's own cadence. Reachable and being-written are
  different claims, and this is the surface that tells them apart.

Gating: needs Docker. The VA image is pinned ``linux/amd64`` (see its
Dockerfile's architecture note), so on Apple Silicon it builds and runs under
emulation — slow on a cold cache. There is no LLM anywhere in this file and no
network beyond the containers: every assertion is mechanical, which is why the
CI lane carries no provider secret.

Container safety: every docker invocation names an exact container, image or
volume — never a wildcard, never ``system prune``, never ``down --volumes``
(which would reach every volume in the project). Teardown goes through
``osprey down``, matching every other e2e in this directory, plus the
removal of the store's one named volume — see ``_discard_store_volume`` for why
a fresh store is a precondition of these assertions rather than housekeeping.

TTL note: this file deploys the REAL compose stack, where mongod runs with the
template's own flags and its TTL monitor is live. That is deliberate — retention
behaving like retention is part of what is under test — and it is why the
document-count assertion below is a one-sided ratio rather than an equality: the
sweeper may legitimately retire aged tail samples between two counts. Auxiliary
fixture containers elsewhere (see
``tests/connectors/test_archiver_world_contracts.py``) switch the monitor off
for the opposite reason; do not copy that setting into this file.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from osprey.simulation.procedural import DEFAULT_NOISE_LEVEL, deviation_bound

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    # dockerbuild: full VA image build + a real multi-service deploy -- runs in
    # the dedicated archiver-world-e2e CI job, never the shared e2e-tests lane
    # (the marker->--ignore pairing is enforced by
    # tests/deployment/test_ci_workflow_wiring.py).
    pytest.mark.dockerbuild,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

PROJECT_NAME = "archiver-world"

# Ports chosen away from the ones a host demo stack habitually holds
# (5064 EPICS CA, 5080, 5432 postgres): a collision there reads as a deploy
# failure and costs an afternoon before anyone suspects the host.
VA_CA_PORT = 5164
MONGO_PORT_HOST = 27117

BUILD_TIMEOUT_SEC = 600
DEPLOY_UP_TIMEOUT_SEC = 2400

# The measured budgets this lane owns. Both are stated against DEFAULT
# va_archiver knobs (30-day retention, 48 h hot span) -- see the module
# docstring for why the fast lanes cannot own them.
SEED_BUDGET_SEC = 180.0
DISK_BUDGET_BYTES = 2 * 1024**3

# From a setpoint write to that value being readable out of the archive. Covers
# the recorder's own cadence plus one poll, not an arbitrary "should be quick".
RECORDER_BUDGET_SEC = 30.0
RECORDER_POLL_SEC = 2.0

#: Budget for one `osprey health --category archiver` run. Generous next to the
#: probe's own 10 s archiver query: the console script pays a cold interpreter
#: and a full config load first, and this is a timeout, not a measurement.
HEALTH_TIMEOUT_SEC = 120.0

#: Canary channel the derived ``archiver_freshness`` check reads. Named here
#: rather than picked from the deployed manifest the way the write test picks
#: its setpoint, because it has to travel into the build's `--override` before
#: any project exists to read a manifest from. The test asserts it really is a
#: channel this machine model serves, so a model that drops it fails loudly
#: instead of reporting a mystery "no samples in the window".
FRESHNESS_CANARY = "SR:DIAG:DCCT:01:CURRENT:RB"

# How far the document count may grow when an eventful scenario is applied.
# One-sided on purpose: densification inserts a bounded number of samples inside
# each event window, while a live TTL sweeper may retire aged tail samples in
# the same interval, so a small net SHRINK is correct behavior. The failure this
# guards is the whole-archive densification class, which multiplies the count
# several-fold -- orders of magnitude outside this bound either way.
APPLY_GROWTH_RATIO = 1.25

# Share of a scenario's intended dense samples that may fall outside coverage.
# See the assertion in the scenario-apply test for why this is not zero: a real
# outage sits nowhere near 5%, while the archive's leading edge routinely does.
UNCOVERED_FRACTION = 0.05

# Channels are sampled per fidelity partition straight from the manifest's own
# `partition` field, so the test needs no classification logic of its own and
# cannot drift from how the manifest classifies.
PARTITIONS = ("pyat-coupled", "sp-echo", "static-noisy")
SEAM_SAMPLES_PER_PARTITION = 3

# How far before the seed anchor the seam window starts. Only needs to be wide
# enough to hold a few seeded samples at the hot cadence; the window's other end
# is the newest sample, so the recorded side needs no allowance.
SEAM_LEAD_TIME = timedelta(minutes=5)

# `seeded 1,234 documents x 2,908 channels (... ) in 12.3s` -- the seeder's own
# report line. Parsed rather than timing `osprey up` as a whole, which would
# fold in a multi-minute image build and measure the wrong thing entirely.
#
# Matched against NORMALIZED output, never the raw capture. The deploy logs
# through a rich console, which hard-wraps each message to the console width and
# pads the continuation with the log gutter's spaces -- so this line arrives
# split across two or three physical lines with ANSI colour codes through it.
# A line-oriented regex silently finds nothing and reads as "the seed never ran".
_SEED_REPORT_RE = re.compile(r"seeded ([\d,]+) documents x ([\d,]+) channels .* in ([\d.]+)s")
_SEED_PROGRESS_RE = re.compile(r"seeding archive: [\d,]+ documents written")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# The recorder announces each enablement transition with exactly one line, and
# says nothing in steady state. Those two lines are the only direct evidence of
# WHEN it noticed a config flip -- see `_await_recorder_transition` for why a
# test about idling must wait for the announcement rather than assume a latency.
#
# Matched against NORMALIZED log output (see `_recorder_logs`), and kept to the
# opening phrase of each message: rich wraps the rest across physical lines, and
# a pattern reaching past the wrap would depend on the console width the
# container happened to render at.
_RECORDER_IDLE_RE = re.compile(r"Idle, writing nothing")
_RECORDER_RECORDING_RE = re.compile(r"Recording \d+ channels every \d+s")


def _normalize(text: str) -> str:
    """Strip ANSI and fold every run of whitespace, so wrapped log lines rejoin."""
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", text))


def _override_yaml() -> str:
    """The ``--override`` content that opts this project into the stored archive.

    Four decisions, each load-bearing:

    ``va_archiver:`` declares the store — which injects the mongodb and
    archiver_recorder services and renders the connector's connection keys.
    Declaring it carries NO knob overrides on purpose: this lane owns the
    measured budgets, and shrinking retention here would make them meaningless.
    ``freshness_channel`` is not such a knob — it changes no size and no cadence,
    it only names the canary the derived ``archiver_freshness`` health check
    reads, and without it that check is never derived and cannot be proven here.

    ``archiver.type`` must be set SEPARATELY. Declaring where an archive lives is
    not the same decision as selecting it as the deployment's archiver, so
    ``va_archiver_config_overrides`` deliberately leaves the type alone — a
    project that only declared the block would deploy the store and then read the
    mock, and every assertion here would pass against synthesized data while the
    real store sat empty beside it.

    ``deployed_services: []`` plus the nulled blocks trims the stack to exactly
    the archiver world. Config overrides are applied BEFORE the build's service
    injectors run, so emptying the list and letting the VA and archiver
    injectors append leaves precisely ``[virtual_accelerator, mongodb,
    archiver_recorder]`` — verified in the built config, not assumed.

    ``virtual_accelerator.live_standin: null`` switches the preset's live
    stand-in off — the delete-the-line escape the profile documents, spelled as
    a null because an override cannot remove a key. The preset ships a second
    simulator as the ``live`` target, and the recorder follows the machine the
    deployment calls live: with the stand-in deployed, the setpoints this lane
    writes to the sandbox VA would never reach the archive it then reads.
    Nulling the key keeps the recorder on the VA under test, and keeps a second
    emulated container out of the build.

    Trimming ``deployed_services`` does not switch off the preset's consumers
    of what it trimmed away, and the build refuses a deploying render whose
    consumer is on for a service it does not run — the ARIEL database and
    hybrid search (``ariel:`` nulled, the section being both consumers'
    switch), the OTLP exporter and the bluesky MCP server are switched off
    here by the keys that refusal names. Nothing in this lane dials any of
    them.

    That trim is not only about speed. The preset's full stack publishes
    postgres, openobserve, the bluesky bridge, Tiled and the panels on fixed
    host ports, and a developer box running any OSPREY demo stack already holds
    them: the deploy then aborts on a port preflight that has nothing to do with
    archiving. Deploying only what is under test makes this lane runnable
    beside a live demo, and cuts several image builds out of a 45-minute cap.
    """
    return (
        "config:\n"
        "  control_system.type: virtual_accelerator\n"
        "  archiver.type: mongodb_archiver\n"
        "  modules.web_terminals.enabled: false\n"
        "  deployed_services: []\n"
        "  ariel:\n"
        "  claude_code.telemetry.enabled: false\n"
        "  claude_code.servers.bluesky.enabled: false\n"
        "va_archiver:\n"
        f"  port_host: {MONGO_PORT_HOST}\n"
        f"  freshness_channel: {FRESHNESS_CANARY}\n"
        "virtual_accelerator:\n"
        "  live_standin: null\n"
        "bluesky: null\n"
        "bluesky_web: null\n"
        "dispatch: null\n"
    )


def _build_project(output_dir: Path) -> Path:
    """``osprey init`` + ``osprey build`` the opt-in repo, out of process.

    Out of process (not ``CliRunner``) because the deploy that follows needs a
    repo on disk built by the same console script an operator would run.
    Returns the deployment REPO root; the render it starts from is
    ``<repo>/build``.

    ``--dev`` is a property of the RENDER, not of the start: the deploy below
    runs ``up --dev`` so the recorder under test is this checkout, and ``up``
    never re-renders — a pinned render would be refused there rather than
    quietly deploying the published release.
    """
    from tests.e2e import _orm_stack

    override_path = output_dir / "archiver-override.yml"
    override_path.write_text(_override_yaml(), encoding="utf-8")
    osprey_bin = _orm_stack.find_osprey_console_script()

    repo = output_dir / PROJECT_NAME
    for label, argv in (
        (
            "osprey init",
            [
                str(osprey_bin),
                "init",
                str(repo),
                "--preset",
                "control-assistant",
                "--no-git",
                "--override",
                str(override_path),
                "--set",
                f"virtual_accelerator.port={VA_CA_PORT}",
            ],
        ),
        (
            "osprey build",
            [
                str(osprey_bin),
                "build",
                "--repo",
                str(repo),
                "--skip-deps",
                "--skip-lifecycle",
                "--dev",
            ],
        ),
    ):
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SEC,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if result.returncode != 0:
            raise AssertionError(
                f"{label} failed (rc={result.returncode}):\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            )
    return repo


class DeployedArchiverWorld:
    """The deployed repo, plus what the seed reported on its way up."""

    def __init__(self, repo: Path, deploy_output: str):
        self.repo = repo
        self.deploy_output = deploy_output
        self.config = _load_config(repo)

    @property
    def store(self) -> dict[str, Any]:
        """Connection parameters, resolved exactly as the deploy resolves them."""
        from osprey.simulation.apply import archiver_store_config

        store = archiver_store_config(self.config, self.repo)
        assert store is not None, "deployed project declares no mongodb_archiver block"
        return store


def _load_config(repo: Path) -> dict[str, Any]:
    """The as-built config the deploy runs on — the render, never the source."""
    from osprey.utils.config import load_project_config

    return load_project_config(str(repo / "build" / "config.yml"), wrap_errors=True)


@pytest.fixture(scope="module")
def archiver_world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedArchiverWorld]:
    """Build and deploy the opt-in project once for every assertion below."""
    from osprey.deployment.compose_generator import resolve_project_name
    from tests.e2e import _orm_stack
    from tests.e2e._deploy_diagnostics import dead_container_logs

    project = resolve_project_name({"project_name": PROJECT_NAME})
    # The store's named volume, spelled EXACTLY — never a wildcard, never a
    # prune, never `down --volumes` (which would reach every volume in the
    # project). Derived from the compose project name so it cannot drift.
    store_volume = f"{project}_archiver_mongodb_data"

    def _dead_containers() -> str:
        """Logs from every container of THIS deployment that is not running."""
        return dead_container_logs(project)

    def _discard_store_volume() -> None:
        """Remove the store's volume so this run starts from a genuinely fresh one.

        Not housekeeping — a correctness precondition. Every run mints a NEW
        MONGO_ROOT_PASSWORD into its own temporary project, but the named volume
        is per-compose-project and survives `osprey down`. A volume left by a
        previous run therefore keeps the credentials it was initialized with, and
        the next deploy is refused by its own store: exactly the stale-volume
        case the deploy reports, arriving here as a self-inflicted one.

        It also protects what this file measures. These tests assert a FIRST
        deploy — a seed that runs, reports and is timed. Against a surviving
        volume the fingerprint would MATCH, the seed would be skipped entirely,
        and the budget assertions would have nothing to measure.
        """
        subprocess.run(["docker", "volume", "rm", "-f", store_volume], capture_output=True)

    base = tmp_path_factory.mktemp("archiver_world_build")
    repo = _build_project(base)

    # The trim `_override_yaml` describes, checked in the BUILT config before
    # anything is deployed rather than taken on trust. The essential members are
    # pinned indirectly — a missing mongodb or recorder reds several tests below
    # — but an EXTRA service is what this catches, and an extra service is the
    # failure that costs an afternoon: it lands on a fixed host port a developer
    # box is already holding and the deploy aborts on a preflight that has
    # nothing to do with archiving.
    built_services = sorted(_load_config(repo).get("deployed_services") or [])
    assert built_services == ["archiver_recorder", "mongodb", "virtual_accelerator"], (
        f"the built project deploys {built_services}, not the archiver world this lane "
        "trims to; config overrides run BEFORE the service injectors, so an emptied "
        "deployed_services plus the VA and archiver injectors must leave exactly these three"
    )

    osprey_bin = _orm_stack.find_osprey_console_script()

    # Force a fresh --dev image so the deployed recorder runs CURRENT source.
    # Exact-named image only. E2E_REUSE_IMAGES=1 skips it for fast local
    # iteration on the test itself; never set it in CI.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.va_image(PROJECT_NAME)],
            capture_output=True,
            text=True,
        )

    # Before the deploy, not only after: a run that died without reaching its
    # teardown (a killed session, a crashed fixture) leaves the volume behind,
    # and the next run must not inherit it.
    _discard_store_volume()

    try:
        up = subprocess.run(
            [str(osprey_bin), "up", "-d", "--dev"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}\n"
                f"--- containers that are not running ---\n{_dead_containers()}"
            )
        yield DeployedArchiverWorld(repo, up.stdout + up.stderr)
    finally:
        down = subprocess.run(
            [str(osprey_bin), "down"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=300,
        )
        _discard_store_volume()
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


@pytest.fixture(autouse=True)
def store_credential_in_env(archiver_world, monkeypatch):
    """Export the store's password under the NAME the connector reads.

    The connector takes the name of the variable holding the password, never the
    password itself — the same indirection the rendered config uses, so a secret
    never has to be spelled in a config file. Exporting it is what makes the
    reads below the agent's real path rather than a special case.

    Through ``monkeypatch`` rather than a bare ``os.environ`` write, and that is
    not tidiness. A bare write outlives this module for the rest of the pytest
    session, and ``_archiver_store_connection`` falls back to the ambient
    environment whenever a project's ``.env`` carries no password — so a later
    e2e bringing up its own store could authenticate against it with THIS
    deployment's credential, and would report whatever that produced as its own
    result.
    """
    store = archiver_world.store
    monkeypatch.setenv(store["password_env"], store["password"])


# ---------------------------------------------------------------------------
# Store access helpers
# ---------------------------------------------------------------------------


def _collection(world: DeployedArchiverWorld):
    from osprey.simulation.apply import archiver_collection

    return archiver_collection(world.store)


def _connector(world: DeployedArchiverWorld):
    """A connected ``MongoDBArchiverConnector``, the agent's own read path.

    Read through the connector rather than pymongo wherever the assertion is
    about what a READER sees: the store holding a value and the connector
    surfacing it are different claims, and only the second one is what an
    operator experiences.
    """
    from osprey.connectors.archiver.mongodb_archiver_connector import MongoDBArchiverConnector

    store = world.store
    connector = MongoDBArchiverConnector()
    asyncio.run(
        connector.connect(
            {
                "host": store["host"],
                "port": store["port"],
                "name": store["database"],
                "collection": store["collection"],
                "auth": store["auth_database"],
                "username": store["username"],
                "password_env": store["password_env"],
            }
        )
    )
    return connector


def _recorder_container() -> str:
    """The recorder's container, spelled exactly and derived from the project name."""
    from osprey.deployment.compose_generator import resolve_project_name

    return f"{resolve_project_name({'project_name': PROJECT_NAME})}-archiver-recorder"


def _recorder_logs(tail: str | None = None) -> str:
    """The recorder's own log, by exact container name, NORMALIZED.

    Normalized for the same reason the seed report above is: the service logs
    through a rich console, which both hard-wraps each message and injects
    colour codes INSIDE the text it highlights — ``control_system.type=`` leaves
    the container with escape sequences sitting between the ``.`` and the
    ``type``. A regex run over the raw capture therefore matches nothing, and
    reads as "the recorder never said that", which is the most misleading answer
    available here.
    """
    command = ["docker", "logs"]
    if tail is not None:
        command += ["--tail", tail]
    logs = subprocess.run([*command, _recorder_container()], capture_output=True, text=True)
    return _normalize(logs.stdout + logs.stderr)


def _recorder_transitions(pattern: re.Pattern[str]) -> int:
    """How many times the recorder has announced this state so far."""
    return len(pattern.findall(_recorder_logs()))


def _await_recorder_transition(
    pattern: re.Pattern[str], baseline: int, timeout_sec: float, what: str
) -> float:
    """Block until the recorder announces ``what`` again, and report how long it took.

    Waiting for the recorder's OWN announcement, rather than sleeping a fixed
    multiple of the poll interval, is what makes the assertion that follows about
    the behaviour under test instead of about propagation speed. The service
    deliberately does not bound that speed at one poll: a poll landing mid-write
    reads a torn config and keeps its previous answer until the next one (see
    ``read_control_system_type``), so "two polls" is an assumption the product
    never made, and a test resting on it fails on an unlucky interleaving while
    reporting a recorder fault that did not happen.

    A flip that never arrives still fails, and says which of the two it was.
    """
    started = time.monotonic()
    deadline = started + timeout_sec
    while True:
        if _recorder_transitions(pattern) > baseline:
            return time.monotonic() - started
        assert time.monotonic() < deadline, (
            f"the recorder never announced {what} within {timeout_sec:.0f}s of the "
            f"control_system.type flip. It re-reads the mounted config on an interval, so "
            f"this is the flip not reaching the container at all -- not a slow recorder.\n"
            f"--- {_recorder_container()} logs ---\n{_recorder_logs(tail='40')}"
        )
        time.sleep(RECORDER_POLL_SEC)


def _as_utc(moment: datetime) -> datetime:
    """Pin the zone on a datetime pymongo may have handed back naive.

    Read as UTC, which is what the store holds. Naive and aware values do not
    compare, so a value that crosses into arithmetic with ``datetime.now(UTC)``
    has to be pinned once, deliberately, rather than depending on which layer
    happened to attach a zone.
    """
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


def _seed_anchor(world: DeployedArchiverWorld) -> datetime:
    """The instant seeded history ends and recorded reality begins.

    Read from the seed manifest the seeder writes into the store, so the seam
    test can put its window ACROSS the seam rather than wherever the clock
    happens to be when it runs.
    """
    from osprey.simulation.archiver_seed import MANIFEST_ID

    with _collection(world) as collection:
        manifest = collection.find_one({"_id": MANIFEST_ID})
    assert manifest is not None, "the store holds no seed manifest, so there is no seam to locate"
    anchor = manifest.get("seeded_at")
    assert anchor is not None, "the seed manifest carries no seeded_at anchor"
    return _as_utc(anchor)


def _manifest_channels(world: DeployedArchiverWorld) -> list[dict[str, Any]]:
    path = world.repo / "build" / "data" / "simulation" / "channel_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["channels"]


def _sample_per_partition(world: DeployedArchiverWorld) -> dict[str, list[dict[str, Any]]]:
    """A few channels from each fidelity partition, taken in manifest order.

    Deterministic rather than random: a seam failure has to be reproducible from
    the test name alone, and a random sample turns "this partition is broken"
    into "this run was unlucky".
    """
    by_partition: dict[str, list[dict[str, Any]]] = {name: [] for name in PARTITIONS}
    for channel in _manifest_channels(world):
        bucket = by_partition.get(str(channel.get("partition")))
        if bucket is not None and len(bucket) < SEAM_SAMPLES_PER_PARTITION:
            bucket.append(channel)
    return by_partition


# ---------------------------------------------------------------------------
# The seed: affordable in time, affordable on disk
# ---------------------------------------------------------------------------


def test_base_seed_completes_within_budget_and_reports_progress(archiver_world):
    """A first deploy seeds inside the measured budget, and says so while it works.

    The duration is the seeder's OWN reported elapsed time, not the wall time of
    ``osprey up``: that call also builds an image, and folding a cold image cache
    into an archive budget would measure the runner, not the feature.

    Progress is asserted too, and is not cosmetic. A silent multi-minute step is
    indistinguishable from a hang to the operator watching it, and the first
    thing they do about a hang is kill it -- which leaves a half-seeded store.
    """
    output = _normalize(archiver_world.deploy_output)
    report = _SEED_REPORT_RE.search(output)
    assert report, (
        "no seed report in `osprey up` output -- the staged bring-up did not seed.\n"
        # The whole capture, not a tail slice: the seed happens EARLY in the
        # deploy and the compose build that follows is far longer, so a tail
        # would cut away exactly the evidence this assertion is about.
        f"--- normalized output ---\n{output}"
    )

    documents = int(report.group(1).replace(",", ""))
    channels = int(report.group(2).replace(",", ""))
    elapsed_s = float(report.group(3))

    assert documents > 0, "seed reported zero documents"
    assert channels > 0, "seed reported zero channels"
    assert elapsed_s <= SEED_BUDGET_SEC, (
        f"base seed took {elapsed_s:.1f}s at default knobs, over the "
        f"{SEED_BUDGET_SEC:.0f}s budget this lane owns "
        f"({documents:,} documents x {channels:,} channels)"
    )
    assert _SEED_PROGRESS_RE.search(output), (
        "seed emitted no progress lines; a silent multi-minute step reads as a hang"
    )
    print(  # noqa: T201
        f"\nMEASURED base seed: {elapsed_s:.1f}s of the {SEED_BUDGET_SEC:.0f}s budget "
        f"({documents:,} documents x {channels:,} channels)"
    )


def test_seeded_archive_fits_the_disk_budget_with_the_declared_compressor(archiver_world):
    """Size and codec together, because the budget only holds if the codec did.

    A store that fits because it is half-empty and a store that fits because zstd
    did its job look identical on the size assertion alone. Reading the
    compressor mongod actually used is what separates them -- and it is a SERVER
    start flag, so getting it wrong is a compose-template regression no unit test
    of the seeder could catch.
    """
    with _collection(archiver_world) as collection:
        stats = collection.database.command("collStats", collection.name)
        documents = collection.count_documents({})

    on_disk = int(stats.get("storageSize", 0)) + int(stats.get("totalIndexSize", 0))
    assert documents > 0, "store is empty -- nothing was seeded"
    assert on_disk <= DISK_BUDGET_BYTES, (
        f"stored archive is {on_disk / 1024**3:.2f} GiB at default knobs, over the "
        f"{DISK_BUDGET_BYTES / 1024**3:.0f} GiB budget this lane owns"
    )

    block_compressor = (
        stats.get("wiredTiger", {}).get("creationString", "")
        if isinstance(stats.get("wiredTiger"), dict)
        else ""
    )
    assert "block_compressor=zstd" in block_compressor, (
        "the collection was not created with the compressor the compose template "
        f"declares; wiredTiger creationString reports: {block_compressor[:200]!r}"
    )
    print(  # noqa: T201
        f"\nMEASURED store size: {on_disk / 1024**3:.2f} GiB of the "
        f"{DISK_BUDGET_BYTES / 1024**3:.0f} GiB budget "
        f"({documents:,} documents, block_compressor=zstd)"
    )


# ---------------------------------------------------------------------------
# The live half reaching the stored half
# ---------------------------------------------------------------------------


def test_written_setpoint_appears_in_the_archive_within_the_recorder_budget(
    archiver_world, monkeypatch
):
    """Write to the machine, then read it back out of history.

    This is the claim the recorder exists for, and nothing short of a deployed
    stack can make it: the value has to cross the control system into the VA, be
    sampled by the recorder over channel access, land in the store, and come back
    through the connector the agent reads with.
    """
    from osprey.connectors.factory import ConnectorFactory, register_builtin_connectors
    from osprey.utils import config as config_module

    # Writes are fail-closed, and the guard does NOT consult the config handed to
    # the connector: `_writes_enabled` re-reads `control_system.writes_enabled`
    # from the GLOBAL config every time (see
    # osprey.connectors.control_system.base), which resolves from CONFIG_FILE or
    # the working directory. Pytest runs from the repo root, which has no
    # config.yml, so without this the guard fails closed and the write is
    # refused — and the test then blames the recorder for a value that was never
    # written. Arming it means pointing the guard at the DEPLOYED project, whose
    # config.yml already declares the posture; the singletons are reset so the
    # new CONFIG_FILE is the one that gets loaded. `monkeypatch` restores all
    # three, and the autouse fixture in conftest clears them again after.
    monkeypatch.setenv("CONFIG_FILE", str(archiver_world.repo / "build" / "config.yml"))
    monkeypatch.setattr(config_module, "_default_config", None)
    monkeypatch.setattr(config_module, "_default_configurable", None)

    limits = json.loads(
        (archiver_world.repo / "build" / "data" / "channel_limits.json").read_text(encoding="utf-8")
    )
    setpoint = next(name for name in sorted(limits) if name.endswith(":SP"))

    async def _write_a_new_setpoint() -> tuple[float, datetime, Any]:
        register_builtin_connectors()  # idempotent; must run before create
        control_system = await ConnectorFactory.create_control_system_connector(
            archiver_world.config["control_system"]
        )
        current = await control_system.read_channel(setpoint)
        target = float(current.value) + 0.05
        written_at = datetime.now(UTC)
        result = await control_system.write_channel(setpoint, target)
        return target, written_at, result

    target, written_at, write_result = asyncio.run(_write_a_new_setpoint())
    assert getattr(write_result, "success", True), (
        f"the write to {setpoint} was refused, so nothing downstream could archive it: "
        f"{getattr(write_result, 'error', write_result)}"
    )

    connector = _connector(archiver_world)
    started = time.monotonic()
    deadline = started + RECORDER_BUDGET_SEC
    seen: list[float] = []
    while True:
        frame = asyncio.run(
            connector.get_data([setpoint], written_at - timedelta(seconds=5), datetime.now(UTC))
        )
        seen = [float(v) for v in frame["value"]] if len(frame) else []
        if any(abs(value - target) < 1e-6 for value in seen):
            print(  # noqa: T201
                f"\nMEASURED write->read: {time.monotonic() - started:.1f}s of the "
                f"{RECORDER_BUDGET_SEC:.0f}s budget ({setpoint} = {target})"
            )
            return
        assert time.monotonic() < deadline, (
            f"{setpoint} was set to {target} but no archived sample carried it within "
            f"{RECORDER_BUDGET_SEC:.0f}s; the archive held {seen[-5:]}"
        )
        time.sleep(RECORDER_POLL_SEC)


def test_a_window_before_coverage_is_reported_as_empty_not_invented(archiver_world, monkeypatch):
    """The honesty claim, stated twice: no points, and a truthful start.

    An archiver that answers a pre-archival question with synthesized values is
    the exact failure this whole feature exists to remove -- the agent cannot
    tell invented history from recorded history, and neither can the operator
    reading its conclusion.
    """
    connector = _connector(archiver_world)
    channel = str(_manifest_channels(archiver_world)[0]["address"])

    # "The oldest sample really held" is a moving target: every document carries
    # its own expireAt (sample time + retention span), and mongod's TTL monitor
    # sweeps on its own once-a-minute phase, eating one coarse bucket per pass.
    # Comparing a probe against a store read taken seconds apart is a race lost
    # whenever a sweep lands between them. So bracket each probe with a store
    # read on both sides: a truthful bound must lie inside the bracket, and when
    # no sweep intervenes the bracket collapses to the old exact comparison.
    def oldest_held() -> datetime:
        with _collection(archiver_world) as collection:
            from osprey.simulation.archiver_seed import oldest_sample

            oldest = oldest_sample(collection)
        assert oldest is not None, "the seeded store reports no oldest sample"
        # pymongo hands back naive datetimes read as UTC; the arithmetic below
        # mixes them with aware ones, so pin the zone rather than letting it
        # depend on which layer happened to attach it.
        return oldest if oldest.tzinfo else oldest.replace(tzinfo=UTC)

    oldest_before = oldest_held()
    metadata = asyncio.run(connector.get_metadata(channel))
    assert metadata.archival_start is not None, "get_metadata reported no archival_start"
    oldest_after = oldest_held()

    archival_start = metadata.archival_start
    if archival_start.tzinfo is None:
        archival_start = archival_start.replace(tzinfo=UTC)

    # Reported coverage is the oldest sample really held, not a declared window.
    assert (oldest_before - archival_start).total_seconds() < 1.0, (
        f"archival_start {archival_start} precedes every sample the store held "
        f"(oldest was {oldest_before}); the bound must be real history, not a declared window"
    )
    assert (archival_start - oldest_after).total_seconds() < 1.0, (
        f"archival_start {archival_start} postdates the oldest sample the store held "
        f"({oldest_after}); the bound must be real history, not a declared window"
    )

    before = archival_start - timedelta(days=2)
    frame = asyncio.run(connector.get_data([channel], before, archival_start - timedelta(hours=1)))
    assert len(frame) == 0, (
        f"a window entirely before coverage began returned {len(frame)} points for "
        f"{channel}; the archive must report what it does not have, not fill it in"
    )

    # -- the same emptiness, as the AGENT is told it --------------------------
    # Through the real MCP tool against the deployed store: the connector
    # reporting an empty frame and the agent being TOLD why are different
    # claims, and only the second one closes the gap this feature is about.
    from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

    monkeypatch.chdir(archiver_world.repo)
    from osprey.mcp_server.control_system.server_context import initialize_server_context

    initialize_server_context()
    from osprey.mcp_server.control_system.tools.archiver_read import archiver_read

    tool = get_tool_fn(archiver_read)
    oldest_before_tool = oldest_held()
    response = asyncio.run(
        tool(
            channels=[channel],
            start_time=before.isoformat(),
            end_time=(archival_start - timedelta(hours=1)).isoformat(),
        )
    )
    oldest_after_tool = oldest_held()
    payload = extract_response_dict(response)
    assert payload["status"] == "success"
    coverage = payload["summary"]["coverage"]
    assert coverage["verdict"] == "window_precedes_archive", coverage
    # The bound the agent is shown is the store's true oldest sample, not a
    # declared window — same claim as above, now at the tool surface.
    reported_start = datetime.fromisoformat(coverage["channels"][channel]["archive_start"])
    assert (oldest_before_tool - reported_start).total_seconds() < 1.0, (
        f"the tool reported archive_start {reported_start}, before every sample the "
        f"store held (oldest was {oldest_before_tool}); the agent-facing bound must be "
        f"real history, not a declared window"
    )
    assert (reported_start - oldest_after_tool).total_seconds() < 1.0, (
        f"the tool reported archive_start {reported_start}, after the oldest sample "
        f"the store held ({oldest_after_tool}); the agent-facing bound must be "
        f"real history, not a declared window"
    )


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def test_seam_between_seeded_and_recorded_history_is_within_the_noise_band(archiver_world):
    """Across all three fidelity partitions, the seam is attributable to noise.

    Seeded history ends at the seed anchor and recorded reality begins there. If
    the two halves were generated from different baselines -- a different epoch
    convention, a different boot value, a retuned generator -- the step at that
    instant is a discontinuity no operator could read as anything but an event
    that never happened.

    The threshold is quoted per channel from ``deviation_bound`` and its own
    default headroom, never a constant written here: a band embedded in this file
    would keep passing after a generator retune it no longer describes.
    """
    # The bound is only meaningful if both halves were generated with the same
    # noise level. The two constants are duplicated deliberately (osprey.simulation
    # must stay importable without the VA service package), so nothing structural
    # stops them drifting -- and a drift would widen this tolerance silently
    # instead of failing. Retuning either side must red THIS lane.
    from osprey.services.virtual_accelerator.ioc.engine_source import (
        DEFAULT_NOISE_LEVEL as VA_NOISE_LEVEL,
    )

    assert DEFAULT_NOISE_LEVEL == VA_NOISE_LEVEL, (
        "the simulation and VA noise levels have drifted "
        f"({DEFAULT_NOISE_LEVEL} vs {VA_NOISE_LEVEL}); the seam bound below describes "
        "the seeded half only, so it would pass against a mismatched recorded half"
    )

    connector = _connector(archiver_world)
    sampled = _sample_per_partition(archiver_world)
    # The window is anchored on the SEAM, not on the clock. Ending it at
    # `archival_end` and reaching back a fixed span looked equivalent and is
    # not: everything before the anchor is seeded and everything after it is
    # recorded, so once more than that span separates the seed from this test --
    # a cold image cache is enough, and CI has a 45-minute allowance -- the
    # window holds recorded samples only. The step assertion then passes without
    # a seam anywhere near it, which is the vacuous green this lane exists to
    # refuse.
    anchor = _seed_anchor(archiver_world)
    for partition in PARTITIONS:
        channels = sampled[partition]
        assert channels, f"no channels found in the {partition!r} partition"
        for channel in channels:
            address = str(channel["address"])
            metadata = asyncio.run(connector.get_metadata(address))
            if metadata.archival_end is None:
                pytest.fail(f"{address} ({partition}) holds no samples at all")

            archival_end = _as_utc(metadata.archival_end)
            assert archival_end >= anchor, (
                f"{address} ({partition}) holds nothing at or after the seed anchor, so "
                "this window covers seeded history only and crosses no seam"
            )
            frame = asyncio.run(
                connector.get_data([address], anchor - SEAM_LEAD_TIME, archival_end)
            )
            values = [float(v) for v in frame["value"]] if len(frame) else []
            if len(values) < 2:
                continue  # a channel with one sample in the window has no seam to cross

            step = max(abs(b - a) for a, b in zip(values, values[1:], strict=False))
            bound = deviation_bound(address)
            assert step <= 2 * bound, (
                f"{address} ({partition}) steps by {step:.6g} across the seed/record "
                f"seam, beyond the {2 * bound:.6g} the generator's own noise band allows"
            )


# ---------------------------------------------------------------------------
# Scenario apply, and the recorder's enablement
# ---------------------------------------------------------------------------


def test_applying_a_scenario_rewrites_windows_not_the_whole_archive(archiver_world):
    """Bounded growth, and full coverage — the two halves of "windows, not archive".

    Growth is one-sided: densification adds a bounded number of samples inside
    each event window, while the live TTL sweeper may retire aged tail samples in
    the same interval, so a small net shrink is correct. The failure guarded here
    multiplies the count several-fold, which is nowhere near either edge.

    ``uncovered`` is the free half of the assertion: it counts grid points the
    rewrite wanted but the store does not hold. On a healthy fresh deploy that is
    zero, and anything else means the archive has holes where the scenario
    expected history -- the recorder-outage class, which no count ratio would
    catch on its own.
    """
    from osprey.simulation.apply import apply_scenarios

    with _collection(archiver_world) as collection:
        before = collection.count_documents({})

    result = apply_scenarios(archiver_world.repo, ["vacuum-burst"], seed_logbook=False)

    with _collection(archiver_world) as collection:
        after = collection.count_documents({})

    assert result.archiver is not None, "apply reported no archive rewrite at all"
    assert result.archiver.skipped is None, (
        f"the archive rewrite was skipped: {result.archiver.skipped}"
    )
    # `uncovered` counts grid points the rewrite wanted but the store does not
    # hold. It is deliberately NOT asserted to be zero, and tightening it back to
    # zero would reintroduce a flake that only shows up on the wrong clock:
    #
    # `vacuum-burst` fires at a fixed time of day, so its window recurs once per
    # retained day. The archive ends at the seed anchor — the moment `osprey up`
    # ran — so whenever the most recent occurrence falls AFTER that anchor, part
    # of its window lies past the end of coverage and cannot be densified. The
    # seeder refusing to manufacture coverage there is the honest behavior this
    # whole feature exists for. `== 0` would therefore be green only when the
    # lane happens to run after that time of day, and CI runs at arbitrary hours.
    #
    # So the bound is "at most one occurrence's worth may be missing", derived
    # from this run's own numbers and the retention knob rather than a constant:
    # a daily event over N retained days contributes roughly (total / N) samples
    # per occurrence. That tolerates exactly the boundary-straddling occurrence
    # above, while a recorder outage or a hole mid-archive leaves many
    # occurrences uncovered and trips it by a wide margin.
    intended = result.archiver.inserted + result.archiver.uncovered
    allowed = UNCOVERED_FRACTION * intended
    assert result.archiver.uncovered <= allowed, (
        f"{result.archiver.uncovered} of {intended} grid points the scenario needed are "
        f"missing from the store — past the {allowed:.0f} a single boundary-straddling "
        "occurrence can account for. That is a coverage hole, not the archive's leading edge."
    )
    assert after <= before * APPLY_GROWTH_RATIO, (
        f"applying one scenario grew the archive from {before:,} to {after:,} documents "
        f"({after / before:.2f}x), past the {APPLY_GROWTH_RATIO}x a windowed rewrite "
        "should ever need -- this is the whole-archive densification signature"
    )
    print(  # noqa: T201
        f"\nMEASURED apply growth: {before:,} -> {after:,} documents "
        f"({after / before:.3f}x of the {APPLY_GROWTH_RATIO}x bound); "
        f"{result.archiver.uncovered} of {intended:,} grid points uncovered "
        f"(bound {allowed:.0f})"
    )


def test_health_reports_the_archive_fresh_while_the_recorder_writes(archiver_world):
    """``osprey health`` is where an operator asks whether history is still real.

    Everything above proves the archive from the inside — the collection, the
    connector, the recorder's own logs. This proves the one surface an operator
    actually looks at, through the console script they would type, against the
    check nobody declared by hand: the ``va_archiver:`` block named a canary
    channel, and the build derived the category, the probe type and a staleness
    threshold from the recorder's cadence.

    A reachable store and a written store are different claims, and this is the
    check that tells them apart — so it earns its place on a deployed stack in a
    way no mocked connector could.
    """
    from tests.e2e import _orm_stack

    limits = json.loads(
        (archiver_world.repo / "build" / "data" / "channel_limits.json").read_text(encoding="utf-8")
    )
    assert FRESHNESS_CANARY in limits, (
        f"{FRESHNESS_CANARY} is not a channel this machine model serves, so the derived "
        f"freshness check would report 'no samples' for a reason that has nothing to do "
        f"with the recorder. Pick a canary from the model and update FRESHNESS_CANARY."
    )

    store = archiver_world.store
    env = {
        **os.environ,
        "CONFIG_FILE": str(archiver_world.repo / "build" / "config.yml"),
        store["password_env"]: store["password"],
    }
    result = subprocess.run(
        [
            str(_orm_stack.find_osprey_console_script()),
            "health",
            "--category",
            "archiver",
            "--json",
        ],
        cwd=str(archiver_world.repo),
        capture_output=True,
        text=True,
        timeout=HEALTH_TIMEOUT_SEC,
        env=env,
    )
    assert result.stdout.strip(), (
        f"`osprey health --category archiver --json` printed nothing "
        f"(rc={result.returncode}):\n--- stderr ---\n{result.stderr}"
    )
    report = json.loads(result.stdout)

    rows = [
        row
        for row in _health_rows(report)
        if row.get("name") == "archiver_freshness" or row.get("category") == "archiver_freshness"
    ]
    assert rows, (
        "the derived archiver_freshness check did not run. The build should have written "
        "health.categories.archiver.checks from the va_archiver block's freshness_channel:\n"
        f"{json.dumps(report, indent=2)[:2000]}"
    )
    (row,) = rows
    assert row.get("status") == "ok", (
        f"the archive is deployed and the recorder is running, so the newest sample for "
        f"{FRESHNESS_CANARY} should be inside the derived threshold. Got "
        f"{row.get('status')!r}: {row.get('message')!r}"
    )


def _health_rows(report: Any) -> list[dict[str, Any]]:
    """Every check row in a ``--json`` health report, whatever it nests them under.

    Written tolerantly on purpose: this lane is about the archive, and pinning
    the report's exact envelope here would make an unrelated reshaping of the
    health JSON fail as an archiver regression.
    """
    if isinstance(report, list):
        rows = report
    else:
        rows = report.get("checks") or report.get("results") or []
        if not rows:
            for category in (report.get("categories") or {}).values():
                if isinstance(category, dict):
                    rows.extend(category.get("checks") or [])
                elif isinstance(category, list):
                    rows.extend(category)
    return [row for row in rows if isinstance(row, dict)]


def test_recorder_idles_on_mock_control_system_and_resumes_after_the_flip(archiver_world):
    """The recorder records a virtual accelerator, and nothing else.

    Recording whatever the control system happens to be would put synthesized
    mock values into a store the agent reads as machine history. The enablement
    is re-read on an interval rather than at startup precisely so this flip needs
    no redeploy -- which is what this test exercises, by flipping the mounted
    config and waiting rather than restarting anything.
    """
    from ruamel.yaml import YAML

    config_path = archiver_world.repo / "build" / "config.yml"
    yaml = YAML()
    yaml.preserve_quotes = True

    def _set_control_system(kind: str) -> None:
        with open(config_path) as handle:
            config = yaml.load(handle)
        config["control_system"]["type"] = kind
        with open(config_path, "w") as handle:
            yaml.dump(config, handle)

    poll_sec = int((archiver_world.config.get("va_archiver") or {}).get("recorder_poll_sec", 30))
    # The window each claim is measured over, once the recorder has SAID which
    # state it is in. Several sample cadences wide, so a single write the
    # recorder should not have made still shows up.
    settle = poll_sec * 2 + 10
    # How long the announcement itself may take. Not `poll_sec`: propagation is
    # bounded by several polls, not one -- see `_await_recorder_transition`.
    announce_timeout = poll_sec * 4 + 30

    idle_baseline = _recorder_transitions(_RECORDER_IDLE_RE)
    recording_baseline = _recorder_transitions(_RECORDER_RECORDING_RE)

    _set_control_system("mock")
    try:
        noticed = _await_recorder_transition(
            _RECORDER_IDLE_RE, idle_baseline, announce_timeout, "idle"
        )
        # Counted by SAMPLE INSTANT, not as a change in the collection total.
        # The store's TTL monitor is live here, so the total moves on its own:
        # a difference of two totals mixes the recorder's writes with the
        # sweeper's deletions, and the sweeper is the larger term. That cuts
        # both ways -- a recorder wrongly writing during idle is masked whenever
        # the sweep is bigger, and a recorder correctly resuming is masked the
        # same way, which is exactly what made the resume half flaky. Every
        # instant after `watch_from` is one nothing else could have written, so
        # this asks the precise question: did the recorder write, at all?
        watch_from = datetime.now(UTC)
        time.sleep(settle)
        with _collection(archiver_world) as collection:
            while_idle = collection.count_documents({"date": {"$gt": watch_from}})
        assert while_idle == 0, (
            f"{while_idle} samples were archived in the {settle:.0f}s after the recorder "
            "announced it was idling on control_system.type='mock'; it must idle rather "
            "than archive synthesized mock values"
        )
        idle_report = (
            f"idle announced {noticed:.0f}s after the config flip; "
            f"{while_idle} samples archived in the {settle:.0f}s that followed"
        )
    finally:
        _set_control_system("virtual_accelerator")

    resumed = _await_recorder_transition(
        _RECORDER_RECORDING_RE, recording_baseline, announce_timeout, "recording"
    )
    watch_from = datetime.now(UTC)
    time.sleep(settle)
    with _collection(archiver_world) as collection:
        after_resume = collection.count_documents({"date": {"$gt": watch_from}})
    assert after_resume > 0, (
        "the recorder announced it was recording again after control_system.type returned "
        f"to 'virtual_accelerator', but archived no sample in the {settle:.0f}s that followed"
    )
    print(  # noqa: T201
        f"\nMEASURED recorder flip: {idle_report}; recording announced {resumed:.0f}s "
        f"after the flip back, {after_resume} samples archived in the {settle:.0f}s after that"
    )
