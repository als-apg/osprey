"""What a whole ``osprey build`` renders for the live stand-in, and without it.

Every unit behind the stand-in has its own module — the derived overrides
(``test_live_standin_overrides.py``), the service injection
(``test_inject_va_gateways.py``), the compose instance loop
(``tests/deployment/test_va_compose_instances.py``), the recorder's choice of
machine (``tests/deployment/test_recorder_standin_compose.py``), the profile
refusals (``test_live_standin_validate.py``). What none of them can show is
that the pieces still agree once a real build has run end to end: the injector
writes ``services.live_standin`` and the override writer points the
``connector.live_standin`` gateways at a port, and nothing but a full render
proves those two are the SAME port, and that the recorder's Channel Access
address is that port too. A deployment where any pair of those disagrees still
builds; it just sends the operator somewhere they were not told about.

The stand-in is a **third control target** here, not a rewrite of the first, and
that is the sharpest thing a whole-build test can pin. ``live`` is the machine
the facility authored under ``epics:``; the build adds a
``control_system.connector.live_standin`` block beside it and touches nothing
else in ``control_system``. So the module builds the exemplar for real, twice —
once with ``virtual_accelerator.live_standin`` set and once with the key removed
— and reads the finished artifacts back the way the things that consume them do:
parsed YAML for the values, raw text for the claims that are about comments and
ordering, and the ``containers`` health category for the row ``osprey health``
grows.

**The off-state build is the anchor.** The promise a stand-in makes to every
deployment that does not want one is that it costs them nothing, and the way to
state that is not a text grep over ``build/``: the config template documents the
profile key in its own prose, the staged ``.j2`` sources are copied into
``build/services/`` verbatim, and a checkout whose path happens to contain the
words matches too. All three are hits that mean nothing. What means something
is the parsed shape — no ``live_standin`` service, no ``live-standin``
container, no ``connector.live_standin`` block, the facility's own gateways — so
that is what is asserted, artifact by artifact.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from ruamel.yaml import YAML

import osprey.health.core.containers as containers_mod
from osprey.cli.build_cmd import build as build_command
from osprey.health.core.containers import containers
from osprey.health.models import CheckResult, Status
from osprey.port_layout import default_port
from osprey.services.virtual_accelerator.manifest.standin_defaults import (
    STANDIN_BPM_ERRORS_DEFAULT,
)
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

#: Every test here renders a deployment for real — seconds each, not
#: milliseconds — which is the property that makes them worth having.
pytestmark = pytest.mark.slow

#: A build with no venv and no lifecycle hooks. Neither is what the stand-in is
#: about, and a real dependency install would dominate this module's runtime.
CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

#: The port the stand-in is reserved on throughout the codebase.
STANDIN_PORT = 5074

#: The exemplar's own virtual accelerator, and therefore the port the stand-in
#: must NOT land on.
VA_PORT = 5064

#: Where the preset's bluesky bridge lands. Looked up rather than spelled: the
#: two collision tests below need A PORT SOME OTHER SERVICE ALREADY HOLDS, and
#: which port that is belongs to the layout — ``tests/test_port_layout.py``
#: pins the number itself. The preset names none, so the bridge takes this slot.
BLUESKY_PORT = default_port("bluesky")

#: What the Control Assistant template's VA block proves, and so what the
#: ``standin`` target must prove too.
VA_PROBE_CHANNEL = "SR:VAC:GAUGE:SR01:PRESSURE:RB"

#: The ``epics`` block the template ships — the ``live`` target's, which a
#: build must render untouched whether or not a stand-in was asked for. The
#: gateways and probe channel ship commented out (authoring them is the
#: go-live edit), so untouched means the timeout and nothing else.
SHIPPED_EPICS_BLOCK = {"timeout": 5.0}

#: The opening of the note ``osprey build`` used to write above an
#: acknowledgment it derived for the operator. Nothing derives one now — the
#: stand-in is not ``live``, so ``live`` has nothing to acknowledge that the
#: operator did not say themselves.
ACK_NOTE_OPENING = "# Written by `osprey build` for the live stand-in: the `epics` gateways"

#: The commented example the template ships for the acknowledgment, and the two
#: commented gateway-port examples beside it. They are the instructions for
#: pointing a deployment at a real machine by hand, and no build consumes them.
COMMENTED_ACK_EXAMPLE = "# live_gateway_acknowledged:"
COMMENTED_PORT_EXAMPLE = "# port: 10091"

#: The template's own end-of-line comment on the strict-limits key. It explains
#: the KEY rather than the shipped value, so it stays true for a deployment that
#: sets either — nothing rewrites it at build time.
LIMITS_COMMENT = "false refuses any channel the database does not list"

#: Where a profile names the machine every session starts on, and the baseline a
#: deployment with no stand-in falls back to.
BASELINE_TYPE_KEY = "control_system.type"
VIRTUAL_ACCELERATOR = "virtual_accelerator"

_ruamel = YAML(typ="rt")


# ── Building the exemplar, with the key and without it ───────────────────────


def _set_live_standin(repo: Path, port: int | None) -> None:
    """Set or clear ``virtual_accelerator.live_standin`` in the repo's profile.

    Written through ruamel rather than by string surgery so this keeps working
    whichever way the shipped preset spells the block once it carries a
    stand-in by default — and so the *cleared* case stays a real off-state test
    rather than a no-op the day the preset starts shipping the key.
    """
    profile_path = repo / "profile.yml"
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = _ruamel.load(handle)
    block = profile["virtual_accelerator"]
    if port is None:
        block.pop("live_standin", None)
    else:
        block["live_standin"] = port
    with profile_path.open("w", encoding="utf-8") as handle:
        _ruamel.dump(profile, handle)


def _set_config_entries(repo: Path, entries: dict[str, Any]) -> None:
    """Add dotted entries to the repo profile's own ``config:`` block."""
    profile_path = repo / "profile.yml"
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = _ruamel.load(handle)
    for key, value in entries.items():
        profile["config"][key] = value
    with profile_path.open("w", encoding="utf-8") as handle:
        _ruamel.dump(profile, handle)


def _drop_profile_blocks(repo: Path, names: tuple[str, ...]) -> None:
    """Remove whole top-level blocks from the repo's profile."""
    profile_path = repo / "profile.yml"
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = _ruamel.load(handle)
    for name in names:
        profile.pop(name, None)
    with profile_path.open("w", encoding="utf-8") as handle:
        _ruamel.dump(profile, handle)


def _invoke_build(repo: Path):
    """Run ``osprey build`` the way an operator standing in the repo would."""
    previous = Path.cwd()
    os.chdir(repo)
    try:
        return CliRunner().invoke(build_command, CI_FLAGS)
    finally:
        os.chdir(previous)


def _build(repo: Path) -> Path:
    result = _invoke_build(repo)
    assert result.exit_code == 0, (
        f"build failed (exit={result.exit_code})\n{result.output}\n{result.exception}"
    )
    return repo / "build"


def _exemplar(
    dest: Path,
    *,
    standin: int | None,
    config: dict[str, Any] | None = None,
    drop_blocks: tuple[str, ...] = (),
) -> Path:
    """A seeded exemplar repo with the stand-in key set (or removed) as asked.

    Taking the stand-in away moves the baseline with it. The exemplar starts
    every session on the stand-in (``control_system.type: live_standin``), and
    ``standin_baseline_errors`` refuses that baseline on a deployment that
    stands no stand-in up — so a profile with the key removed and the baseline
    left behind is not an off-state deployment, it is an incoherent one, and
    would test the refusal rather than the absence. The baseline therefore goes
    back to the sandbox VA, which is the deployment an operator who never asked
    for a stand-in actually has. A caller naming ``control_system.type`` itself
    is left alone.
    """
    dest.mkdir(parents=True, exist_ok=True)
    repo = build_exemplar_repo(dest / EXEMPLAR_DIRNAME, seed_env=True)
    _set_live_standin(repo, standin)
    if drop_blocks:
        _drop_profile_blocks(repo, drop_blocks)
    entries = dict(config or {})
    if standin is None:
        entries.setdefault(BASELINE_TYPE_KEY, VIRTUAL_ACCELERATOR)
    if entries:
        _set_config_entries(repo, entries)
    return repo


@pytest.fixture(scope="module")
def standin_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``build/`` of an exemplar deployment that stands a live stand-in up."""
    repo = _exemplar(tmp_path_factory.mktemp("standin"), standin=STANDIN_PORT)
    return _build(repo)


@pytest.fixture(scope="module")
def plain_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``build/`` of the same deployment with the stand-in key removed."""
    repo = _exemplar(tmp_path_factory.mktemp("plain"), standin=None)
    return _build(repo)


# ── Reading the artifacts back ───────────────────────────────────────────────


def _config(build: Path) -> dict[str, Any]:
    return yaml.safe_load((build / "config.yml").read_text(encoding="utf-8"))


def _config_text(build: Path) -> str:
    return (build / "config.yml").read_text(encoding="utf-8")


def _compose(build: Path, service: str) -> dict[str, Any]:
    path = build / "services" / service / "docker-compose.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rendered_composes(build: Path) -> dict[str, dict[str, Any]]:
    """Every rendered service compose file, keyed by its service directory."""
    return {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((build / "services").glob("*/docker-compose.yml"))
    }


def _rendered_configs(build: Path) -> dict[str, dict[str, Any]]:
    """The deployment's own rendered config, plus one per persona project."""
    configs = {"<deployment>": _config(build)}
    for path in sorted(build.glob(f"{EXEMPLAR_DIRNAME}-*/config.yml")):
        configs[path.parent.name] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return configs


# ── FR-10: what the rendered config says about the third target ──────────────


class TestTheRenderedConfigDescribesTheStandIn:
    """``build/config.yml`` after a build that asked for a stand-in."""

    def test_live_standin_render_declares_a_second_instance_of_one_service(
        self, standin_build
    ) -> None:
        """One template directory, two service blocks, deployed in order.

        The shared ``path`` is the whole shape: the stand-in is a second
        INSTANCE of the virtual accelerator's compose template, not a second
        service, so nothing is staged under ``services/live_standin/`` and the
        two blocks differ only in the port they serve.
        """
        config = _config(standin_build)

        assert config["services"]["live_standin"] == {
            "path": "./services/virtual_accelerator",
            "port": STANDIN_PORT,
        }
        assert (
            config["services"]["live_standin"]["path"]
            == (config["services"]["virtual_accelerator"]["path"])
        )
        deployed = config["deployed_services"]
        assert deployed.count("live_standin") == 1
        assert deployed.index("live_standin") == deployed.index("virtual_accelerator") + 1
        assert not (standin_build / "services" / "live_standin").exists()

    def test_live_standin_render_dials_the_stand_in_from_both_gateway_lanes(
        self, standin_build
    ) -> None:
        """Loopback, the profile's port, name-server transport, both roles.

        Both roles and not just ``read_only``: a write-enabled session moves to
        ``write_access``, and a stand-in that only the readonly lane reaches
        would send the write that matters to whatever the template shipped.
        """
        config = _config(standin_build)
        standin = config["control_system"]["connector"]["live_standin"]

        assert standin["gateways"] == {
            "read_only": {
                "address": "localhost",
                "port": STANDIN_PORT,
                "use_name_server": True,
            },
            "write_access": {
                "address": "localhost",
                "port": STANDIN_PORT,
                "use_name_server": True,
            },
        }
        # The same port the service block names — the agreement no unit test of
        # either writer can make on its own.
        assert (
            standin["gateways"]["read_only"]["port"] == config["services"]["live_standin"]["port"]
        )

    def test_live_standin_render_carries_the_probe_channel_across(self, standin_build) -> None:
        """A target with no probe channel is never switched to, so it carries one."""
        connectors = _config(standin_build)["control_system"]["connector"]

        assert connectors["live_standin"]["probe_channel"] == VA_PROBE_CHANNEL
        assert (
            connectors["live_standin"]["probe_channel"]
            == connectors["virtual_accelerator"]["probe_channel"]
        )

    def test_live_standin_render_leaves_the_live_targets_own_block_alone(
        self, standin_build
    ) -> None:
        """``live`` is the facility's machine on a stand-in deployment too.

        The reason the stand-in is a third target rather than a rewrite: the
        gateways under ``epics:`` read exactly as the facility authored them,
        and the build adds no probe channel there either. This is what lets a
        deployment already pointed at hardware rehearse beside it.
        """
        epics = _config(standin_build)["control_system"]["connector"]["epics"]

        assert epics == SHIPPED_EPICS_BLOCK
        assert "probe_channel" not in epics

    def test_live_standin_render_takes_its_limits_posture_from_the_profile(
        self, standin_build
    ) -> None:
        """How strictly the deployment runs is stated by the profile, not derived.

        The exemplar preset authors the strict pair itself, and that is where
        the rendered values come from — ``test_live_standin_overrides.py`` makes
        the other half of the claim, that a profile which drops the pair gets
        the template's value back rather than a derived one.

        What is pinned here is the rendered *line*: while the stand-in was
        ``live``, the build flipped this key and then had to rewrite the comment
        beside it to stop the line contradicting itself. Nothing rewrites it
        now, so the shipped comment has to explain the key rather than the value
        the template happened to ship — and it must still read true beside the
        ``false`` the profile asked for.
        """
        text = _config_text(standin_build)
        limits = _config(standin_build)["control_system"]["limits_checking"]

        assert limits["enabled"] is True
        assert limits["allow_unlisted_channels"] is False
        line = next(row for row in text.splitlines() if "allow_unlisted_channels" in row)
        assert "allow_unlisted_channels: false" in line
        assert LIMITS_COMMENT in line

    def test_live_standin_render_derives_no_operator_acknowledgment(self, standin_build) -> None:
        """``live`` has nothing to acknowledge that the operator did not say.

        The acknowledgment is a claim about the real machine, and while the
        build pointed ``live`` at the stand-in it had to make that claim on the
        operator's behalf. ``live`` is the facility's own block again, so the
        key goes back to being the operator's — the commented example the
        template ships stays standing, and no value is written above it.
        """
        config = _config(standin_build)
        text = _config_text(standin_build)

        target_switch = config["control_system"].get("target_switch") or {}
        assert "live_gateway_acknowledged" not in target_switch
        assert "    live_gateway_acknowledged:" not in text
        assert COMMENTED_ACK_EXAMPLE in text
        assert ACK_NOTE_OPENING not in text


# ── SC-2: what a stand-in costs an epics deployment ──────────────────────────


def test_live_standin_render_adds_only_the_standin_block_to_an_epics_deployment(
    tmp_path: Path,
) -> None:
    """A facility pointed at its own machine can stand a rehearsal up beside it.

    The whole feature in one assertion, and the one no unit can make: build the
    same ``type: epics`` deployment twice, once with the stand-in key and once
    without, and the two rendered configs differ by the stand-in's own three
    additions — its connector block, its service registration, and its
    ``deployed_services`` entry — and by nothing else.

    Stated in both directions on purpose. The parsed comparison says no VALUE
    moved; the text diff says no LINE was taken away, which is the half that
    catches a build consuming a comment the facility was meant to keep reading.

    ``va_archiver:`` comes off both profiles because the exemplar ships one and
    ``standin_archive_errors`` refuses that block beside a stand-in on an
    ``epics`` baseline: the archive belongs to the machine it records, and a
    deployment whose real machine is live cannot have its recorder follow the
    stand-in. That refusal has its own test; what this one is about is the
    ``epics`` block, so the archive is taken out of the picture on BOTH sides —
    keeping the two profiles identical apart from the one key under test.
    """
    epics_baseline = {"control_system.type": "epics"}
    standin_repo = _exemplar(
        tmp_path / "with",
        standin=STANDIN_PORT,
        config=epics_baseline,
        drop_blocks=("va_archiver",),
    )
    plain_repo = _exemplar(
        tmp_path / "without",
        standin=None,
        config=epics_baseline,
        drop_blocks=("va_archiver",),
    )
    with_standin = _build(standin_repo)
    without = _build(plain_repo)

    rendered = _config(with_standin)
    baseline = _config(without)

    # Two repos means two checkout paths, and the render states its own. That
    # is the fixture's difference, not the stand-in's, so it comes out of both
    # sides here and out of the text below.
    assert rendered.pop("project_root") == str(standin_repo)
    assert baseline.pop("project_root") == str(plain_repo)

    standin_block = rendered["control_system"]["connector"].pop("live_standin")
    assert set(standin_block) == {"gateways", "probe_channel"}
    assert standin_block["gateways"]["read_only"]["port"] == STANDIN_PORT
    assert rendered["services"].pop("live_standin")["port"] == STANDIN_PORT
    rendered["deployed_services"].remove("live_standin")

    assert rendered == baseline

    removed = [
        line
        for line in difflib.unified_diff(
            _config_text(without).replace(str(plain_repo), "<repo>").splitlines(),
            _config_text(with_standin).replace(str(standin_repo), "<repo>").splitlines(),
            lineterm="",
            n=0,
        )
        if line.startswith("-") and not line.startswith("---")
    ]
    assert removed == [], "the build took lines away from the facility's own render"


# ── The compose files the operator ends up with ──────────────────────────────


class TestTheRenderedComposeStandsTwoMachinesUp:
    """``build/services/*/docker-compose.yml`` after the same build."""

    def test_live_standin_render_describes_a_second_container(self, standin_build) -> None:
        """Two soft-IOCs out of one template, each named for what it serves."""
        compose = _compose(standin_build, "virtual_accelerator")

        assert list(compose["services"]) == ["virtual-accelerator", "live-standin"]
        standin = compose["services"]["live-standin"]
        assert standin["ports"] == [f"127.0.0.1:{STANDIN_PORT}:{STANDIN_PORT}/tcp"]
        assert standin["environment"]["EPICS_CA_SERVER_PORT"] == str(STANDIN_PORT)
        assert compose["services"]["virtual-accelerator"]["environment"][
            "EPICS_CA_SERVER_PORT"
        ] == str(VA_PORT)
        # One image, built once: the two instances can never disagree about
        # their Channel Access stack.
        assert standin["image"] == compose["services"]["virtual-accelerator"]["image"]

    def test_live_standin_render_ships_the_perturbation_as_an_overridable_default(
        self, standin_build
    ) -> None:
        """The stand-in reads differently, from its own variable, by default.

        An instance that perturbs nothing reads identically to the machine
        beside it, and telling the two apart is the whole point — so the
        default is baked into the render rather than left to the operator's
        ``.env``, and it arrives under a variable of its own so setting a fault
        on one machine cannot set it on both.

        Substituted on UNSET (``${VAR-default}``) and not on empty
        (``${VAR:-default}``): an operator who writes ``VA_STANDIN_BPM_ERRORS=``
        is asking for a stand-in that reads clean, and the colon form would hand
        them the perturbation back.
        """
        services = _compose(standin_build, "virtual_accelerator")["services"]

        assert services["live-standin"]["environment"]["VA_BPM_ERRORS"] == (
            f"${{VA_STANDIN_BPM_ERRORS-{STANDIN_BPM_ERRORS_DEFAULT}}}"
        )
        assert services["virtual-accelerator"]["environment"]["VA_BPM_ERRORS"] == (
            "${VA_BPM_ERRORS:-}"
        )

    def test_live_standin_render_records_the_stand_in_not_the_sandbox(self, standin_build) -> None:
        """The archive belongs to the machine, so the recorder follows the stand-in.

        Both wiring sites are read back separately because they have to name
        the same instance: a recorder that waits on one machine and reads from
        another is a deploy-time race that looks like an empty archive.
        """
        recorder = _compose(standin_build, "archiver_recorder")["services"]["archiver-recorder"]

        assert recorder["environment"]["EPICS_CA_NAME_SERVERS"] == f"live-standin:{STANDIN_PORT}"
        assert recorder["depends_on"]["live-standin"] == {"condition": "service_healthy"}
        assert "virtual-accelerator" not in recorder["depends_on"]


# ── FR-7: the row ``osprey health`` grows ────────────────────────────────────


class _FakeProc:
    """Minimal subprocess stand-in for the runtime's ``--version`` call."""

    def __init__(self, stdout: bytes) -> None:
        self._stdout = stdout
        self.returncode: int | None = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, b""


async def _probe(spec, ctx):  # noqa: ANN001, ANN202 - test double
    return CheckResult(spec["name"], spec["category"], Status.OK, f"{spec['container']}: running")


async def test_live_standin_render_grows_a_container_health_row(
    standin_build, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``osprey health`` watches the stand-in because the build deployed it.

    The category derives one ``container_<service>`` row per entry in
    ``deployed_services``, so this is really an assertion about the config the
    build wrote: a stand-in that never joined that list would run unwatched,
    and the deployment's own health report would say nothing about a machine it
    stands up. Fed the REAL rendered config rather than a hand-built one, since
    the list is exactly what the build is being asked about.
    """
    monkeypatch.setattr(containers_mod, "get_runtime_command", lambda *_a, **_k: ["docker"])

    async def _fake_exec(*_a: object, **_k: object) -> _FakeProc:
        return _FakeProc(b"Docker version 27.0.0")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    rows = await containers(_config(standin_build), probe=_probe)()

    names = [row.name for row in rows]
    assert "container_live_standin" in names
    assert names.count("container_live_standin") == 1
    assert names.index("container_live_standin") == names.index("container_virtual_accelerator") + 1


# ── FR-1: a build that did not ask for one ───────────────────────────────────


class TestABuildWithoutTheKeyIsUntouched:
    """The stand-in costs a deployment that does not want one nothing."""

    def test_live_standin_render_off_leaves_no_stand_in_in_any_rendered_config(
        self, plain_build
    ) -> None:
        """Not the deployment's config, and not any persona's either.

        Every rendered config is checked rather than only the deployment's,
        because the stand-in's port is projected into attached renders ungated
        — which is right when there IS one, and would be a phantom machine in
        a persona's roster when there is not.
        """
        for name, config in _rendered_configs(plain_build).items():
            assert "live_standin" not in config.get("services", {}), name
            assert "live_standin" not in (config.get("deployed_services") or []), name
            connector = config["control_system"].get("connector") or {}
            assert "live_standin" not in connector, name

    def test_live_standin_render_off_leaves_no_stand_in_container(self, plain_build) -> None:
        """No second instance, and nothing waiting on or reading from one."""
        for service, compose in _rendered_composes(plain_build).items():
            for name, block in (compose.get("services") or {}).items():
                assert name != "live-standin", service
                assert "live-standin" not in (block.get("depends_on") or {}), f"{service}/{name}"
                environment = block.get("environment") or {}
                if isinstance(environment, dict):
                    assert "VA_STANDIN_BPM_ERRORS" not in " ".join(
                        str(value) for value in environment.values()
                    ), f"{service}/{name}"

        va = _compose(plain_build, "virtual_accelerator")
        assert list(va["services"]) == ["virtual-accelerator"]
        recorder = _compose(plain_build, "archiver_recorder")["services"]["archiver-recorder"]
        assert recorder["environment"]["EPICS_CA_NAME_SERVERS"] == f"virtual-accelerator:{VA_PORT}"

    def test_live_standin_render_off_keeps_the_facilitys_own_epics_block(self, plain_build) -> None:
        """The shipped block, untouched — no gateways invented, no probe channel copied in."""
        control_system = _config(plain_build)["control_system"]

        assert control_system["connector"]["epics"] == SHIPPED_EPICS_BLOCK
        assert "probe_channel" not in control_system["connector"]["epics"]
        # The profile's own strict pair, unchanged by the stand-in going away:
        # the limits posture was never the stand-in's to decide either way.
        assert control_system["limits_checking"]["allow_unlisted_channels"] is False

    def test_live_standin_render_off_keeps_the_templates_own_examples(self, plain_build) -> None:
        """The commented examples are the instructions for going to the machine.

        A build that derives nothing must leave them standing — the
        acknowledgment example, the two commented gateway ports, and the
        end-of-line comment on the strict-limits key, which explains the key
        rather than the value and so is true either way.
        """
        text = _config_text(plain_build)

        assert COMMENTED_ACK_EXAMPLE in text
        assert "    live_gateway_acknowledged:" not in text
        assert text.count(COMMENTED_PORT_EXAMPLE) == 2
        line = next(row for row in text.splitlines() if "allow_unlisted_channels" in row)
        assert LIMITS_COMMENT in line

    def test_live_standin_render_off_is_stable_across_a_rebuild(self, tmp_path: Path) -> None:
        """Rebuilding the same repo rewrites the same bytes.

        Scoped to the artifacts the stand-in would have changed — the rendered
        configs and the two compose files it touches — rather than the whole
        tree, because two other things there move for reasons of their own and
        pinning them here would pin someone else's invariant:
        ``.osprey-manifest.json`` carries a wall-clock timestamp, and the
        bluesky-web sidecar's roster secrets are resolved from the personas'
        rendered configs, which a FIRST build has not written yet.
        """
        repo = _exemplar(tmp_path, standin=None)
        build = _build(repo)

        def snapshot() -> dict[str, str]:
            files = {
                f"config:{name}": path
                for name, path in {"<deployment>": build / "config.yml"}.items()
            }
            for path in sorted(build.glob(f"{EXEMPLAR_DIRNAME}-*/config.yml")):
                files[f"config:{path.parent.name}"] = path
            for service in ("virtual_accelerator", "archiver_recorder"):
                files[f"compose:{service}"] = build / "services" / service / "docker-compose.yml"
            return {key: path.read_text(encoding="utf-8") for key, path in files.items()}

        first = snapshot()
        _build(repo)
        assert snapshot() == first


# ── The refusals, reached through a real build ───────────────────────────────


class TestTheBuildRefusesAnIncoherentStandIn:
    """Faults that only a profile can state, refused where an operator sees them."""

    def _refuse(self, repo: Path, caplog: pytest.LogCaptureFixture) -> str:
        with caplog.at_level(logging.ERROR):
            result = _invoke_build(repo)
        assert result.exit_code != 0, result.output
        return caplog.text

    def test_live_standin_render_refuses_a_derived_key_spelled_in_the_profile(
        self, tmp_path: Path, caplog
    ) -> None:
        """One fact, two homes, free to disagree — named with the way out.

        Pinned on a stand-in gateway leaf, which is now the only kind of key the
        build reserves: the refusal sends an author who wants to address a
        machine to the ``epics`` block, because that block is theirs and the
        stand-in's is not.
        """
        repo = _exemplar(
            tmp_path,
            standin=STANDIN_PORT,
            config={"control_system.connector.live_standin.gateways.read_only.port": 5064},
        )

        text = self._refuse(repo, caplog)

        assert "The stand-in owns that key" in text
        assert "control_system.connector.live_standin.gateways.read_only.port" in text
        assert "control_system.connector.epics" in text
        # Refused before anything is published, so the previous build stands.
        assert not (repo / "build" / "config.yml").exists()

    def test_live_standin_render_refuses_a_port_another_service_claims(
        self, tmp_path: Path, caplog
    ) -> None:
        """The clash is named by the dotted key its author would edit.

        ``bluesky.port`` is a port the exemplar really claims, so this is the
        collision an operator actually meets rather than a staged one.
        """
        repo = _exemplar(tmp_path, standin=BLUESKY_PORT)

        text = self._refuse(repo, caplog)

        assert (
            f"virtual_accelerator.live_standin ({BLUESKY_PORT}) collides with "
            f"bluesky.port ({BLUESKY_PORT})" in text
        )
        assert not (repo / "build" / "config.yml").exists()
