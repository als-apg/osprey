"""What a whole ``osprey build`` renders for the live stand-in, and without it.

Every unit behind the stand-in has its own module — the derived overrides
(``test_live_standin_overrides.py``), the service injection
(``test_inject_va_gateways.py``), the compose instance loop
(``tests/deployment/test_va_compose_instances.py``), the recorder's choice of
machine (``tests/deployment/test_recorder_standin_compose.py``), the profile
refusals (``test_live_standin_validate.py``). What none of them can show is
that the pieces still agree once a real build has run end to end: the injector
writes ``services.live_standin`` and the override writer points the ``epics``
gateways at a port, and nothing but a full render proves those two are the SAME
port, that the recorder's Channel Access address is that port too, and that the
acknowledgment the target switch reads names the endpoint a session would
actually dial. A deployment where any pair of those disagrees still builds; it
just sends the operator somewhere they were not told about.

So this module builds the exemplar deployment for real, twice — once with
``virtual_accelerator.live_standin`` set and once with the key removed — and
reads the finished artifacts back the way the things that consume them do:
parsed YAML for the values, raw text for the claims that are about comments and
ordering (a comment is what an operator reads when judging whether a rendered
line means what it says), and the ``containers`` health category for the row
``osprey health`` grows.

**The off-state build is the anchor.** The promise a stand-in makes to every
deployment that does not want one is that it costs them nothing, and the way to
state that is not a text grep over ``build/``: the config template documents the
profile key in its own prose, the staged ``.j2`` sources are copied into
``build/services/`` verbatim, and a checkout whose path happens to contain the
words matches too. All three are hits that mean nothing. What means something
is the parsed shape — no ``live_standin`` service, no ``live-standin``
container, no ``live_gateway_acknowledged`` value, the facility's own gateways —
so that is what is asserted, artifact by artifact.
"""

from __future__ import annotations

import asyncio
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
from osprey.cli.build_profile_standin import STRICT_LIMITS_COMMENT
from osprey.health.core.containers import containers
from osprey.health.models import CheckResult, Status
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

#: What the Control Assistant template's VA block proves, and so what the live
#: target must prove too once it is the stand-in.
VA_PROBE_CHANNEL = "SR:VAC:GAUGE:SR01:PRESSURE:RB"

#: The shipped ALS production ``epics`` gateways — what a build with no
#: stand-in must still render, untouched.
SHIPPED_EPICS_GATEWAYS = {
    "read_only": {
        "address": "cagw-alsdmz.als.lbl.gov",
        "port": 5064,
        "use_name_server": False,
    },
    "write_access": {
        "address": "cagw-alsdmz.als.lbl.gov",
        "port": 5084,
        "use_name_server": False,
    },
}

#: The opening of the note ``osprey build`` writes above the acknowledgment it
#: derived. Its six lines are pinned byte-for-byte where they are produced
#: (``test_inject_va_gateways.py``); what matters here is that the render puts
#: the note immediately above the key, so the value is never read without it.
ACK_NOTE_OPENING = "# Written by `osprey build` for the live stand-in: the `epics` gateways"

#: The commented example the template ships for the acknowledgment, and the two
#: commented gateway-port examples beside it. A build that derives nothing must
#: leave all three standing — they are the instructions for going live by hand.
COMMENTED_ACK_EXAMPLE = "# live_gateway_acknowledged:"
COMMENTED_PORT_EXAMPLE = "# port: 5094"

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


def _exemplar(dest: Path, *, standin: int | None, config: dict[str, Any] | None = None) -> Path:
    """A seeded exemplar repo with the stand-in key set (or removed) as asked."""
    repo = build_exemplar_repo(dest / EXEMPLAR_DIRNAME, seed_env=True)
    _set_live_standin(repo, standin)
    if config:
        _set_config_entries(repo, config)
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


def _line_no(text: str, needle: str) -> int:
    for index, line in enumerate(text.splitlines()):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} not found in:\n{text}")


# ── FR-9: what the rendered config says about the live target ────────────────


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

    def test_live_standin_render_dials_the_stand_in_from_both_epics_gateways(
        self, standin_build
    ) -> None:
        """Loopback, the profile's port, name-server transport, both roles.

        Both roles and not just ``read_only``: a write-enabled session moves to
        ``write_access``, and a stand-in that only the readonly lane reaches
        would send the write that matters to whatever the template shipped.
        """
        epics = _config(standin_build)["control_system"]["connector"]["epics"]

        assert epics["gateways"] == {
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
            epics["gateways"]["read_only"]["port"]
            == (_config(standin_build)["services"]["live_standin"]["port"])
        )

    def test_live_standin_render_carries_the_probe_channel_across(self, standin_build) -> None:
        """A target with no probe channel is never switched to, so it carries one."""
        connectors = _config(standin_build)["control_system"]["connector"]

        assert connectors["epics"]["probe_channel"] == VA_PROBE_CHANNEL
        assert (
            connectors["epics"]["probe_channel"]
            == connectors["virtual_accelerator"]["probe_channel"]
        )

    def test_live_standin_render_leaves_the_sandbox_gateway_rows_portless(
        self, standin_build
    ) -> None:
        """The VA's own rows are default-filled from its service port.

        Written out they would state the same fact twice, and the second copy
        is the one that goes stale when the service port moves.
        """
        va_gateways = _config(standin_build)["control_system"]["connector"]["virtual_accelerator"][
            "gateways"
        ]

        assert va_gateways, "the sandbox VA still has gateway rows"
        for role, row in va_gateways.items():
            assert "port" not in row, f"{role}: {row}"

    def test_live_standin_render_runs_strict_limits_and_says_why(self, standin_build) -> None:
        """The value is strict and the comment beside it explains the strictness.

        The template ships this key permissive with an inline comment calling it
        a tutorial convenience. A rendered line that flipped the value and kept
        the comment is the sentence an operator reads when deciding whether the
        deployment is safe, so the comment is part of the assertion.
        """
        limits = _config(standin_build)["control_system"]["limits_checking"]
        assert limits["enabled"] is True
        assert limits["allow_unlisted_channels"] is False

        text = _config_text(standin_build)
        line = next(row for row in text.splitlines() if "allow_unlisted_channels" in row)
        assert "allow_unlisted_channels: false" in line
        assert STRICT_LIMITS_COMMENT in line
        assert "Permissive" not in line

    def test_live_standin_render_acknowledges_the_gateway_once_and_explains_it(
        self, standin_build
    ) -> None:
        """The acknowledgment names the endpoint, is written once, and is annotated.

        Without the acknowledgment the deployment's own ``live`` target refuses
        itself, so the build derives it — and having derived it, it says so:
        the note above the value is how an operator going live knows the value
        is not theirs and what to replace it with. The commented example the
        template ships must be gone, because two spellings of one key in one
        file is the reader's problem, not the writer's.
        """
        config = _config(standin_build)
        text = _config_text(standin_build)

        assert config["control_system"]["target_switch"]["live_gateway_acknowledged"] == (
            f"localhost:{STANDIN_PORT}"
        )
        # Once as a key. The literal also appears in the template's own prose
        # above `target_switch:`, which is documentation and stays.
        assert text.count("    live_gateway_acknowledged:") == 1
        assert COMMENTED_ACK_EXAMPLE not in text
        assert _line_no(text, ACK_NOTE_OPENING) < _line_no(text, "    live_gateway_acknowledged:")

    def test_live_standin_render_keeps_the_probe_interval_comment_whole(
        self, standin_build
    ) -> None:
        """The note is inserted beside a key whose own comment wraps two lines.

        ``probe_interval_s`` is the key ruamel parks the acknowledgment's
        comment token on, so a writer that replaced that token instead of
        editing it would truncate this key's explanation to its first line.
        Both lines are asserted, in order, above the note.
        """
        text = _config_text(standin_build)

        first = _line_no(text, "probe_interval_s: 30")
        second = _line_no(text, "# every target's gateways")
        assert "# Seconds between background reachability probes of" in text.splitlines()[first]
        assert second == first + 1
        assert second < _line_no(text, ACK_NOTE_OPENING)


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
        """
        services = _compose(standin_build, "virtual_accelerator")["services"]

        assert services["live-standin"]["environment"]["VA_BPM_ERRORS"] == (
            f"${{VA_STANDIN_BPM_ERRORS:-{STANDIN_BPM_ERRORS_DEFAULT}}}"
        )
        assert services["virtual-accelerator"]["environment"]["VA_BPM_ERRORS"] == (
            "${VA_BPM_ERRORS:-}"
        )

    def test_live_standin_render_records_the_stand_in_not_the_sandbox(self, standin_build) -> None:
        """The archive belongs to the machine, so the recorder follows ``live``.

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
    and the deployment's own health report would say nothing about the machine
    it calls ``live``. Fed the REAL rendered config rather than a hand-built
    one, since the list is exactly what the build is being asked about.
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
            target_switch = config["control_system"].get("target_switch") or {}
            assert "live_gateway_acknowledged" not in target_switch, name

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
        """The shipped production gateways, and no probe channel copied in."""
        control_system = _config(plain_build)["control_system"]

        assert control_system["connector"]["epics"]["gateways"] == SHIPPED_EPICS_GATEWAYS
        assert "probe_channel" not in control_system["connector"]["epics"]
        assert control_system["limits_checking"]["allow_unlisted_channels"] is True

    def test_live_standin_render_off_keeps_the_templates_own_examples(self, plain_build) -> None:
        """The commented examples are the instructions for going live by hand.

        A build that derives nothing must leave them standing — the
        acknowledgment example, the two commented gateway ports, and the
        tutorial comment that is still true because the value is still
        permissive.
        """
        text = _config_text(plain_build)

        assert COMMENTED_ACK_EXAMPLE in text
        assert "    live_gateway_acknowledged:" not in text
        assert text.count(COMMENTED_PORT_EXAMPLE) == 2
        assert ACK_NOTE_OPENING not in text
        line = next(row for row in text.splitlines() if "allow_unlisted_channels" in row)
        assert "Permissive mode for tutorial" in line
        assert STRICT_LIMITS_COMMENT not in line

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

        The refusal has to name the go-live steps rather than just the
        clash, because an author who wrote that key wanted the live machine and
        the answer is not "delete this line": it is to stop asking for a
        stand-in first. Pinned on a ``limits_checking`` leaf, since the
        strict-posture keys are derived for the same reason the gateways are
        and are the ones an author is likeliest to reach for.
        """
        repo = _exemplar(
            tmp_path,
            standin=STANDIN_PORT,
            config={"control_system.limits_checking.allow_unlisted_channels": True},
        )

        text = self._refuse(repo, caplog)

        assert "Going live is three steps: delete `virtual_accelerator.live_standin`" in text
        assert "control_system.limits_checking.allow_unlisted_channels" in text
        # Refused before anything is published, so the previous build stands.
        assert not (repo / "build" / "config.yml").exists()

    def test_live_standin_render_refuses_an_epics_baseline(self, tmp_path: Path, caplog) -> None:
        """A deployment already pointed at hardware has nothing to stand in for."""
        repo = _exemplar(tmp_path, standin=STANDIN_PORT, config={"control_system.type": "epics"})

        text = self._refuse(repo, caplog)

        assert "control_system.type: epics with virtual_accelerator.live_standin" in text
        assert "Going live is three steps: delete `virtual_accelerator.live_standin`" in text
        assert not (repo / "build" / "config.yml").exists()

    def test_live_standin_render_refuses_a_port_another_service_claims(
        self, tmp_path: Path, caplog
    ) -> None:
        """The clash is named by the dotted key its author would edit.

        ``bluesky.port`` is a port the exemplar really claims, so this is the
        collision an operator actually meets rather than a staged one.
        """
        repo = _exemplar(tmp_path, standin=8090)

        text = self._refuse(repo, caplog)

        assert "virtual_accelerator.live_standin (8090) collides with bluesky.port (8090)" in text
        assert not (repo / "build" / "config.yml").exists()

    def test_live_standin_render_reports_every_profile_fault_in_one_list(
        self, tmp_path: Path, caplog
    ) -> None:
        """Two faults, one refusal: an author fixes both and builds once.

        Both of these are profile faults, so both are accumulated by
        ``BuildProfile.validate`` and raised together. The ``config:``-duplicate
        refusal above is deliberately NOT part of this claim: it is raised
        later, by the render, and validation runs first — so a profile carrying
        a duplicate AND a profile fault is told about the profile fault, and
        meets the duplicate on its next attempt.
        """
        repo = _exemplar(tmp_path, standin=8090, config={"control_system.type": "epics"})

        text = self._refuse(repo, caplog)

        assert "collides with bluesky.port (8090)" in text
        assert "control_system.type: epics with virtual_accelerator.live_standin" in text
