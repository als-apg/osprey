"""``osprey up`` — starting a deployment repo from ``build/`` as it was built.

Three properties, and everything here serves one of them.

**It never re-renders the services stack.** ``up`` reads ``build/config.yml``
and the compose files a build left beside it. The tests assert that by the
absence of ``prepare_compose_files`` on every path: a start verb that re-rendered
them would make "what is running" unanswerable from the filesystem. The
web-terminal stack is the deliberate exception, and it covers three files: its
compose file, nginx config and landing page are re-rendered at every start
because they follow the user roster rather than the build. Persona projects are
not among them — ``osprey build`` renders one per delta and a start only checks
they are there — so a build wiping ``build/`` is what puts a changed delta into
force.

**It refuses rather than guessing.** A profile edited since the build, a profile
that no longer resolves, a repo with no build at all, a ``.env`` that does not
exist — each is a refusal with a named remedy, and each names only remedies that
are real. ``--build`` and ``--as-built`` are the two documented ways through, and
which of them applies is asserted per state, because they are not
interchangeable: ``--as-built`` cannot start a build that was never made.

**It never prints a secret where a secret should not go.** The mint notice for a
web-terminal password goes to a terminal or nowhere at all.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

import osprey.cli.deploy_cmd as deploy_cmd
from osprey.cli.deploy_cmd import up_verb
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import REPO_ID_LABEL, repo_identity
from osprey.deployment.web_terminals import provision
from osprey.utils.workspace import container_image_context
from tests.cli._lifecycle_build import stub_build
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo


@pytest.fixture
def lifecycle_repo(tmp_path: Path) -> Path:
    """The gold-standard three-zone deployment repo, with no ``.env`` in it.

    Built here rather than imported from the shared fixture module: this is the
    only module under ``tests/deployment/`` that wants it, and re-exporting a
    fixture into a module whose tests then take it by name shadows the import.
    """
    return build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME)


# A rendered config with one deployed service, which is all the compose-file
# lookup and the token mint need. `build_dir` is spelled exactly as the real
# render spells it: relative to the repo root, which is both `project_root` and
# the compose project directory.
_RENDERED_CONFIG = """\
project_name: als-exemplar
build_dir: ./build
deployed_services:
  - event_dispatcher
services:
  event_dispatcher:
    path: services/event_dispatcher
claude_code:
  provider: anthropic
"""

_TOP_COMPOSE = "networks:\n  osprey-network:\n"


def _service_compose(bind_address: str = "127.0.0.1") -> str:
    """One service publishing one port, as a real render spells it.

    The bind address is baked in at render time, which is why reachability is a
    property of the build and no start-time flag can change it.
    """
    return (
        "services:\n"
        "  event-dispatcher:\n"
        "    image: example:local\n"
        "    ports:\n"
        f'      - "{bind_address}:18020:8020/tcp"\n'
    )


def _stamp_key_digests(repo: Path, build: Path) -> None:
    """Add the per-key digests a real build stamps beside the fingerprint.

    They are what lets a refusal say *which* part of the profile moved, and a
    build that stamped none produces the honest but vaguer "this build recorded
    no detail about which". Both shapes are real, so the tests that assert on
    named keys have to build the shape that has them.
    """
    from osprey.deployment.staleness import KEY_DIGESTS_MANIFEST_KEY, profile_fingerprint

    fingerprint = profile_fingerprint(repo / "profile.yml")
    assert fingerprint is not None
    manifest_path = build / ".osprey-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["creation"][KEY_DIGESTS_MANIFEST_KEY] = fingerprint.key_digests
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def render_build(
    repo: Path,
    *,
    stamped_hash: str | None = "current",
    version: str | None = None,
    config: str = _RENDERED_CONFIG,
    bind_address: str = "127.0.0.1",
    with_compose: bool = True,
) -> Path:
    """Put a build in *repo* that ``up`` can start.

    ``stub_build`` writes the config and the provenance manifest the drift check
    reads; this adds the compose files, in the layout the pinned invocation
    contract requires — output beside its templates under ``build/services/``,
    which is what ``build_dir: ./build`` names when read from the repo root.
    """
    build = stub_build(repo, stamped_hash=stamped_hash, version=version, config=config)
    if stamped_hash == "current":
        _stamp_key_digests(repo, build)
    if with_compose:
        services = build / "services"
        (services / "event_dispatcher").mkdir(parents=True, exist_ok=True)
        (services / "docker-compose.yml").write_text(_TOP_COMPOSE, encoding="utf-8")
        (services / "event_dispatcher" / "docker-compose.yml").write_text(
            _service_compose(bind_address), encoding="utf-8"
        )
    return build


@pytest.fixture
def started(monkeypatch):
    """Stub everything that would touch a container runtime; record what ran.

    Returns a dict that stays empty until a start actually reaches ``compose``,
    so ``not started`` is the assertion for "nothing was started" — the claim
    every refusal message in this verb makes.
    """
    record: dict = {}

    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    # The host-port preflight probes real listeners on this machine; a unit test
    # must not depend on which ports happen to be free here. Exposure is read
    # from the rendered bindings by a different function, which stays live.
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)

    def _fake_run(cmd, env=None, check=False):
        record.setdefault("cmds", []).append(list(cmd))
        record["env"] = dict(env or {})

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    def _fake_image_build(config, dev_mode, env, build_context=None):
        record["build_context"] = build_context

    monkeypatch.setattr(container_lifecycle, "_build_project_image", _fake_image_build)

    # A start must never re-render. Anything reaching for the renderer is a
    # failure of the property this module exists to pin, not a missing stub.
    def _forbidden(*args, **kwargs):
        raise AssertionError("osprey up must not render compose files")

    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", _forbidden)
    return record


def run_up(repo: Path, *args: str, **kwargs) -> object:
    """Invoke ``osprey up`` against *repo* from outside it."""
    return CliRunner().invoke(up_verb, ["--repo", str(repo), *args], **kwargs)


def up_argv(started: dict) -> list[str]:
    """The final ``compose up`` argv, from the commands the start ran."""
    for cmd in reversed(started.get("cmds", [])):
        if "up" in cmd:
            return cmd
    raise AssertionError(f"no `compose up` in {started.get('cmds')}")


# ---------------------------------------------------------------------------
# It starts what the build rendered, and renders nothing
# ---------------------------------------------------------------------------


def test_up_starts_the_compose_files_the_build_left(lifecycle_repo, started):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output
    argv = up_argv(started)
    # The pinned invocation contract: one project directory, repo-anchored -f
    # paths, the repo's own .env.
    assert argv[:2] == ["docker", "compose"]
    assert "--project-directory" in argv
    assert argv[argv.index("--project-directory") + 1] == str(lifecycle_repo)
    files = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-f"]
    assert files == [
        str(lifecycle_repo / "build" / "services" / "docker-compose.yml"),
        str(lifecycle_repo / "build" / "services" / "event_dispatcher" / "docker-compose.yml"),
    ]
    assert argv[argv.index("--env-file") + 1] == str(lifecycle_repo / ".env")


def test_up_is_correct_from_a_subdirectory(lifecycle_repo, started, monkeypatch):
    """The repo, not the working directory, decides what is started."""
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    monkeypatch.chdir(lifecycle_repo / "data")

    result = CliRunner().invoke(up_verb, ["-d"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert up_argv(started)[up_argv(started).index("--project-directory") + 1] == str(
        lifecycle_repo
    )
    # And it put the working directory back where it found it.
    assert Path.cwd() == lifecycle_repo / "data"


def test_the_project_image_is_built_from_the_render_not_the_source(lifecycle_repo, started):
    """A start builds the image from build output, never from the source zone.

    Specifically from the container copy of that output — the repo the build
    renders under ``build/.image/`` against the ``/app/<project>`` path a
    container sees, rather than this host's. Building from ``build/`` itself
    would still be build output, but it is the HOST's: the image would ship a
    ``project_root`` and an ``OSPREY_CONFIG`` naming the machine that built it.
    """
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    assert run_up(lifecycle_repo, "-d").exit_code == 0
    context = started["build_context"]
    assert context == container_image_context(lifecycle_repo, lifecycle_repo.name)
    assert lifecycle_repo / "build" in context.parents, (
        "the image context must be build output, not the source zone"
    )


def test_a_start_exports_no_repo_identity_variable(lifecycle_repo, started):
    """The identity is BAKED into the render, so nothing needs it in the env.

    An interpolated ``${OSPREY_REPO_ID}`` label would be silently empty for
    anyone who runs ``docker compose`` by hand without the variable exported —
    and the two verbs that read this label back (``down``'s label fallback,
    ``reset``'s cross-checkout refusal) decide what to DESTROY from it. A
    speculative export beside the baked value would be a second source for one
    fact, so there is none.
    """
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    assert run_up(lifecycle_repo, "-d").exit_code == 0
    assert "OSPREY_REPO_ID" not in started["env"]


# ---------------------------------------------------------------------------
# The host process resolves the same config the containers are handed
# ---------------------------------------------------------------------------


def test_host_side_work_during_a_start_reads_the_render_not_the_working_directory(
    lifecycle_repo, started, monkeypatch
):
    """A start does real work in-process, and it must read ``build/config.yml``.

    Compose hands every service ``CONFIG_FILE`` naming the render, so a container
    resolves the facility zone correctly. The host process that stages and seeds
    the archiver store before that compose runs is the one participant in the
    same contract that used to be left out: it chdirs to the repo ROOT, where the
    three-zone layout puts only ``profile.yml``, so the global resolver found no
    ``config.yml`` at all and every synthesis call degraded to UTC.

    Asserted on the resolved zone rather than on the variable, because UTC is
    what a facility that configures nothing gets anyway: the bug is only visible
    as a WRONG ANSWER when the render names a real zone.
    """
    from osprey.utils.config import get_facility_timezone

    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(
        lifecycle_repo, config=_RENDERED_CONFIG + "system:\n  timezone: America/Los_Angeles\n"
    )

    seen: dict = {}

    def _probe(config, dev_mode, env, build_context=None):
        seen["tz"] = str(get_facility_timezone())

    monkeypatch.setattr(container_lifecycle, "_build_project_image", _probe)

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output
    assert seen["tz"] == "America/Los_Angeles"


def test_a_start_leaves_no_config_anchor_behind_in_the_environment(
    lifecycle_repo, started, monkeypatch
):
    """The anchor is scoped to the start, like the working directory is.

    ``CONFIG_FILE`` is a generic enough name that leaving one deployment's render
    exported would redirect the next OSPREY process this shell runs — including a
    verb pointed at a DIFFERENT repo with ``--repo``.
    """
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    assert run_up(lifecycle_repo, "-d").exit_code == 0
    assert "CONFIG_FILE" not in os.environ


def test_a_missing_build_is_a_refusal_naming_the_build_verb(lifecycle_repo, started, caplog):
    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 1
    assert "No build found" in caplog.text
    assert "osprey build" in caplog.text
    assert not started


def test_a_build_without_its_compose_files_is_not_silently_started(lifecycle_repo, started, caplog):
    """A manifest and a config alone are not a deployment.

    Starting an empty ``-f`` list would bring nothing up and report success —
    the failure mode a start verb that reads its inputs has to guard, since
    there is no render step left to notice the gap.
    """
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, with_compose=False)

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 1
    assert "no compose files were rendered" in caplog.text
    assert "osprey build" in caplog.text
    assert not started


# ---------------------------------------------------------------------------
# The dev-flavor gate
# ---------------------------------------------------------------------------


def _framework_service_compose(dev: bool) -> str:
    """A service whose image build installs the framework, in either flavor.

    ``OSPREY_VERSION`` in ``build.args`` is what marks a render as
    framework-installing; ``osprey build --dev`` additionally emits
    ``OSPREY_DEV: "1"``. Spelled here exactly as the service templates render
    them.
    """
    dev_arg = '        OSPREY_DEV: "1"\n' if dev else ""
    return (
        "services:\n"
        "  event-dispatcher:\n"
        "    image: example:local\n"
        "    build:\n"
        "      context: ./build/services/event_dispatcher\n"
        "      args:\n"
        '        OSPREY_VERSION: "2026.6.2"\n'
        f"{dev_arg}"
        "    ports:\n"
        '      - "127.0.0.1:18020:8020/tcp"\n'
    )


def test_dev_refuses_a_pinned_render_and_names_the_dev_build(lifecycle_repo, started, monkeypatch):
    """``up --dev`` on a non-dev render cannot be honored, and says how to.

    ``up`` never re-renders, so the sidecar contexts hold no wheel and no
    ``OSPREY_DEV`` arg — starting would bring up containers running the pinned
    release under a flag that means "run my local code".
    """
    import osprey.deployment.wheel_build as wheel_build

    monkeypatch.setattr(wheel_build, "preflight_dev_mode", lambda: None)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    build = render_build(lifecycle_repo)
    (build / "services" / "event_dispatcher" / "docker-compose.yml").write_text(
        _framework_service_compose(dev=False), encoding="utf-8"
    )

    with pytest.raises(container_lifecycle.DevModeUnavailableError) as excinfo:
        container_lifecycle.up_as_built(lifecycle_repo, detached=True, dev_mode=True)

    assert "rendered without --dev" in excinfo.value.reason
    assert "event_dispatcher" in excinfo.value.reason
    assert "osprey build --dev" in excinfo.value.remedy
    assert "osprey up --build --dev" in excinfo.value.remedy
    assert not started


def test_a_dev_render_started_plain_warns_it_bakes_the_local_checkout(
    lifecycle_repo, started, caplog
):
    """Plain ``up`` starts a dev render — what was built — but says what it is."""
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    build = render_build(lifecycle_repo)
    (build / "services" / "event_dispatcher" / "docker-compose.yml").write_text(
        _framework_service_compose(dev=True), encoding="utf-8"
    )

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output
    assert started
    assert "dev render" in caplog.text
    assert "local osprey checkout" in caplog.text


# ---------------------------------------------------------------------------
# The drift gate
# ---------------------------------------------------------------------------


def test_drift_refuses_and_names_what_changed(lifecycle_repo, started, caplog):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    profile = lifecycle_repo / "profile.yml"
    profile.write_text(profile.read_text(encoding="utf-8") + "\nmodel: opus\n", encoding="utf-8")

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 1
    assert "profile.yml or a file it points at has changed" in caplog.text
    assert "model" in caplog.text
    # Both exits, spelled as the operator would type them.
    assert "osprey up --build" in caplog.text
    assert "osprey up --as-built" in caplog.text
    assert "Nothing was started" in caplog.text
    assert not started


def test_as_built_starts_the_drifted_build_and_says_so(lifecycle_repo, started, caplog):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    profile = lifecycle_repo / "profile.yml"
    profile.write_text(profile.read_text(encoding="utf-8") + "\nmodel: opus\n", encoding="utf-8")

    result = run_up(lifecycle_repo, "-d", "--as-built")

    assert result.exit_code == 0, result.output
    assert started
    assert "as it was rendered" in caplog.text
    assert "will not match profile.yml" in caplog.text


def test_build_chains_the_render_then_starts(lifecycle_repo, started, monkeypatch):
    """``--build`` re-renders through the same verb an operator would type."""
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, stamped_hash="stale")
    chained: list[Path] = []

    def _fake_chain(ctx, repo_root, *, dev=False):
        chained.append(repo_root)
        # A real build would leave a matching fingerprint behind; the gate has
        # already been passed by then, so only the render's effect matters here.
        render_build(repo_root)

    monkeypatch.setattr(deploy_cmd, "_chain_build", _fake_chain)

    result = run_up(lifecycle_repo, "-d", "--build")

    assert result.exit_code == 0, result.output
    assert chained == [lifecycle_repo]
    assert started


def test_build_and_as_built_are_refused_together(lifecycle_repo, started):
    render_build(lifecycle_repo)

    result = run_up(lifecycle_repo, "-d", "--build", "--as-built")

    assert result.exit_code == 2
    assert "at most one" in result.output
    assert not started


def test_as_built_is_not_an_escape_from_having_no_build(lifecycle_repo, started, caplog):
    """The one refusal ``--as-built`` does not answer, and it says so."""
    result = run_up(lifecycle_repo, "-d", "--as-built")

    assert result.exit_code == 1
    assert "--as-built starts a build that already exists" in caplog.text
    assert not started


def test_build_is_an_escape_from_having_no_build(lifecycle_repo, started, monkeypatch):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    monkeypatch.setattr(
        deploy_cmd, "_chain_build", lambda ctx, repo, *, dev=False: render_build(repo)
    )

    assert run_up(lifecycle_repo, "-d", "--build").exit_code == 0
    assert started


def test_an_unverifiable_build_refuses_with_the_as_built_escape(lifecycle_repo, started, caplog):
    """A build carrying no fingerprint cannot be vouched for either way."""
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, stamped_hash=None)

    refused = run_up(lifecycle_repo, "-d")
    assert refused.exit_code == 1
    assert "cannot verify profile ↔ build consistency" in caplog.text
    assert "osprey up --as-built" in caplog.text
    assert not started

    assert run_up(lifecycle_repo, "-d", "--as-built").exit_code == 0
    assert started


def test_as_built_on_an_unverifiable_build_claims_no_mismatch(lifecycle_repo, started, caplog):
    """A failed comparison is not a finding.

    On DRIFT the mismatch is established and the warning says so. Here nothing
    is known either way — the build may match the profile exactly — so a warning
    borrowed from the DRIFT case would be inventing a result out of a comparison
    that could not be made.
    """
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, stamped_hash=None)

    assert run_up(lifecycle_repo, "-d", "--as-built").exit_code == 0
    assert "stays unknown" in caplog.text
    assert "will not match profile.yml" not in caplog.text


def test_the_env_remedy_only_says_copy_when_there_is_something_to_copy(
    lifecycle_repo, started, monkeypatch, caplog
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (lifecycle_repo / ".env.example").unlink()
    render_build(lifecycle_repo)

    assert run_up(lifecycle_repo, "-d").exit_code == 1
    assert "cp .env.example .env" not in caplog.text
    assert "Create" in caplog.text and ".env" in caplog.text
    assert not started


def test_an_unparseable_profile_refuses_rather_than_reading_as_clean(
    lifecycle_repo, started, caplog
):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    (lifecycle_repo / "profile.yml").write_text("{ not: [valid", encoding="utf-8")

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 1
    assert "cannot verify profile ↔ build consistency" in caplog.text
    assert not started


def test_version_skew_warns_and_starts(lifecycle_repo, started, caplog):
    """A framework upgrade must not stand between an operator and their stack."""
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, version="1900.1.1")

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output
    assert started
    assert "rendered by osprey 1900.1.1" in caplog.text


# ---------------------------------------------------------------------------
# The .env preflight
# ---------------------------------------------------------------------------


def test_a_missing_env_refuses_and_names_the_example(lifecycle_repo, started, monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    render_build(lifecycle_repo)

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 1
    assert "No .env" in caplog.text
    assert "cp .env.example .env" in caplog.text
    assert "ANTHROPIC_API_KEY" in caplog.text  # what to put in it
    assert not (lifecycle_repo / ".env").exists()
    assert not started


def test_a_missing_env_is_not_seeded_without_a_terminal(
    lifecycle_repo, started, monkeypatch, caplog
):
    """A non-interactive run gets the refusal, never a silent harvest.

    The exported key is right there; taking it unasked would copy a secret into
    a file the operator did not ask to create, on a run nobody is watching.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-shell")
    monkeypatch.setattr(deploy_cmd, "_stdin_is_a_terminal", lambda: False)
    render_build(lifecycle_repo)

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 1
    assert not (lifecycle_repo / ".env").exists()
    assert "sk-from-the-shell" not in result.output
    assert "sk-from-the-shell" not in caplog.text
    assert not started


def test_an_accepted_offer_seeds_env_at_0600_and_starts(lifecycle_repo, started, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-shell")
    monkeypatch.setattr(deploy_cmd, "_stdin_is_a_terminal", lambda: True)
    render_build(lifecycle_repo)

    result = run_up(lifecycle_repo, "-d", input="y\n")

    assert result.exit_code == 0, result.output
    env_path = lifecycle_repo / ".env"
    assert "ANTHROPIC_API_KEY=sk-from-the-shell" in env_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    # The offer names the variable, never its value.
    assert "ANTHROPIC_API_KEY" in result.output
    assert "sk-from-the-shell" not in result.output
    assert started


def test_a_declined_offer_refuses_and_writes_nothing(lifecycle_repo, started, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-shell")
    monkeypatch.setattr(deploy_cmd, "_stdin_is_a_terminal", lambda: True)
    render_build(lifecycle_repo)

    result = run_up(lifecycle_repo, "-d", input="n\n")

    assert result.exit_code == 1
    assert not (lifecycle_repo / ".env").exists()
    assert not started


def test_an_existing_env_is_never_prompted_about(lifecycle_repo, started, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-shell")
    monkeypatch.setattr(deploy_cmd, "_stdin_is_a_terminal", lambda: True)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=already-here\n", encoding="utf-8")
    render_build(lifecycle_repo)

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output
    assert "Seed one from your shell" not in result.output
    assert (lifecycle_repo / ".env").read_text(encoding="utf-8").startswith("ANTHROPIC_API_KEY=alr")


# ---------------------------------------------------------------------------
# Secrets stay in the one .env
# ---------------------------------------------------------------------------


def test_minted_tokens_land_in_the_repo_env_and_are_copied_nowhere(
    lifecycle_repo, started, monkeypatch
):
    """One repo, one secret store.

    A profile write-back used to keep a project ``.env`` and a profile ``.env``
    in step. Here they are one file in one directory, so a copy would be a
    second thing to maintain rather than a backup.

    Asserted structurally — the repo holds exactly one ``.env`` when the deploy
    is done — rather than by watching a copier not fire. The property is that no
    second copy exists, and a check that names the copier could only ever pin
    the absence of *that* copier.
    """
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    assert run_up(lifecycle_repo, "-d").exit_code == 0

    from osprey.utils.dotenv import parse_dotenv_file

    env = parse_dotenv_file(lifecycle_repo / ".env")
    assert env["EVENT_DISPATCHER_TOKEN"]
    assert env["DISPATCH_WORKER_TOKEN"]

    # `.env.example` is documentation and carries no values; `.env.auth` /
    # `.env.production` are the web stack's own, written only when it deploys.
    copies = [
        path
        for path in lifecycle_repo.rglob(".env")
        if path.is_file() and path != lifecycle_repo / ".env"
    ]
    assert copies == [], f"the deploy left a second secret store: {copies}"


def test_the_mint_is_idempotent_across_starts(lifecycle_repo, started, monkeypatch):
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    from osprey.utils.dotenv import parse_dotenv_file

    assert run_up(lifecycle_repo, "-d").exit_code == 0
    first = parse_dotenv_file(lifecycle_repo / ".env")
    assert run_up(lifecycle_repo, "-d").exit_code == 0
    second = parse_dotenv_file(lifecycle_repo / ".env")

    assert first["EVENT_DISPATCHER_TOKEN"] == second["EVENT_DISPATCHER_TOKEN"]
    assert (lifecycle_repo / ".env").read_text(encoding="utf-8").count(
        "EVENT_DISPATCHER_TOKEN="
    ) == 1


# ---------------------------------------------------------------------------
# Exposure is a property of the build
# ---------------------------------------------------------------------------


def test_a_wildcard_build_is_treated_as_exposed_without_the_flag(
    lifecycle_repo, started, monkeypatch, caplog
):
    """The rendered binding decides, and it is the only thing that does.

    There is no start-time flag to arm or disarm the empty-token guard: a build
    that publishes on a wildcard address gets the fail-closed rules because of
    what it publishes.
    """
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "")
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    (lifecycle_repo / ".env").write_text(
        "ANTHROPIC_API_KEY=x\nEVENT_DISPATCHER_TOKEN=\n", encoding="utf-8"
    )
    render_build(lifecycle_repo, bind_address="0.0.0.0")

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 1
    assert "refusing to start a deployment that is reachable off-host" in caplog.text
    assert "event-dispatcher publish on 0.0.0.0" in caplog.text
    assert not started


def test_a_web_terminal_deploy_counts_as_exposed(lifecycle_repo, started, monkeypatch, caplog):
    """A published-port check alone would call a web deploy private.

    Every service in the web stack runs ``network_mode: host`` and its nginx
    binds every interface unrestricted, so the deployment is reachable off-host
    with no ``ports:`` entry anywhere in the services compose files.
    """
    import yaml

    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    (lifecycle_repo / ".env.production").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    monkeypatch.setattr(container_lifecycle, "deploy_up_web_terminals", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "preflight_web_terminals", lambda *a, **k: None)
    render_build(
        lifecycle_repo,
        config=yaml.safe_dump(
            {
                "project_name": "als-exemplar",
                "build_dir": "./build",
                "deployed_services": [],
                "claude_code": {"provider": "anthropic"},
                "modules": {"web_terminals": {"enabled": True}},
            }
        ),
        with_compose=False,
    )

    assert run_up(lifecycle_repo, "-d").exit_code == 0
    assert "web-terminal stack runs on the host network" in caplog.text


# ---------------------------------------------------------------------------
# Repo identity
# ---------------------------------------------------------------------------


def test_repo_identity_is_stable_short_and_path_derived(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()

    identity = repo_identity(one)
    assert len(identity) == 12
    assert all(character in "0123456789abcdef" for character in identity)
    assert identity == repo_identity(one)
    assert identity != repo_identity(two)


def test_repo_identity_follows_a_symlink_to_one_answer(tmp_path):
    """Two spellings of one directory are one checkout, not two."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    assert repo_identity(link) == repo_identity(real)


def test_the_identity_is_a_content_free_label_of_the_path(tmp_path):
    """It answers "which checkout", so editing the repo must not change it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    before = repo_identity(repo)
    (repo / "profile.yml").write_text("provider: anthropic\n", encoding="utf-8")

    assert repo_identity(repo) == before


def test_two_checkouts_of_one_deployment_get_different_identities(tmp_path):
    """The case the label exists for.

    ``COMPOSE_PROJECT_NAME`` comes from the repo's directory NAME, so two
    clones of one deployment on a host share a project name and a volume
    namespace. Without a per-checkout identity, a destructive verb in one
    cannot tell the other's containers from its own.
    """
    one = tmp_path / "a" / "als-exemplar"
    two = tmp_path / "b" / "als-exemplar"
    one.mkdir(parents=True)
    two.mkdir(parents=True)

    assert one.name == two.name
    assert repo_identity(one) != repo_identity(two)


def test_the_render_bakes_the_identity_into_the_label_metadata(tmp_path):
    """``_inject_project_metadata`` is where the literal comes from."""
    from osprey.deployment.compose_generator import _inject_project_metadata

    repo = tmp_path / "als-exemplar"
    repo.mkdir()
    injected = _inject_project_metadata({"project_name": "demo", "project_root": str(repo)})

    assert injected["osprey_labels"]["repo_id"] == repo_identity(repo)


def test_the_web_stack_bakes_the_same_identity_as_the_services_stack(tmp_path):
    """One deployment, one identity, whichever of its two compose files is read."""
    from osprey.deployment.web_terminals.render import render_web_terminals

    repo = tmp_path / "als-exemplar"
    repo.mkdir()
    rendered = render_web_terminals(
        {
            "project_root": str(repo),
            "facility": {"prefix": "ex"},
            "deploy": {"fqdn": "example.invalid"},
            "modules": {
                "web_terminals": {
                    "enabled": True,
                    "nginx_port": 9080,
                    "web_base_port": 8100,
                    "artifact_base_port": 8200,
                    "ariel_base_port": 8300,
                    "lattice_base_port": 8400,
                    "users": ["alice"],
                }
            },
        }
    )
    compose = rendered["docker-compose.web.yml"]

    identity = repo_identity(repo)
    assert f'{REPO_ID_LABEL}: "{identity}"' in compose
    # Baked, not interpolated: a `${VAR}` here would resolve to empty for
    # anyone invoking compose without the variable exported.
    assert "${OSPREY_REPO_ID}" not in compose


# ---------------------------------------------------------------------------
# A minted web-terminal password never reaches a retained log
# ---------------------------------------------------------------------------


def _password_roster(project_root: Path) -> dict:
    """A web-terminals stanza with password auth and one unseeded user."""
    (project_root / ".env").touch()
    # A bare-string roster entry: `normalize_users` drops an object entry that
    # carries no explicit `index`, so this is the shape a one-user roster takes.
    return {"enabled": True, "auth": {"method": "password"}, "users": ["alice"]}


def test_a_mint_prints_the_password_on_a_terminal(tmp_path, monkeypatch, capsys):
    """The show-once design is unchanged where a person is watching."""
    monkeypatch.setattr(provision, "_stdout_is_a_terminal", lambda: True)
    monkeypatch.setattr(provision, "_require_auth_sidecar_image", lambda web_terminals: None)

    provision._provision_auth_secrets(_password_roster(tmp_path), str(tmp_path))

    printed = capsys.readouterr().out
    assert "Minted a login password for web-terminal user 'alice'" in printed
    assert "shown this once" in printed


def test_a_mint_prints_no_password_when_stdout_is_not_a_terminal(
    tmp_path, monkeypatch, capsys, caplog
):
    """The referred-out finding: in CI, stdout is a retained job log.

    Nothing that could be a password may appear there. What replaces it is a
    warning that says the plaintext is gone — never that it can be read back out
    of ``.env.auth``, which holds hashes.
    """
    monkeypatch.setattr(provision, "_stdout_is_a_terminal", lambda: False)
    monkeypatch.setattr(provision, "_require_auth_sidecar_image", lambda web_terminals: None)

    provision._provision_auth_secrets(_password_roster(tmp_path), str(tmp_path))

    printed = capsys.readouterr().out
    assert "Minted a login password" not in printed
    assert "shown this once" not in printed

    # Nothing resembling the secret reached stdout OR the log: the hash is the
    # only surviving artifact, and the plaintext half of it is in neither sink.
    stored = (tmp_path / ".env.auth").read_text(encoding="utf-8")
    assert "OSPREY_AUTH_PW_HASH_ALICE=" in stored
    hashes = [
        line.split("=", 1)[1]
        for line in stored.splitlines()
        if line.startswith("OSPREY_AUTH_PW_HASH_ALICE=")
    ]
    for value in hashes:
        assert value not in printed
        assert value not in caplog.text

    assert "did NOT print it" in caplog.text
    assert "osprey users passwd" in caplog.text
    assert "OSPREY_AUTH_PW_<USER>" in caplog.text
    # The remedy must not send anyone looking for a plaintext that is gone.
    assert "read them from .env.auth" not in caplog.text


def test_a_supplied_password_is_never_echoed_either_way(tmp_path, monkeypatch, capsys):
    """A password the operator already knows is hashed in without a notice."""
    monkeypatch.setattr(provision, "_stdout_is_a_terminal", lambda: True)
    monkeypatch.setattr(provision, "_require_auth_sidecar_image", lambda web_terminals: None)
    roster = _password_roster(tmp_path)
    (tmp_path / ".env").write_text("OSPREY_AUTH_PW_ALICE=chosen-by-hand\n", encoding="utf-8")

    provision._provision_auth_secrets(roster, str(tmp_path))

    printed = capsys.readouterr().out
    assert "chosen-by-hand" not in printed
    assert "Minted" not in printed


def test_the_up_path_reaches_the_sink_aware_mint(lifecycle_repo, started, monkeypatch):
    """The whole point of the fix: a real ``osprey up`` uses the guarded sink.

    Asserted through the deploy path rather than on the helper alone, because
    the defect was never in the helper — it was in which sink the deploy handed
    it.
    """
    monkeypatch.setattr(provision, "_stdout_is_a_terminal", lambda: False)
    sinks: list = []
    real = provision.ensure_auth_credentials

    def _record(usernames, project_root, *, echo=print):
        sinks.append(echo)
        return real(usernames, project_root, echo=echo)

    monkeypatch.setattr(provision, "ensure_auth_credentials", _record)
    monkeypatch.setattr(provision, "_require_auth_sidecar_image", lambda web_terminals: None)

    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    config = json.loads(
        json.dumps(
            {
                "project_name": "als-exemplar",
                "build_dir": "./build",
                "deployed_services": [],
                "claude_code": {"provider": "anthropic"},
                "modules": {
                    "web_terminals": {
                        "enabled": True,
                        "auth": {"method": "password"},
                        "users": ["alice"],
                    }
                },
            }
        )
    )
    import yaml

    # Registry mode expects CI to have rendered this already; the mint runs
    # after that gate, and the mint is what this test is about.
    (lifecycle_repo / ".env.production").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, config=yaml.safe_dump(config), with_compose=False)
    monkeypatch.setattr(container_lifecycle, "deploy_up_web_terminals", lambda *a, **k: None)

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output
    assert sinks and all(sink is not print for sink in sinks)
    assert "Minted a login password" not in result.output


# ---------------------------------------------------------------------------
# The legacy path is untouched
# ---------------------------------------------------------------------------


def test_the_legacy_deploy_up_still_renders(tmp_path, monkeypatch):
    """``osprey up`` keeps its render-then-start shape until Phase 3 deletes it.

    The shared tail must not have changed what the old verb does, or every
    project still on it would start something other than what it started
    yesterday.
    """
    rendered: list = []

    def _prepare(config_path, dev_mode=False, expose_network=False):
        rendered.append(config_path)
        return {"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", _prepare)
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: None)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert rendered == [str(tmp_path / "config.yml")]
    # And the legacy path still writes its .env where it always did.
    assert (tmp_path / ".env").is_file()


def test_up_as_built_is_reachable_without_a_working_directory_in_the_repo(
    lifecycle_repo, started, tmp_path, monkeypatch
):
    """The deployment function takes the repo; it does not read it off the cwd."""
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    container_lifecycle.up_as_built(lifecycle_repo, detached=True)

    assert up_argv(started)[up_argv(started).index("--project-directory") + 1] == str(
        lifecycle_repo
    )
    assert Path.cwd() == elsewhere


def test_dev_mode_refuses_before_any_work_when_it_cannot_be_honored(
    lifecycle_repo, started, monkeypatch
):
    """``--dev`` that cannot stage a wheel must not start the released code."""
    from osprey.deployment.errors import DevModeUnavailableError

    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    def _refuse():
        raise DevModeUnavailableError(reason="not an editable install", remedy="pip install -e .")

    monkeypatch.setattr("osprey.deployment.wheel_build.preflight_dev_mode", _refuse)

    result = run_up(lifecycle_repo, "-d", "--dev")

    assert result.exit_code == 1
    assert "not an editable install" in result.output
    assert "Nothing was deployed" in result.output  # rendered on the console, not logged
    assert not started


def test_the_working_directory_survives_a_refusal(lifecycle_repo, started, monkeypatch):
    """A verb that chdirs must put it back however it ends.

    Refused here by the ``--dev`` preflight, which is the refusal that happens
    latest — inside ``_resolve_as_built_inputs``, after ``up`` has already
    chdir'd into the repo. An earlier refusal would leave the restore untested.
    """
    from osprey.deployment.errors import DevModeUnavailableError

    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, bind_address="127.0.0.1")

    def _refuse():
        raise DevModeUnavailableError(reason="not an editable install", remedy="pip install -e .")

    monkeypatch.setattr("osprey.deployment.wheel_build.preflight_dev_mode", _refuse)
    before = Path.cwd()

    assert run_up(lifecycle_repo, "-d", "--dev").exit_code == 1
    assert Path.cwd() == before
    assert os.getcwd() == str(before)


def test_minting_targets_the_repo_env_from_any_directory(lifecycle_repo, started, monkeypatch):
    """The mint must follow the repo, not the process's working directory.

    ``_ensure_service_tokens`` falls back to a cwd-relative ``.env``, which was
    correct only because the legacy path chdir'd into the project first. If a
    start verb ever moves or drops that chdir, an unqualified fallback writes
    the tokens into whatever directory the operator was standing in while
    ``--env-file`` still points at the repo's ``.env`` — and the stack comes up
    with its fail-closed tokens unset. So the path is passed explicitly, and
    this pins it from a directory that is deliberately not the repo.
    """
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    elsewhere = lifecycle_repo.parent / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    container_lifecycle.up_as_built(lifecycle_repo, detached=True)

    from osprey.utils.dotenv import parse_dotenv_file

    assert parse_dotenv_file(lifecycle_repo / ".env")["EVENT_DISPATCHER_TOKEN"]
    assert not (elsewhere / ".env").exists()
    # And the --env-file the invocation carries is that same file.
    argv = up_argv(started)
    assert argv[argv.index("--env-file") + 1] == str(lifecycle_repo / ".env")


# ---------------------------------------------------------------------------
# Build → discover: the two halves have to agree about where compose files go
# ---------------------------------------------------------------------------


@pytest.fixture
def built_repo(lifecycle_repo):
    """A deployment repo with a REAL ``osprey build`` behind it.

    Slower than the stub every other test here uses, and the point: the defect
    these tests exist for was an integration one. Each half was
    self-consistent — the build wrote where its own config told it to, and the
    discovery read where the config told it to — while disagreeing about what
    that config meant, so nothing short of running both could catch it.

    ``--skip-deps`` (no project venv) and ``--skip-lifecycle`` (no profile
    hooks) keep it to the render, which is the part under test.
    """
    from click.testing import CliRunner as _CliRunner

    from osprey.cli.build_cmd import build as build_cmd

    result = _CliRunner().invoke(
        build_cmd,
        ["--repo", str(lifecycle_repo), "--skip-deps", "--skip-lifecycle"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return lifecycle_repo


def test_a_real_build_lands_compose_where_up_looks_for_it(built_repo, started):
    """SC: the two halves of the contract, exercised end to end.

    The rendered files spell their own mounts against the repo root
    (``./build/services/<svc>/…``), which is the directory compose is pinned to,
    so that is where the files themselves have to be. A build that wrote them
    one level deeper produced a stack whose every bind-mount source pointed at a
    directory holding templates instead of config.
    """
    top = built_repo / "build" / "services" / "docker-compose.yml"
    assert top.is_file(), sorted(p.name for p in (built_repo / "build").iterdir())

    # Rendered output cohabits with the templates it came from, differing only
    # by filename — the layout the mount paths require.
    bluesky = built_repo / "build" / "services" / "bluesky"
    assert (bluesky / "docker-compose.yml").is_file()
    assert (bluesky / "docker-compose.yml.j2").is_file()
    assert (bluesky / "config.yml").is_file()
    assert not (built_repo / "build" / "build").exists()

    # And `up`'s discovery finds them, from the repo root, off the shipped
    # config — the half that would have gone on reading an empty directory.
    from osprey.utils.config import load_project_config

    config = load_project_config(str(built_repo / "build" / "config.yml"), wrap_errors=True)
    discovered = container_lifecycle.as_built_compose_files(config, built_repo)

    assert discovered
    assert str(top) in [str((built_repo / f).resolve()) for f in discovered]
    for compose_file in discovered:
        assert (built_repo / compose_file).is_file()


def test_a_real_build_leaves_no_runtime_state_in_the_output_zone(built_repo):
    """``rm -rf build/`` must lose nothing, so nothing durable may be rendered there.

    Three renderers used to write agent-data directories into the tree they were
    rendering — the retired ``_agent_data`` spelling, the VA scenario-state
    directory, and a Claude-artifact backup — each anchored on the render root
    instead of the repo. All three are runtime state, and the repo's own
    ``var/agent_data`` is where it belongs and stays.
    """
    build = built_repo / "build"

    assert not (build / "_agent_data").exists()
    assert not (build / "var").exists()
    # The durable location is untouched and still there.
    assert (built_repo / "var" / "agent_data").is_dir()
    assert (built_repo / "var" / "audit").is_dir()


def test_a_real_build_bakes_the_repo_identity_into_every_stack(built_repo):
    """CC-1, as rendered: a literal on EVERY container and EVERY volume.

    Read off the real render rather than the injection helper, because the
    thing that has to be true is that a verb reading these files back — ``down``
    finding this deployment's containers, ``reset`` refusing another checkout's
    — sees a value, not an unexpanded variable.

    Enumerated from the rendered documents, never filtered by what already
    carries the label. An earlier version of this test collected the files
    containing ``REPO_ID_LABEL`` and asserted things about those, which made it
    structurally incapable of noticing a service that had no label at all —
    and two of them (``bluesky-redis`` and ``tiled``, the only services whose
    templates carried no ``labels:`` block to append to) went unlabelled
    exactly that way. An unlabelled container is not cosmetic: ``down``'s
    label-driven fallback would leave a Redis and a Tiled server running while
    the operator believed the stack was down.
    """
    import yaml

    identity = repo_identity(built_repo)
    compose_files = sorted((built_repo / "build" / "services").rglob("docker-compose.yml"))
    assert compose_files

    unlabelled_services: list[str] = []
    unlabelled_volumes: list[str] = []
    seen_services = 0
    seen_volumes = 0

    for compose_file in compose_files:
        text = compose_file.read_text(encoding="utf-8")
        assert "${OSPREY_REPO_ID}" not in text, f"{compose_file} interpolates rather than bakes"
        document = yaml.safe_load(text) or {}
        where = compose_file.relative_to(built_repo)

        for name, definition in (document.get("services") or {}).items():
            seen_services += 1
            labels = (definition or {}).get("labels") or {}
            if labels.get(REPO_ID_LABEL) != identity:
                unlabelled_services.append(f"{where}::{name}")

        for name, definition in (document.get("volumes") or {}).items():
            seen_volumes += 1
            labels = definition.get("labels") or {} if isinstance(definition, dict) else {}
            if labels.get(REPO_ID_LABEL) != identity:
                unlabelled_volumes.append(f"{where}::{name}")

    assert seen_services, "the exemplar rendered no services to check"
    assert seen_volumes, "the exemplar rendered no named volumes to check"
    assert not unlabelled_services, f"containers without {REPO_ID_LABEL}: {unlabelled_services}"
    assert not unlabelled_volumes, f"volumes without {REPO_ID_LABEL}: {unlabelled_volumes}"


def test_the_web_stack_labels_every_container_and_volume_too(tmp_path):
    """The other half of the deployment, which the build does not render.

    ``osprey build`` renders the services stack; the web stack's compose file is
    written at start by the web-terminal path. Both are one deployment, so a
    verb reading labels back has to find them on both — and the web stack is
    where the per-user containers and their durable volumes live.
    """
    import yaml

    from osprey.deployment.web_terminals.render import render_web_terminals

    repo = tmp_path / "als-exemplar"
    repo.mkdir()
    rendered = render_web_terminals(
        {
            "project_root": str(repo),
            "facility": {"prefix": "ex"},
            "deploy": {"fqdn": "example.invalid"},
            "modules": {
                "web_terminals": {
                    "enabled": True,
                    "nginx_port": 9080,
                    "web_base_port": 8100,
                    "artifact_base_port": 8200,
                    "ariel_base_port": 8300,
                    "lattice_base_port": 8400,
                    # Auth on, so the sidecar renders and is covered too.
                    # allow_insecure_http because password auth otherwise
                    # refuses to render without TLS, which is not what this
                    # test is about.
                    "auth": {"method": "password", "allow_insecure_http": True},
                    "users": ["alice", "bob"],
                }
            },
        }
    )

    identity = repo_identity(repo)
    document = yaml.safe_load(rendered["docker-compose.web.yml"]) or {}

    services = document.get("services") or {}
    volumes = document.get("volumes") or {}
    assert services and volumes

    unlabelled = [
        name
        for name, definition in {**services, **volumes}.items()
        if ((definition or {}).get("labels") or {}).get(REPO_ID_LABEL) != identity
    ]
    assert not unlabelled, f"web-stack entries without {REPO_ID_LABEL}: {unlabelled}"
