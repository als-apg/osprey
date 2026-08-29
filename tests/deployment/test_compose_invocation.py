"""The compose invocation contract: one pinned base for every stack (TR-3).

Every compose command OSPREY runs is built from one argv base::

    <runtime> compose [--progress plain]
        --project-directory <repo-root>
        -f <repo-root>/build/<file> ...
        [--env-file <repo-root>/.env]
        <subcommand> ...

``--project-directory`` is the load-bearing part. Compose resolves every
relative path in every ``-f`` file — bind-mount sources, ``build:`` contexts,
``env_file:`` entries — against that ONE directory, and derives it from the
first ``-f`` file's location when it is not given. Before it was pinned, the
services stack resolved its paths against ``build/services/`` and the web stack
against whatever directory it happened to be invoked from; nothing in either
file could name both the render zone and the repo's own ``.env`` correctly, and
"whatever directory it was invoked from" is not a property of the deployment at
all.

Three groups of tests, matching the three ways that used to go wrong:

* the argv itself — pinned base, repo-anchored ``-f``, repo-anchored env file;
* the templates — every relative host path spelled against the repo root, and
  every one of them resolving to something a build actually renders;
* the web stack — its artifacts written into ``build/`` (never into the tracked
  source zone at the repo root), and the roster verbs acting on that same file.
  The last one is a security property, not a tidiness one: ``osprey users
  passwd`` reported success while recreating nothing, so a rotated password was
  written to ``.env.auth`` and never put in force.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from osprey.deployment import compose_generator, container_lifecycle, status_display
from osprey.deployment.compose_generator import (
    COMPOSE_ENV_FILENAME,
    compose_base_cmd,
    compose_env_file_args,
    compose_provider_env,
    resolve_repo_root,
)
from osprey.deployment.compose_merge import MERGED_COMPOSE_FILENAME

# Bound at import, before any fixture can patch the module attribute, so the
# provider seam itself can be exercised while the suite's autouse stub keeps
# every other test off the host probe.
from osprey.deployment.container_lifecycle import ENV_MERGED_RELPATH
from osprey.deployment.container_lifecycle import _compose_provider as _real_compose_provider
from osprey.deployment.runtime_helper import (
    ComposeProvider,
    ComposeProviderInfo,
    UnsupportedComposeProviderError,
)
from osprey.deployment.web_terminals import lifecycle, provision
from osprey.deployment.web_terminals.artifacts import (
    WEB_COMPOSE_FILENAME,
    web_artifacts_dir,
    web_compose_file,
    write_web_terminal_artifacts,
)
from osprey.deployment.web_terminals.auth_credentials import (
    AUTH_ENV_FILENAME,
    PW_HASH_VAR_PREFIX,
)
from osprey.services.auth_sidecar.passwords import verify_password
from osprey.utils.dotenv import parse_dotenv_file
from osprey.utils.workspace import BUILD_DIR_NAME
from tests.deployment.test_up_as_built import _RENDERED_CONFIG, render_build
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

#: A rendered config that also turns the web-terminal stack on — the shape whose
#: teardown runs before ``down``'s label fallback.
_WEB_CONFIG = _RENDERED_CONFIG + "modules:\n  web_terminals:\n    enabled: true\n"

#: The same, filled in far enough that the web artifacts actually RENDER — the
#: verbs below run the real render, because the merged document the podman shape
#: addresses is built from the file it writes.
_WEB_STACK_CONFIG = _RENDERED_CONFIG + (
    "facility:\n"
    "  name: Demo Light Source\n"
    "  prefix: dls\n"
    "  timezone: UTC\n"
    "registry:\n"
    "  url: registry.example.org\n"
    "deploy:\n"
    "  fqdn: deploy.example.org\n"
    "modules:\n"
    "  web_terminals:\n"
    "    enabled: true\n"
    "    users:\n"
    "      - alice\n"
    "    auth:\n"
    "      method: password\n"
    "      allow_insecure_http: true\n"
)

# ---------------------------------------------------------------------------
# The pinned argv
# ---------------------------------------------------------------------------


def test_base_cmd_pins_project_directory_files_and_env_file(tmp_path: Path) -> None:
    """The whole contract in one argv, in order."""
    (tmp_path / COMPOSE_ENV_FILENAME).write_text("A=1\n", encoding="utf-8")

    cmd = compose_base_cmd(
        ["docker", "compose"],
        ["build/services/docker-compose.yml", "build/services/bluesky/docker-compose.yml"],
        tmp_path,
    )

    assert cmd == [
        "docker",
        "compose",
        "--project-directory",
        str(tmp_path),
        "-f",
        str(tmp_path / "build" / "services" / "docker-compose.yml"),
        "-f",
        str(tmp_path / "build" / "services" / "bluesky" / "docker-compose.yml"),
        "--env-file",
        str(tmp_path / COMPOSE_ENV_FILENAME),
    ]


def test_base_cmd_is_independent_of_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same repo, different cwd, byte-identical argv.

    The point of the contract: an invocation describes the deployment, not the
    directory the operator was standing in when they typed the verb.
    """
    repo = tmp_path / "repo"
    (repo / "build").mkdir(parents=True)
    (repo / COMPOSE_ENV_FILENAME).write_text("A=1\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(repo)
    from_root = compose_base_cmd(["docker", "compose"], ["build/docker-compose.yml"], repo)
    monkeypatch.chdir(elsewhere)
    from_elsewhere = compose_base_cmd(["docker", "compose"], ["build/docker-compose.yml"], repo)

    assert from_root == from_elsewhere


def test_base_cmd_leaves_an_absolute_compose_file_alone(tmp_path: Path) -> None:
    absolute = tmp_path / "build" / "docker-compose.web.yml"
    cmd = compose_base_cmd(["podman", "compose"], [absolute], tmp_path)
    assert cmd[cmd.index("-f") + 1] == str(absolute)


def test_base_cmd_omits_the_env_file_when_the_repo_has_none(tmp_path: Path) -> None:
    """Compose hard-fails on an ``--env-file`` naming a file that is not there."""
    cmd = compose_base_cmd(["docker", "compose"], ["build/docker-compose.yml"], tmp_path)
    assert "--env-file" not in cmd


def test_env_file_is_the_repos_own_not_the_working_directorys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.env`` in the cwd is not this deployment's secret store.

    Resolved against the cwd, a subdirectory invocation found no ``.env`` (and
    started the stack with empty credentials), while an invocation made from
    some other deployment's directory found the WRONG one.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / COMPOSE_ENV_FILENAME).write_text("REPO=1\n", encoding="utf-8")
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    (stranger / COMPOSE_ENV_FILENAME).write_text("STRANGER=1\n", encoding="utf-8")

    monkeypatch.chdir(stranger)
    assert compose_env_file_args(repo) == ["--env-file", str(repo / COMPOSE_ENV_FILENAME)]


def test_repo_root_prefers_the_config_path_over_a_container_project_root(tmp_path: Path) -> None:
    """``--runtime-root`` records a path that exists only inside a container.

    Taking ``project_root`` at face value would pin compose to ``/app/<name>``
    on a host that has no such directory, so the config's own location wins.
    """
    build = tmp_path / BUILD_DIR_NAME
    build.mkdir()
    config_path = build / "config.yml"
    config_path.write_text("project_root: /app/als-assistant\n", encoding="utf-8")

    resolved = resolve_repo_root({"project_root": "/app/als-assistant"}, config_path)

    assert resolved == tmp_path.absolute()


def test_repo_root_falls_back_to_the_cwd_for_an_unusable_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_root({"project_root": "/app/als-assistant"}) == tmp_path.absolute()


# ---------------------------------------------------------------------------
# The same contract, in the shape podman-compose can parse
# ---------------------------------------------------------------------------
#
# podman-compose parses no ``--project-directory`` — no version of it does — so
# the pin has to be carried by the ``-f`` list and the environment instead: one
# merged document AT THE REPO ROOT (whose own directory is therefore the root
# every version chdirs into) plus ``COMPOSE_PROJECT_DIR`` for the versions that
# read it. And one pre-merged ``--env-file``, because a repeated flag means
# "keep the last" there rather than "merge these". Same deployment, same project
# directory, same values interpolated — a different spelling of it.


def _repo_with_chain(tmp_path: Path) -> Path:
    """A repo carrying both env-chain files and two rendered compose files."""
    (tmp_path / ".env.shared").write_text("SHARED=1\nBOTH=shared\n", encoding="utf-8")
    (tmp_path / COMPOSE_ENV_FILENAME).write_text("LOCAL=1\nBOTH=local\n", encoding="utf-8")
    services = tmp_path / BUILD_DIR_NAME / "services"
    services.mkdir(parents=True)
    (services / "docker-compose.yml").write_text(
        "services:\n  event-dispatcher:\n    image: demo:local\n", encoding="utf-8"
    )
    (services / "bluesky" / "docker-compose.yml").parent.mkdir()
    (services / "bluesky" / "docker-compose.yml").write_text(
        "services:\n  bluesky:\n    image: bluesky:local\n", encoding="utf-8"
    )
    return tmp_path


_CHAIN_COMPOSE_FILES = [
    f"{BUILD_DIR_NAME}/services/docker-compose.yml",
    f"{BUILD_DIR_NAME}/services/bluesky/docker-compose.yml",
]


def test_podman_compose_shape_is_one_merged_document_and_one_merged_env_file(
    tmp_path: Path,
) -> None:
    """The whole podman-compose contract in one argv, in order."""
    repo = _repo_with_chain(tmp_path)

    cmd = compose_base_cmd(
        ["podman", "compose"],
        _CHAIN_COMPOSE_FILES,
        repo,
        provider=ComposeProvider.PODMAN_COMPOSE,
    )

    assert cmd == [
        "podman",
        "compose",
        "-f",
        str(repo / MERGED_COMPOSE_FILENAME),
        "--env-file",
        str(repo / ENV_MERGED_RELPATH),
    ]


def test_the_podman_shape_never_passes_project_directory(tmp_path: Path) -> None:
    """No podman-compose version parses the flag; passing it aborts the verb.

    Asserted on its own, and not only through the argv above, because this is
    the whole reason the shape exists: a ``--project-directory`` that reached
    podman-compose would not be ignored, it would fail the invocation outright.
    """
    cmd = compose_base_cmd(
        ["podman", "compose"],
        _CHAIN_COMPOSE_FILES,
        _repo_with_chain(tmp_path),
        provider=ComposeProvider.PODMAN_COMPOSE,
    )

    assert "--project-directory" not in cmd


def test_the_single_f_file_carries_every_service_the_f_list_declared(tmp_path: Path) -> None:
    """One ``-f`` is only correct if nothing was dropped on the way into it.

    The merge is what makes the single file legitimate: a service that fell out
    of it is a service the deploy silently never starts.
    """
    repo = _repo_with_chain(tmp_path)

    compose_base_cmd(
        ["podman", "compose"],
        _CHAIN_COMPOSE_FILES,
        repo,
        provider=ComposeProvider.PODMAN_COMPOSE,
    )

    merged = yaml.safe_load((repo / MERGED_COMPOSE_FILENAME).read_text(encoding="utf-8"))
    assert set(merged["services"]) == {"event-dispatcher", "bluesky"}


def test_the_podman_shapes_env_file_is_the_whole_chain_with_the_local_file_winning(
    tmp_path: Path,
) -> None:
    """One file, because a repeated flag keeps only the last one there.

    Handed the chain as two flags, podman-compose would interpolate ``.env``
    alone and every default that lives only in ``.env.shared`` would resolve
    empty — with no error anywhere.
    """
    repo = _repo_with_chain(tmp_path)

    cmd = compose_base_cmd(
        ["podman", "compose"],
        _CHAIN_COMPOSE_FILES,
        repo,
        provider=ComposeProvider.PODMAN_COMPOSE,
    )

    assert cmd.count("--env-file") == 1
    merged = parse_dotenv_file(repo / ENV_MERGED_RELPATH)
    assert merged["SHARED"] == "1"
    assert merged["LOCAL"] == "1"
    assert merged["BOTH"] == "local"


def test_the_docker_shape_is_what_an_unprobed_and_a_docker_v2_call_both_get(
    tmp_path: Path,
) -> None:
    """The branch is opt-in: naming Docker Compose v2 changes nothing.

    A caller that has not probed gets today's argv, and a caller that probed and
    found docker gets the same one — so the podman shape can only ever be
    reached deliberately.
    """
    repo = _repo_with_chain(tmp_path)

    unprobed = compose_base_cmd(["docker", "compose"], _CHAIN_COMPOSE_FILES, repo)
    docker_v2 = compose_base_cmd(
        ["docker", "compose"],
        _CHAIN_COMPOSE_FILES,
        repo,
        provider=ComposeProvider.DOCKER_V2,
    )

    assert unprobed == docker_v2
    assert unprobed == [
        "docker",
        "compose",
        "--project-directory",
        str(repo),
        "-f",
        str(repo / _CHAIN_COMPOSE_FILES[0]),
        "-f",
        str(repo / _CHAIN_COMPOSE_FILES[1]),
        "--env-file",
        str(repo / ".env.shared"),
        "--env-file",
        str(repo / COMPOSE_ENV_FILENAME),
    ]
    assert not (repo / MERGED_COMPOSE_FILENAME).exists()


def test_the_docker_shape_leaves_no_merged_document_behind(tmp_path: Path) -> None:
    """The artifact belongs to the shape that needs it.

    A ``.osprey-compose.yml`` written on a docker deploy would sit at the repo
    root going stale, and be read by nothing.
    """
    repo = _repo_with_chain(tmp_path)

    compose_base_cmd(["docker", "compose"], _CHAIN_COMPOSE_FILES, repo)

    assert not (repo / MERGED_COMPOSE_FILENAME).exists()


def test_the_project_directory_is_pinned_in_the_environment_for_podman_compose(
    tmp_path: Path,
) -> None:
    """The half of the pin that cannot be spelled in argv.

    podman-compose 1.4.1+ take the project directory from this variable, and
    from the working directory when it is unset — which would resolve every
    relative path in the merged document against wherever the verb was typed.
    """
    assert compose_provider_env(ComposeProvider.PODMAN_COMPOSE, tmp_path) == {
        "COMPOSE_PROJECT_DIR": str(tmp_path)
    }


def test_the_docker_shape_needs_no_environment_pin(tmp_path: Path) -> None:
    """It carries the project directory in argv, so nothing is layered on."""
    assert compose_provider_env(None, tmp_path) == {}
    assert compose_provider_env(ComposeProvider.DOCKER_V2, tmp_path) == {}


def test_the_shape_a_verb_emits_is_the_one_the_probe_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam itself: the runtime the module resolved, probed once.

    Bound to the module-level names the deploy tests patch, so a test can put a
    provider in front of a whole lifecycle verb without a runtime on the host —
    and so the answer for one invocation cannot come from a different runtime
    than the argv it shapes.
    """
    probed: list[list[str]] = []
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config=None: ["podman", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle,
        "detect_compose_provider",
        lambda cmd=None, config=None: (
            probed.append(list(cmd))
            or ComposeProviderInfo(
                ComposeProvider.PODMAN_COMPOSE, (1, 0, 6), "1.0.6", "podman-compose"
            )
        ),
    )

    assert _real_compose_provider({}) is ComposeProvider.PODMAN_COMPOSE
    assert probed == [["podman", "compose"]]


# Every verb below is exercised against a podman-compose host, because the two
# shapes are only distinguishable there: with DOCKER_V2 the argv is identical to
# the one an unprobed call already produces, so a call site that dropped the
# provider on the floor would look exactly like one that threaded it.


def _podman_host(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Put a podman-compose host in front of the lifecycle verbs.

    Returns the list every compose invocation is recorded into, newest last.
    """
    runs: list[dict] = []

    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config=None: ["podman", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "_compose_provider", lambda config=None: ComposeProvider.PODMAN_COMPOSE
    )
    # Probes real listeners on this machine; which ports are free here is not
    # what any of these tests are about.
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda cmd, env=None, **kwargs: (
            runs.append({"cmd": list(cmd), "env": dict(env or {})})
            or subprocess.CompletedProcess(list(cmd), 0, "", "")
        ),
    )
    return runs


def _assert_podman_shaped(run: dict, repo: Path) -> None:
    """The podman-compose contract, as one recorded invocation should carry it.

    The merged document is identified by being the only compose file in the
    argv rather than by counting ``-f``: a subcommand's own flags can spell that
    too (``rm -f`` does), and this has to hold for every invocation a verb makes.
    """
    cmd = run["cmd"]
    assert "--project-directory" not in cmd, cmd
    assert cmd[cmd.index("-f") + 1] == str(repo / MERGED_COMPOSE_FILENAME), cmd
    assert [arg for arg in cmd if arg.endswith((".yml", ".yaml"))] == [
        str(repo / MERGED_COMPOSE_FILENAME)
    ], cmd
    assert cmd.count("--env-file") == 1, cmd
    assert cmd[cmd.index("--env-file") + 1] == str(repo / ENV_MERGED_RELPATH), cmd
    assert run["env"].get("COMPOSE_PROJECT_DIR") == str(repo), run["env"]


def _started_repo(tmp_path: Path, config: str | None = None, **kwargs) -> Path:
    """An exemplar repo with a build in it, and a ``.env`` to interpolate from."""
    repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME)
    render_build(repo, **({"config": config} if config else {}), **kwargs)
    (repo / COMPOSE_ENV_FILENAME).write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    return repo


def _outside(tmp_path: Path) -> Path:
    """A directory that is not the deployment — where an operator may be standing.

    Nothing a verb does may depend on it, and nothing may be written into it.
    """
    outside = tmp_path / "not-the-deployment"
    outside.mkdir(exist_ok=True)
    return outside


def test_down_emits_the_whole_podman_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end, through a real verb: argv, environment and artifacts.

    The unit tests above pin the seam; this pins what actually reaches the
    runtime when an operator types a verb on a podman-compose host — including
    the ``COMPOSE_PROJECT_DIR`` half, which lives in the environment and would
    otherwise be easy to thread into the argv builder and forget at the call
    site.
    """
    repo = _started_repo(tmp_path)
    runs = _podman_host(monkeypatch)

    container_lifecycle.down_deployment(repo)

    assert len(runs) == 1, runs
    assert runs[0]["cmd"] == [
        "podman",
        "compose",
        "-f",
        str(repo / MERGED_COMPOSE_FILENAME),
        "--env-file",
        str(repo / ENV_MERGED_RELPATH),
        "down",
    ]
    assert runs[0]["env"]["COMPOSE_PROJECT_DIR"] == str(repo)
    assert (repo / MERGED_COMPOSE_FILENAME).is_file()


def test_up_emits_the_podman_shape_on_every_invocation_it_makes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start makes several compose calls, and one shape has to cover them all.

    The stale-container ``rm -f`` and the ``up`` are separate invocations of the
    same project. Shaped differently they would address two different project
    directories, and the ``rm`` would be reconciling a project the ``up`` never
    creates.
    """
    repo = _started_repo(tmp_path)
    runs = _podman_host(monkeypatch)

    container_lifecycle.up_as_built(repo, detached=True)

    assert len(runs) >= 2, runs
    for run in runs:
        _assert_podman_shaped(run, repo)
    assert any("up" in run["cmd"] for run in runs), runs


def test_the_legacy_down_verb_emits_the_podman_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deploy_down`` is the last thing its process does, so it propagates the
    child's own exit status rather than returning — the argv and its env are
    asserted from the recorded run."""
    repo = _started_repo(tmp_path)
    runs = _podman_host(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        container_lifecycle.deploy_down(str(container_lifecycle.as_built_config_path(repo)))

    assert exit_info.value.code == 0
    assert runs, "deploy_down never reached a compose invocation"
    _assert_podman_shaped(runs[-1], repo)
    assert runs[-1]["cmd"][-1] == "down"


def test_the_legacy_restart_verb_emits_the_podman_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _started_repo(tmp_path)
    runs = _podman_host(monkeypatch)
    config = yaml.safe_load(
        container_lifecycle.as_built_config_path(repo).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (config, [f"{BUILD_DIR_NAME}/services/docker-compose.yml"]),
    )

    container_lifecycle.deploy_restart(str(container_lifecycle.as_built_config_path(repo)))

    assert runs, "deploy_restart never reached a compose invocation"
    _assert_podman_shaped(runs[-1], repo)
    assert runs[-1]["cmd"][-1] == "restart"


def test_the_archiver_staging_runs_in_the_shape_the_start_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store is brought up by a separate invocation of the same project.

    Left unthreaded it would create the store's container against a different
    project directory than the ``up`` that goes on to run beside it — so what is
    asserted here is the threading, the argv being ``compose_base_cmd``'s and
    already pinned above.
    """
    repo = _started_repo(tmp_path)
    _podman_host(monkeypatch)
    staged: list[dict] = []
    monkeypatch.setattr(container_lifecycle, "_archiver_store_deployed", lambda config: True)
    monkeypatch.setattr(
        container_lifecycle,
        "_stage_archiver_store",
        lambda *args, **kwargs: staged.append(kwargs),
    )

    container_lifecycle.up_as_built(repo, detached=True)

    assert staged, "the archiver staging step never ran"
    assert staged[0]["provider"] is ComposeProvider.PODMAN_COMPOSE


def test_the_ariel_staging_runs_in_the_shape_the_start_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The logbook twin of the archiver test above, for the same reason.

    ARIEL's store is brought up by its own invocation of the same project;
    unthreaded it would run docker-shaped in the middle of a podman-shaped
    ``up``, addressing a different project directory with part of the env chain
    dropped.
    """
    repo = _started_repo(tmp_path)
    _podman_host(monkeypatch)
    staged: list[dict] = []
    monkeypatch.setattr(container_lifecycle, "_ariel_store_deployed", lambda config: True)
    monkeypatch.setattr(
        container_lifecycle,
        "_stage_ariel_store",
        lambda *args, **kwargs: staged.append(kwargs),
    )

    container_lifecycle.up_as_built(repo, detached=True)

    assert staged, "the ARIEL staging step never ran"
    assert staged[0]["provider"] is ComposeProvider.PODMAN_COMPOSE


def test_a_refused_provider_still_reaches_the_label_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recovery path must not be gated on the thing that is broken.

    ``down`` falls back to removing this checkout's labelled containers when
    there are no compose files to act on — the one way to stop a deployment
    compose cannot address. Resolving the provider for the web-terminal teardown
    ahead of that fallback made an unsupported host abort before reaching it,
    which is precisely the host that needs it.
    """
    repo = _started_repo(tmp_path, config=_WEB_CONFIG, with_compose=False)
    swept: list[Path] = []
    monkeypatch.setattr(
        container_lifecycle,
        "_compose_provider",
        lambda config=None: (_ for _ in ()).throw(UnsupportedComposeProviderError("no provider")),
    )
    monkeypatch.setattr(
        container_lifecycle, "_down_by_label", lambda config, root: swept.append(Path(root))
    )

    container_lifecycle.down_deployment(repo)

    assert swept == [repo]


def test_clean_emits_the_podman_shape_on_both_of_its_invocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clean`` destroys volumes, so the project it addresses has to be right.

    A ``down --volumes`` aimed at a project directory the operator happened to
    be standing in reports success having removed nothing — and the deployment
    it was meant to clear is still there for the rebuild that follows.
    """
    repo = _started_repo(tmp_path)
    runs: list[dict] = []
    config = yaml.safe_load(
        container_lifecycle.as_built_config_path(repo).read_text(encoding="utf-8")
    )
    # `clean` takes no config PATH, so the repo has to be handed to it: left to
    # resolve its own, it would answer this working directory, and a
    # `down --volumes` there is aimed at nothing.
    monkeypatch.chdir(_outside(tmp_path))
    monkeypatch.setattr(
        container_lifecycle, "_compose_provider", lambda config=None: ComposeProvider.PODMAN_COMPOSE
    )
    monkeypatch.setattr(
        compose_generator, "get_runtime_command", lambda config=None: ["podman", "compose"]
    )
    # The capture helper is the seam `clean_deployment` actually runs through,
    # so it is the one stubbed: reaching past it to `subprocess.run` would
    # depend on how the helper spawns the child, and it writes a spool file on
    # the way past.
    monkeypatch.setattr(
        compose_generator,
        "run_captured",
        lambda cmd, *, env=None, **kwargs: (
            runs.append({"cmd": list(cmd), "env": dict(env or {})})
            or subprocess.CompletedProcess(list(cmd), 0, "", "")
        ),
    )

    compose_generator.clean_deployment(
        list(container_lifecycle.as_built_compose_files(config, repo)), config, repo_root=repo
    )

    assert len(runs) == 2, runs
    for run in runs:
        _assert_podman_shaped(run, repo)
    assert [run["cmd"][-1] for run in runs] == ["--remove-orphans", "all"]
    assert not (_outside(tmp_path) / MERGED_COMPOSE_FILENAME).exists()


def test_logs_emits_the_podman_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``logs`` is read-only, which is exactly why the shape matters.

    A misaddressed ``down`` fails visibly; a misaddressed ``logs`` prints
    nothing at all and reads as a deployment whose services log nothing.
    """
    repo = _started_repo(tmp_path)
    monkeypatch.setattr(
        container_lifecycle, "_compose_provider", lambda config=None: ComposeProvider.PODMAN_COMPOSE
    )
    monkeypatch.setattr(
        status_display, "get_runtime_command", lambda config=None: ["podman", "compose"]
    )

    cmd, env = status_display.logs_command(repo)

    _assert_podman_shaped({"cmd": cmd, "env": env}, repo)
    assert cmd[-1] == "logs"


def _podman_web_host(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """A podman-compose host with the web path's non-compose collaborators muted.

    Everything that reaches past compose — persona and sidecar image builds, the
    post-``up`` hooks, the host-reachability probe — is stubbed; the compose
    invocations themselves, the artifact render they address and the merge that
    render feeds are all real, because they are what is being asserted.
    """
    runs = _podman_host(monkeypatch)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    monkeypatch.setattr(container_lifecycle, "preflight_web_terminals", lambda *a, **k: None)
    for name in (
        "ensure_env_production",
        "build_persona_images",
        "build_auth_sidecar_image",
        "verify_persona_renders",
        "_reconcile_web_stack_recreates",
        "reload_nginx_config",
        "enable_linger",
        "seed_user_containers",
        "run_verify_script",
        "warn_if_web_stack_unreachable",
    ):
        monkeypatch.setattr(provision, name, lambda *a, **k: None)
    return runs


def test_the_web_stack_start_emits_the_podman_shape_in_argv_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web path builds its own argv AND its own subprocess environment.

    It is handed the provider-shaped ``--env-file`` fragment the start already
    resolved, so a provider left unthreaded here produces the one combination
    nothing else can: a single pre-merged env file spliced into a docker-shaped
    argv, run with no ``COMPOSE_PROJECT_DIR`` at all.
    """
    repo = _started_repo(tmp_path, config=_WEB_STACK_CONFIG)
    runs = _podman_web_host(monkeypatch)
    # Deliberately NOT standing in the deployment: the start hands the web path
    # the repo root it resolved, and the config's recorded `project_root` is a
    # container path, so a web path that re-resolved the root instead would land
    # on this directory.
    monkeypatch.chdir(_outside(tmp_path))

    container_lifecycle.up_as_built(repo, detached=True)

    assert runs, "the web path never reached a compose invocation"
    for run in runs:
        _assert_podman_shaped(run, repo)
    assert any(cmd[-2:] == ["up", "-d"] for cmd in (run["cmd"] for run in runs)), runs


def test_the_web_stack_teardown_emits_the_podman_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``down``'s web teardown is a separate invocation of the same project."""
    repo = _started_repo(tmp_path, config=_WEB_STACK_CONFIG)
    config = yaml.safe_load(
        container_lifecycle.as_built_config_path(repo).read_text(encoding="utf-8")
    )
    write_web_terminal_artifacts(config, repo)
    runs = _podman_web_host(monkeypatch)
    monkeypatch.chdir(_outside(tmp_path))

    container_lifecycle.down_deployment(repo)

    assert len(runs) == 2, runs
    for run in runs:
        _assert_podman_shaped(run, repo)
        assert run["cmd"][-1] == "down"


def test_the_web_path_writes_its_merged_document_into_the_repo_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The podman shape WRITES, so its repo root is a file-system effect.

    ``down_deployment`` is handed the repo as an argument and the start resolves
    it from the config path, but the web path used to re-derive its own — and
    ``resolve_repo_root``'s last resort is the working directory. The merged
    document is written at whatever that answers, so a verb run from outside the
    deployment left ``.osprey-compose.yml`` in an unrelated directory and pointed
    the invocation at it.
    """
    repo = _started_repo(tmp_path, config=_WEB_STACK_CONFIG)
    config = yaml.safe_load(
        container_lifecycle.as_built_config_path(repo).read_text(encoding="utf-8")
    )
    write_web_terminal_artifacts(config, repo)
    _podman_web_host(monkeypatch)
    outside = _outside(tmp_path)
    monkeypatch.chdir(outside)

    container_lifecycle.up_as_built(repo, detached=True)
    container_lifecycle.down_deployment(repo)

    assert (repo / MERGED_COMPOSE_FILENAME).is_file()
    assert not (outside / MERGED_COMPOSE_FILENAME).exists()
    assert (repo / ENV_MERGED_RELPATH).is_file()
    assert not (outside / ENV_MERGED_RELPATH).exists()


def test_a_roster_verb_probes_for_the_shape_nobody_handed_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``passwd``/``decommission``/``prune`` enter from the CLI with nothing
    threaded down, and must still emit what their host parses — both halves.
    """
    repo = _started_repo(tmp_path, config=_WEB_STACK_CONFIG)
    config = yaml.safe_load(
        container_lifecycle.as_built_config_path(repo).read_text(encoding="utf-8")
    )
    write_web_terminal_artifacts(config, repo)
    runs = _podman_web_host(monkeypatch)

    provision.force_recreate_auth_sidecar(config, repo_root=repo, env={})

    assert len(runs) == 1, runs
    _assert_podman_shaped(runs[0], repo)
    assert runs[0]["cmd"][-1] == provision.AUTH_SERVICE_NAME


def test_the_roster_verbs_nginx_reload_carries_both_halves_of_the_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reload after a decommission/prune builds its own environment too.

    It is the one invocation whose argv comes from ``web_stack_compose_cmd``
    while its environment is assembled at the call site, so it is the one place
    where the two halves can disagree: a podman-shaped argv carries no
    ``--project-directory``, and without ``COMPOSE_PROJECT_DIR`` the reload
    resolves the project against whatever directory the operator typed the verb
    in — reloading nothing, or something else's nginx.
    """
    repo = _started_repo(tmp_path, config=_WEB_STACK_CONFIG)
    config = yaml.safe_load(
        container_lifecycle.as_built_config_path(repo).read_text(encoding="utf-8")
    )
    write_web_terminal_artifacts(config, repo)
    _podman_web_host(monkeypatch)
    # Stand somewhere that is NOT the deployment, which is what the environment
    # half exists to survive.
    monkeypatch.chdir(tmp_path)

    reloads: list[dict] = []
    monkeypatch.setattr(
        lifecycle,
        "reload_nginx_config",
        lambda cmd, env: reloads.append({"cmd": list(cmd), "env": dict(env)}),
    )

    lifecycle._reconcile_auth_after_user_removal(config, ["bob"], rerendered=True, repo_root=repo)

    assert len(reloads) == 1, reloads
    _assert_podman_shaped(reloads[0], repo)


def test_neither_shape_selects_services_with_a_compose_profile(tmp_path: Path) -> None:
    """``profiles:`` is not part of the invocation contract, on either provider.

    podman-compose's handling of profiles varies by version, and the failure
    mode is the worst one available: a service quietly left out of the project,
    with a deploy that reports success. Every deployed service is in the ``-f``
    list, and everything in the ``-f`` list starts.
    """
    repo = _repo_with_chain(tmp_path)

    docker = compose_base_cmd(["docker", "compose"], _CHAIN_COMPOSE_FILES, repo)
    podman = compose_base_cmd(
        ["podman", "compose"],
        _CHAIN_COMPOSE_FILES,
        repo,
        provider=ComposeProvider.PODMAN_COMPOSE,
    )

    for cmd in (docker, podman):
        assert "--profile" not in cmd
        assert not any(argument.startswith("--profile") for argument in cmd)


# ---------------------------------------------------------------------------
# The templates, audited against the pinned base
# ---------------------------------------------------------------------------

_TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
_SERVICE_TEMPLATES = sorted((_TEMPLATE_ROOT / "services").rglob("docker-compose.yml.j2"))
_WEB_TEMPLATE = _TEMPLATE_ROOT / "modules" / "web_terminals" / "docker-compose.web.yml.j2"

#: Every relative host path a compose template can spell: bind-mount sources,
#: ``build:`` contexts, and ``env_file:`` entries.
_HOST_PATH_RE = re.compile(
    r"^\s*(?:-\s+(?P<mount>\./[^:\s]+):|context:\s+(?P<context>\./\S+)|-\s+(?P<env>\./\S+)\s*$)",
    re.MULTILINE,
)


def _relative_host_paths(text: str) -> list[str]:
    return [
        m.group("mount") or m.group("context") or m.group("env")
        for m in _HOST_PATH_RE.finditer(text)
    ]


def test_the_audit_covers_every_shipped_service_template() -> None:
    """A template added later must not slip past the two audits below."""
    names = {path.parent.name for path in _SERVICE_TEMPLATES}
    assert "services" in names, "the top-level services template must be in the sweep"
    assert len(_SERVICE_TEMPLATES) >= 10, _SERVICE_TEMPLATES


@pytest.mark.parametrize("template", _SERVICE_TEMPLATES, ids=lambda p: p.parent.name)
def test_no_service_template_climbs_out_of_the_project_directory(template: Path) -> None:
    """``../../`` reached the project root from ``build/services/``.

    With the project directory pinned at the repo root there is nothing above it
    to reach, and a surviving ``../../`` now points OUTSIDE the deployment — at
    a sibling checkout, or at the operator's home directory.
    """
    for path in _relative_host_paths(template.read_text(encoding="utf-8")):
        assert ".." not in Path(path).parts, f"{template.parent.name}: {path}"


@pytest.mark.parametrize("template", [*_SERVICE_TEMPLATES, _WEB_TEMPLATE], ids=lambda p: p.stem)
def test_render_output_is_addressed_under_the_build_zone(template: Path) -> None:
    """A path naming render output has to spell ``build/``; a path naming the
    repo's own source or state zones must not.

    The two zones are the reason the pin matters: ``data/`` and ``.env`` are the
    operator's, tracked or durable, and survive ``rm -rf build/``; everything a
    build produces is disposable and lives under it. One project directory,
    named zones, no ambiguity.
    """
    repo_owned = ("./data/", "./.env", "./var/")
    for path in _relative_host_paths(template.read_text(encoding="utf-8")):
        if path.startswith(repo_owned):
            continue
        assert path.startswith(f"./{BUILD_DIR_NAME}/"), f"{template.stem}: {path}"


def test_dispatch_worker_env_file_is_the_repos_own_env_chain() -> None:
    """The worker's provider auth comes from the deployment's own env chain.

    Not from a copy inside the render — there is none, and there must not be:
    the tokens ``osprey up`` mints append to the repo's files, and a build wipes
    everything under ``build/``. The paths are no longer literals: the render
    context carries the chain members it found at the repo root, and the
    template lists exactly those, so what this checks is that nothing reaches
    past the pinned project directory. Which files the block names, in which
    order, is pinned on a RENDER in
    ``test_compose_generator.py::test_worker_env_file_lists_the_whole_chain_in_precedence_order``.
    """
    text = (_TEMPLATE_ROOT / "services" / "dispatch_worker" / "docker-compose.yml.j2").read_text(
        encoding="utf-8"
    )
    assert "for env_chain_file in osprey_env_chain" in text
    assert "- {{ env_chain_file }}" in text
    assert "../../.env" not in text


@pytest.mark.parametrize("template", [*_SERVICE_TEMPLATES, _WEB_TEMPLATE], ids=lambda p: p.stem)
def test_no_shipped_template_declares_a_compose_profile(template: Path) -> None:
    """The other half of the invocation's no-profiles rule (asserted on the argv
    above): nothing to select, so nothing that can be left out.

    An argv that carries no ``--profile`` is only safe while the templates
    declare none — a service rendered under a profile would be excluded by
    default, on both providers, and the deploy would still report success.
    Scanned as text rather than parsed, because a template is Jinja2 first and
    YAML only once it has been rendered.
    """
    for line in template.read_text(encoding="utf-8").splitlines():
        assert not line.strip().startswith("profiles:"), f"{template.stem}: {line.strip()}"


def test_web_stack_nginx_mounts_name_the_files_the_writer_writes(tmp_path: Path) -> None:
    """The rendered mount sources and the render's own destinations are one set.

    They are decided in two different places — the template's mount lines and
    :func:`write_web_terminal_artifacts`'s destination — so this resolves the
    former against the pinned project directory and checks each one against what
    the latter actually put there. Left unchecked, nginx starts with no config
    and the whole web tier serves 404s.

    Two mount shapes, one invariant. Most sources are single rendered files. The
    per-user secret templates are mounted as a DIRECTORY, because nginx's
    entrypoint envsubsts whatever it finds there — which is exactly why its
    contents are checked the same way: anything in that directory becomes live
    nginx config at container start, so a file the writer did not put there
    (a decommissioned user's leftover snippet) is a header still being injected
    for someone off the roster. An empty directory is legitimate — a roster
    whose secrets are not provisioned yet renders no snippets — but the
    directory itself must exist, or the container runtime materializes it
    root-owned.
    """
    config = {
        "facility": {"prefix": "als", "name": "ALS"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": ["alice"],
            }
        },
    }
    written = set(write_web_terminal_artifacts(config, repo_root=tmp_path))

    compose = yaml.safe_load(web_compose_file(tmp_path).read_text(encoding="utf-8"))
    # Bind mounts only. The envsubst output directory is a tmpfs declared on the
    # service's own `tmpfs:` key, not in this list: it has no host source and
    # nothing for the writer to have written.
    sources = [
        volume.split(":", 1)[0]
        for volume in compose["services"]["nginx"]["volumes"]
        if isinstance(volume, str)
    ]

    assert sources, "the nginx service must mount its rendered config"
    destinations = {path.resolve() for path in written}
    for source in sources:
        resolved = (tmp_path / source).resolve()
        assert resolved.exists(), f"{source} resolves to nothing under the project directory"
        if resolved.is_dir():
            for path in resolved.rglob("*"):
                if path.is_file():
                    assert path.resolve() in destinations, (
                        f"{source} holds {path.name}, which this render did not write"
                    )
            continue
        assert resolved.is_file()
        assert resolved in destinations


# ---------------------------------------------------------------------------
# The web stack's artifacts: rendered into build/, never into the source zone
# ---------------------------------------------------------------------------


def _web_config(users: list[str]) -> dict:
    return {
        "project_name": "demo-project",
        "facility": {"name": "Demo Light Source", "prefix": "dls", "timezone": "UTC"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": users,
                "auth": {"method": "password", "allow_insecure_http": True},
            }
        },
    }


def test_no_compose_or_nginx_artifact_lands_in_the_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repo root is tracked source. Nothing rendered belongs there.

    A stray ``docker-compose.web.yml`` and ``nginx/`` at the root dirtied
    ``git status`` on a fresh deployment, survived ``rm -rf build/``, and — the
    part that actually broke things — went on being read by the verbs that
    resolved their paths against the working directory.
    """
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())

    write_web_terminal_artifacts(_web_config(["alice"]))

    new_at_root = set(tmp_path.iterdir()) - before
    assert new_at_root == {tmp_path / BUILD_DIR_NAME}
    assert (web_artifacts_dir(tmp_path) / WEB_COMPOSE_FILENAME).is_file()
    assert (web_artifacts_dir(tmp_path) / "nginx" / "nginx.conf").is_file()


def test_the_web_stack_argv_addresses_the_rendered_build_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])

    cmd = provision.web_stack_compose_cmd({"project_name": "demo"}, [])

    assert cmd[cmd.index("--project-directory") + 1] == str(tmp_path)
    assert cmd[cmd.index("-f") + 1] == str(web_compose_file(tmp_path))


# ---------------------------------------------------------------------------
# The security property: a rotated password is put IN FORCE
# ---------------------------------------------------------------------------


def _repo_with_rendered_web_stack(tmp_path: Path) -> Path:
    """A repo whose stack has been rendered — i.e. one that IS deployed."""
    config = _web_config(["alice"])
    config_path = tmp_path / BUILD_DIR_NAME / "config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_web_terminal_artifacts(config, repo_root=tmp_path)
    return config_path


def test_passwd_rotation_recreates_the_sidecar_against_the_rendered_build_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rotation must be IN FORCE, not merely written down.

    Compose bakes ``env_file`` content into a container at creation, so the new
    hash reaches the sidecar only through a recreate. That recreate was gated on
    a probe of the working directory for ``docker-compose.web.yml`` — which a
    three-zone repo never has at its root — so the probe found nothing, warned
    at WARNING level, returned, and the CLI printed success over a password the
    running sidecar had never been given.
    """
    monkeypatch.chdir(tmp_path)
    config_path = _repo_with_rendered_web_stack(tmp_path)
    (tmp_path / AUTH_ENV_FILENAME).write_text(
        f"{PW_HASH_VAR_PREFIX}ALICE=stale\n", encoding="utf-8"
    )
    (tmp_path / COMPOSE_ENV_FILENAME).write_text("A=1\n", encoding="utf-8")

    argv: list[list[str]] = []
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(lifecycle, "_require_running_runtime", lambda config: None)
    monkeypatch.setattr(
        provision.subprocess,
        "run",
        lambda cmd, **kwargs: argv.append(list(cmd)) or _completed(),
    )

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    # The credential is stored...
    stored = parse_dotenv_file(tmp_path / AUTH_ENV_FILENAME).get(f"{PW_HASH_VAR_PREFIX}ALICE", "")
    assert verify_password("alices-new-password", stored)

    # ...AND put in force, by a recreate addressed at the file the running stack
    # was started from.
    recreates = [cmd for cmd in argv if "--force-recreate" in cmd]
    assert len(recreates) == 1, argv
    recreate = recreates[0]
    assert recreate[-1] == "auth", "a bare recreate would bounce every live terminal"
    assert recreate[recreate.index("-f") + 1] == str(web_compose_file(tmp_path))
    assert recreate[recreate.index("--project-directory") + 1] == str(tmp_path)


def test_passwd_rotation_ignores_a_stray_root_level_compose_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover root-level render is not the deployed stack.

    Debris from before the artifacts moved into ``build/`` must not resurrect
    the old resolution: acting on it would recreate a sidecar from a compose
    file the running stack was never started from.
    """
    monkeypatch.chdir(tmp_path)
    config = _web_config(["alice"])
    config_path = tmp_path / BUILD_DIR_NAME / "config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / WEB_COMPOSE_FILENAME).write_text("services: {}\n", encoding="utf-8")

    argv: list[list[str]] = []
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(lifecycle, "_require_running_runtime", lambda config: None)
    monkeypatch.setattr(
        provision.subprocess,
        "run",
        lambda cmd, **kwargs: argv.append(list(cmd)) or _completed(),
    )

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert argv == []


class _completed:
    """The subset of ``CompletedProcess`` the recreate path reads."""

    returncode = 0
    stdout = ""
    stderr = ""
