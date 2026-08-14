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
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import (
    COMPOSE_ENV_FILENAME,
    compose_base_cmd,
    compose_env_file_args,
    resolve_repo_root,
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


def test_dispatch_worker_env_file_is_the_repos_single_dotenv() -> None:
    """The worker's provider auth comes from the deployment's one ``.env``.

    Not from a copy inside the render — there is none, and there must not be:
    the tokens ``osprey up`` mints append to the repo's file, and a build wipes
    everything under ``build/``.
    """
    text = (_TEMPLATE_ROOT / "services" / "dispatch_worker" / "docker-compose.yml.j2").read_text(
        encoding="utf-8"
    )
    assert "- ./.env" in text
    assert "../../.env" not in text


def test_web_stack_nginx_mounts_name_the_files_the_writer_writes(tmp_path: Path) -> None:
    """The rendered mount sources and the render's own destinations are one set.

    They are decided in two different places — the template's mount lines and
    :func:`write_web_terminal_artifacts`'s destination — so this resolves the
    former against the pinned project directory and checks each one is a file
    the latter actually wrote. Left unchecked, nginx starts with no config and
    the whole web tier serves 404s.
    """
    config = {
        "facility": {"prefix": "als", "name": "ALS"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 8080,
                "web_base_port": 9000,
                "artifact_base_port": 9100,
                "ariel_base_port": 9200,
                "lattice_base_port": 9300,
                "users": ["alice"],
            }
        },
    }
    written = set(write_web_terminal_artifacts(config, repo_root=tmp_path))

    compose = yaml.safe_load(web_compose_file(tmp_path).read_text(encoding="utf-8"))
    sources = [volume.split(":", 1)[0] for volume in compose["services"]["nginx"]["volumes"]]

    assert sources, "the nginx service must mount its rendered config"
    for source in sources:
        resolved = (tmp_path / source).resolve()
        assert resolved.is_file(), f"{source} resolves to nothing under the project directory"
        assert resolved in {path.resolve() for path in written}


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
                "nginx_port": 8080,
                "web_base_port": 9000,
                "artifact_base_port": 9100,
                "ariel_base_port": 9200,
                "lattice_base_port": 9300,
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
