"""The layer cache every image build OSPREY runs takes from the environment.

``OSPREY_BUILD_CACHE_FROM`` / ``OSPREY_BUILD_CACHE_TO`` name a BuildKit cache
source and sink, with ``{key}`` standing for the image's recipe. CI sets the
first in every lane that deploys and both in main's warm-up, so PR lanes reuse
main's dependency layers instead of building them on an empty runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.deployment import container_lifecycle
from osprey.deployment.build_cache import (
    AUTH_SIDECAR_KEY,
    CACHE_FROM_ENV,
    CACHE_TO_ENV,
    PROJECT_KEY,
    build_cache_args,
    compose_cache_override,
    service_cache_key,
)
from osprey.deployment.compose_generator import compose_base_cmd
from osprey.deployment.runtime_helper import ComposeProvider
from osprey.deployment.web_terminals import provision, render_images

SOURCE = "type=registry,ref=ghcr.io/org/cache:{key}"
SINK = "type=registry,ref=ghcr.io/org/cache:{key},mode=max"

SERVICES = """\
services:
  bluesky-bridge:
    image: p-bluesky-bridge:local
    build:
      context: ./build/services/bluesky
      dockerfile: Dockerfile
  mongodb:
    image: mongo:7
"""


@pytest.fixture
def cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CACHE_FROM_ENV, SOURCE)
    monkeypatch.setenv(CACHE_TO_ENV, SINK)


@pytest.fixture(autouse=True)
def _no_host_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane that sets the variables must not reach the cases that leave them unset."""
    monkeypatch.delenv(CACHE_FROM_ENV, raising=False)
    monkeypatch.delenv(CACHE_TO_ENV, raising=False)


def _compose_file(root: Path, text: str = SERVICES) -> str:
    path = root / "build" / "services" / "docker-compose.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _f_entries(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == "-f"]


# ---------------------------------------------------------------------------
# The flags a direct build takes
# ---------------------------------------------------------------------------


def test_unset_adds_no_flags() -> None:
    assert build_cache_args("docker", PROJECT_KEY) == []


def test_both_specs_name_the_image_key() -> None:
    environ = {CACHE_FROM_ENV: SOURCE, CACHE_TO_ENV: SINK}
    assert build_cache_args("docker", PROJECT_KEY, environ) == [
        "--cache-from",
        "type=registry,ref=ghcr.io/org/cache:project",
        "--cache-to",
        "type=registry,ref=ghcr.io/org/cache:project,mode=max",
    ]


def test_a_reader_gets_the_source_alone() -> None:
    assert build_cache_args("/usr/bin/docker", AUTH_SIDECAR_KEY, {CACHE_FROM_ENV: SOURCE}) == [
        "--cache-from",
        "type=registry,ref=ghcr.io/org/cache:auth-sidecar",
    ]


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_variable_adds_nothing(value: str) -> None:
    assert build_cache_args("docker", PROJECT_KEY, {CACHE_FROM_ENV: value}) == []


def test_podman_gets_no_flags() -> None:
    """Podman's --cache-from names a repository, not a BuildKit spec."""
    assert build_cache_args("podman", PROJECT_KEY, {CACHE_FROM_ENV: SOURCE}) == []


# ---------------------------------------------------------------------------
# A compose service's key and entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "key"),
    [
        ({"context": "./build/services/bluesky", "dockerfile": "Dockerfile"}, "bluesky"),
        ({"context": "./build/services/event_dispatcher/"}, "event_dispatcher"),
        ("./build/services/qmd", "qmd"),
        # The project directory is named after the deployment: no shared key.
        ({"context": "."}, None),
        ({}, None),
        ({"context": "https://github.com/org/repo.git#main"}, None),
        (["not", "a", "build"], None),
    ],
)
def test_the_key_is_the_context_directory_name(build: object, key: str | None) -> None:
    assert service_cache_key(build) == key


def test_only_a_built_service_gets_an_entry(tmp_path: Path) -> None:
    """An entry carrying `build:` would make an image-only service buildable."""
    files = [_compose_file(tmp_path)]
    document = compose_cache_override(tmp_path, files, {CACHE_FROM_ENV: SOURCE, CACHE_TO_ENV: SINK})
    assert document == {
        "services": {
            "bluesky-bridge": {
                "build": {
                    "cache_from": ["type=registry,ref=ghcr.io/org/cache:bluesky"],
                    "cache_to": ["type=registry,ref=ghcr.io/org/cache:bluesky,mode=max"],
                }
            }
        }
    }


def test_no_variable_no_override(tmp_path: Path) -> None:
    assert compose_cache_override(tmp_path, [_compose_file(tmp_path)], {}) is None


def test_an_unreadable_file_is_left_to_compose(tmp_path: Path) -> None:
    files = [str(tmp_path / "missing.yml"), _compose_file(tmp_path)]
    document = compose_cache_override(tmp_path, files, {CACHE_FROM_ENV: SOURCE})
    assert document is not None
    assert list(document["services"]) == ["bluesky-bridge"]


# ---------------------------------------------------------------------------
# Every compose invocation carries it
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("cache_env")
def test_the_docker_argv_passes_the_cache_override_last(tmp_path: Path) -> None:
    files = [_compose_file(tmp_path)]

    argv = compose_base_cmd(["docker", "compose"], files, tmp_path, env_file_args=[])

    entries = _f_entries(argv)
    assert entries[0] == files[0]
    override = Path(entries[-1])
    assert override.parent == tmp_path / "build"
    assert override.name.startswith("osprey-build-cache.")
    document = yaml.safe_load(override.read_text(encoding="utf-8"))
    assert document["services"]["bluesky-bridge"]["build"]["cache_from"] == [
        "type=registry,ref=ghcr.io/org/cache:bluesky"
    ]


@pytest.mark.usefixtures("cache_env")
def test_two_invocations_over_one_set_write_one_file(tmp_path: Path) -> None:
    files = [_compose_file(tmp_path)]
    first = compose_base_cmd(["docker", "compose"], files, tmp_path, env_file_args=[])
    second = compose_base_cmd(["docker", "compose"], files, tmp_path, env_file_args=[])
    assert first == second


@pytest.mark.usefixtures("cache_env")
def test_podman_compose_gets_no_cache_override(tmp_path: Path) -> None:
    files = [_compose_file(tmp_path)]

    argv = compose_base_cmd(
        ["podman", "compose"],
        files,
        tmp_path,
        env_file_args=[],
        provider=ComposeProvider.PODMAN_COMPOSE,
    )

    assert not any("osprey-build-cache" in arg for arg in argv)
    assert not list((tmp_path / "build").glob("osprey-build-cache.*"))


@pytest.mark.usefixtures("cache_env")
def test_podman_through_docker_compose_gets_no_cache_override(tmp_path: Path) -> None:
    files = [_compose_file(tmp_path)]

    argv = compose_base_cmd(["podman", "compose"], files, tmp_path, env_file_args=[])

    assert _f_entries(argv) == files


def test_unset_leaves_the_argv_as_it_was(tmp_path: Path) -> None:
    files = [_compose_file(tmp_path)]

    argv = compose_base_cmd(["docker", "compose"], files, tmp_path, env_file_args=[])

    assert _f_entries(argv) == files


# ---------------------------------------------------------------------------
# The direct builds carry it, before the context
# ---------------------------------------------------------------------------


def _cache_flags(cmd: list[str]) -> list[tuple[str, str]]:
    return [(arg, cmd[i + 1]) for i, arg in enumerate(cmd) if arg in ("--cache-from", "--cache-to")]


@pytest.mark.usefixtures("cache_env")
def test_the_project_image_reads_and_writes_the_project_key() -> None:
    cmd = container_lifecycle._project_image_build_cmd({"project_name": "p"}, "docker", "/proj")

    assert _cache_flags(cmd) == [
        ("--cache-from", "type=registry,ref=ghcr.io/org/cache:project"),
        ("--cache-to", "type=registry,ref=ghcr.io/org/cache:project,mode=max"),
    ]
    assert cmd[-1] == "/proj"


@pytest.mark.usefixtures("cache_env")
def test_a_render_image_shares_the_project_key(tmp_path: Path) -> None:
    """Built from the project Dockerfile, so its dependency layers are the project image's."""
    cmd = render_images._render_image_build_cmd(
        "docker", str(tmp_path), "p-alice:local", "p", {"project_name": "p"}
    )

    assert ("--cache-from", "type=registry,ref=ghcr.io/org/cache:project") in _cache_flags(cmd)
    assert cmd[-1] == str(tmp_path)


@pytest.mark.usefixtures("cache_env")
def test_the_auth_sidecar_reads_its_own_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    context = tmp_path / "build" / "auth"
    context.mkdir(parents=True)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(
        provision,
        "_materialize_auth_build_context",
        lambda repo_root, dev_mode: (context, dev_mode),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(provision, "run_captured", lambda cmd, **kwargs: calls.append(cmd))
    monkeypatch.setattr(provision, "_report_step", lambda *args, **kwargs: None)
    config = {
        "project_name": "demo",
        "modules": {"web_terminals": {"image_source": "local", "auth": {"method": "password"}}},
    }

    provision.build_auth_sidecar_image(config, False, {})

    (cmd,) = calls
    assert ("--cache-from", "type=registry,ref=ghcr.io/org/cache:auth-sidecar") in _cache_flags(cmd)
    assert cmd[-1] == str(context)


@pytest.mark.usefixtures("cache_env")
def test_a_podman_project_build_carries_no_cache_flags() -> None:
    cmd = container_lifecycle._project_image_build_cmd({"project_name": "p"}, "podman", "/proj")

    assert _cache_flags(cmd) == []
