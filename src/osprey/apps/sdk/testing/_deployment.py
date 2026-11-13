"""A synthetic deployment, built for real, and an application backend served on one login's render.

:func:`synthetic_deployment` makes a deployment repository the way an operator
does (``osprey init``), puts the caller's facility tree, applications and logins
in it, and returns a :class:`Deployment`. :meth:`Deployment.build` runs a real
``osprey build`` and reads back a :class:`RenderedTree`; :meth:`Deployment.serve`
hands back a :class:`ServedTree`: the environment the build gave the
application's container, with its mounts read as the host paths they bind and
its configuration pointed at the one login's render.

Every login is a ``tiers/`` file stating its write ceiling, and every session
runs on the simulator served in process (``va``): a build renders logins, access
and writes are judged per login, and the machine an owner acts on is that
owner's record in the control tree at run time. A test that needs an owner on
another machine writes that owner's record into :attr:`ServedTree.control_tree`.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from osprey.apps.sdk.testing._build import init_repo, invoke_build
from osprey.apps.sdk.testing._facility import plain_tree, write_tree
from osprey.utils.workspace import APPS_DIRNAME, BUILD_DIR_NAME, PROFILE_FILENAME, TIERS_DIRNAME

__all__ = ["Deployment", "RenderedTree", "ServedTree", "synthetic_deployment"]

#: The logins a deployment gets when the caller names none: one, asking before a write.
DEFAULT_LOGINS: Mapping[str, str] = {"operator": "asked"}

#: A compose interpolation: ``${NAME}`` or ``${NAME:-default}``.
_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-[^}]*)?\}")

#: The variables whose container path is replaced by the login's render.
_CONFIG_ENV = ("CONFIG_FILE", "OSPREY_CONFIG")


def synthetic_deployment(
    root: Path,
    *,
    apps: Mapping[str, Path],
    facility: Mapping[str, Any] | None = None,
    logins: Mapping[str, str] = DEFAULT_LOGINS,
    profile: Mapping[str, Any] | None = None,
) -> Deployment:
    """Make a deployment repository holding a facility tree, applications and logins.

    The repository is what ``osprey init`` writes for the in-process simulator,
    with its facility replaced by *facility*, each application copied under
    ``apps/`` and listed in the profile, and one ``tiers/<login>.yml`` per login.
    Nothing is built.

    Args:
        root: The repository to create; it must not exist yet.
        apps: Each application's name and the directory holding its ``app.yml``.
        facility: The ``data/facility`` tree, in the shape
            :func:`~osprey.apps.sdk.testing.write_tree` writes; ``None`` is
            :func:`~osprey.apps.sdk.testing.plain_tree`. The measurement kinds
            built are the ones its measurement files list.
        logins: Each login's name and its ``control_system.writes`` ceiling.
        profile: Top-level profile keys merged over the profile ``osprey init``
            wrote, mappings key by key (an application's ``config``, a
            package in ``dependencies``).

    Returns:
        The deployment.

    Raises:
        AssertionError: ``osprey init`` failed; its output is the message.
        ValueError: No login is named.
    """
    if not logins:
        raise ValueError("a deployment needs at least one login")
    made = init_repo(root)
    if made.exit_code != 0:
        raise AssertionError(f"osprey init failed:\n{made.output}")

    tree = root / "data" / "facility"
    shutil.rmtree(tree)
    write_tree(tree, copy.deepcopy(dict(plain_tree() if facility is None else facility)))

    for name, source in apps.items():
        shutil.copytree(
            source,
            root / APPS_DIRNAME / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.py[co]"),
        )

    profile_path = root / PROFILE_FILENAME
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    listed: dict[str, Any] = {name: {} for name in apps}
    document["apps"] = _merged(document.get("apps") or {}, listed)
    document = _merged(document, dict(profile or {}))
    profile_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    tiers = root / TIERS_DIRNAME
    tiers.mkdir(exist_ok=True)
    for login, ceiling in logins.items():
        body = {"name": login, "config": {"control_system.writes": ceiling}}
        (tiers / f"{login}.yml").write_text(yaml.safe_dump(body, sort_keys=False), "utf-8")
    return Deployment(root, tuple(logins))


def _merged(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """*base* with *over* laid on it, mappings merged key by key, anything else replaced."""
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merged(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Deployment:
    """A synthetic deployment repository: build it, then serve a backend on one login.

    Attributes:
        root: The repository, the directory holding ``profile.yml``. A test may
            add files under it before :meth:`build`.
        logins: The logins it renders, in the order they were named.
    """

    def __init__(self, root: Path, logins: tuple[str, ...]) -> None:
        """Hold the repository *root* with its *logins*; :func:`synthetic_deployment` makes one."""
        self.root = root
        self.logins = logins
        self._rendered: RenderedTree | None = None

    def build(self) -> RenderedTree:
        """Run ``osprey build`` and read the render.

        Returns:
            The render.

        Raises:
            AssertionError: The build failed; what it printed is the message.
        """
        result = invoke_build(self.root)
        if result.exit_code != 0:
            raise AssertionError(f"osprey build failed:\n{result.output}")
        self._rendered = RenderedTree.read(self.root, result.output)
        return self._rendered

    def serve(self, login: str, container: str | None = None) -> ServedTree:
        """One application container's environment on *login*'s render, built first if need be.

        Args:
            login: The login whose render the backend is given.
            container: The application container, by its ``app.yml`` key; may be
                omitted when the deployment runs one.

        Returns:
            The served tree.
        """
        rendered = self._rendered if self._rendered is not None else self.build()
        return rendered.serve(login, container)


@dataclass(frozen=True)
class RenderedTree:
    """What one ``osprey build`` rendered.

    Attributes:
        root: The repository.
        build_dir: The render root, ``<root>/build``.
        configs: Each login's rendered ``config.yml``.
        facts: The facts document applications read
            (:func:`osprey.apps.sdk.read_facts`).
        facility: The built facility file, parsed.
        output: What the build printed.
    """

    root: Path
    build_dir: Path
    configs: Mapping[str, Path]
    facts: Mapping[str, Any]
    facility: Mapping[str, Any]
    output: str = ""

    @classmethod
    def read(cls, root: Path, output: str = "") -> RenderedTree:
        """Read the render under *root*.

        Args:
            root: The built repository.
            output: What the build printed.

        Returns:
            The render.
        """
        import json

        from osprey.apps.sdk import read_facts
        from osprey.deployment.web_terminals.render_naming import render_projection
        from osprey.facility.render import FACILITY_FILE
        from osprey_connectors.workspace import rendered_config_path

        build_dir = root / BUILD_DIR_NAME
        host = _load(rendered_config_path(root))
        renders = render_projection(root, str(host["project_name"]))
        configs = {
            login: root / entry["project_path"] / "config.yml" for login, entry in renders.items()
        }
        return cls(
            root=root,
            build_dir=build_dir,
            configs=configs,
            facts=read_facts(build_dir),
            facility=json.loads((build_dir / FACILITY_FILE).read_text(encoding="utf-8")),
            output=output,
        )

    def config(self, login: str) -> dict[str, Any]:
        """*login*'s rendered ``config.yml``, parsed."""
        return _load(self._config_path(login))

    def skills(self, login: str) -> dict[str, Path]:
        """The agent skills *login*'s render holds, applications' ones included: name to file."""
        skills = self._config_path(login).parent / ".claude" / "skills"
        return {path.parent.name: path for path in sorted(skills.glob("*/SKILL.md"))}

    def serve(self, login: str, container: str | None = None) -> ServedTree:
        """One application container's environment on *login*'s render.

        The environment is the one the build wrote into the container's compose
        service: each mounted path read as the host path it binds, each
        interpolated secret given a test value, and the configuration the
        login's render.

        Args:
            login: The login whose render the backend is given.
            container: The application container, by its ``app.yml`` key; may be
                omitted when the deployment runs one.

        Returns:
            The served tree.

        Raises:
            KeyError: The deployment has no such login, or no such application container.
        """
        from osprey.apps.manifest import launch_token_env
        from osprey.deployment.compose_generator import control_target_tree_dir
        from osprey.interfaces.web_auth import (
            ROSTER_SECRET_ENV_PREFIX,
            roster_owners,
        )

        config_path = self._config_path(login)
        key, service = self._container(container)
        mounts = _mounts(self.root, service.get("volumes") or [])
        env: dict[str, str] = {}
        for name, raw in _environment(service).items():
            env[name] = _host_value(_INTERPOLATION.sub(_test_value, raw), mounts)
        for name in _CONFIG_ENV:
            env[name] = str(config_path)

        # The terminal's own reading of the roster, on this environment's value.
        owners = roster_owners(env)
        harvested = {
            owner: env.get(ROSTER_SECRET_ENV_PREFIX + suffix, "").strip()
            for suffix, owner in owners.items()
        }
        secrets = {owner: secret for owner, secret in harvested.items() if secret}
        app = env.get("OSPREY_APP_NAME", "")
        token_env = launch_token_env(app) if app else None
        return ServedTree(
            login=login,
            container=key,
            app=app,
            config_path=config_path,
            control_tree=Path(control_target_tree_dir(self.config(login), self.root)),
            env=env,
            secrets=secrets,
            launch_token_env=token_env,
        )

    def _config_path(self, login: str) -> Path:
        if login not in self.configs:
            named = ", ".join(self.configs) or "none"
            raise KeyError(f"no login {login!r}; the logins are {named}")
        return self.configs[login]

    def _container(self, container: str | None) -> tuple[str, dict[str, Any]]:
        composes = sorted((self.build_dir / "services").glob("*/docker-compose.yml"))
        services: dict[str, dict[str, Any]] = {
            key: service
            for compose in composes
            for key, service in (_load(compose).get("services") or {}).items()
            if "OSPREY_APP_NAME" in _environment(service)
        }
        if container is None and len(services) == 1:
            return next(iter(services.items()))
        if container in services:
            return container, services[container]
        named = ", ".join(services) or "none"
        raise KeyError(f"no application container {container!r}; the containers are {named}")


@dataclass(frozen=True)
class ServedTree:
    """An application container's environment on one login's render, for an in-process backend.

    Build the backend's app inside :meth:`environment`, then call it with a
    test client and :meth:`headers`::

        served = deployment.serve("operator")
        with served.environment():
            with TestClient(create_app()) as client:
                client.get("/api/state", headers=served.headers())

    Attributes:
        login: The login whose render the backend is given.
        container: The application container's key.
        app: The application's name.
        config_path: The login's rendered ``config.yml``.
        control_tree: The control tree every owner's record is written in.
        env: The container's environment, as host paths and test secrets.
        secrets: Each owner the panel's grant admits, and the secret that names them.
        launch_token_env: The variable holding the panel's launch token, if it has one.
    """

    login: str
    container: str
    app: str
    config_path: Path
    control_tree: Path
    env: Mapping[str, str]
    secrets: Mapping[str, str] = field(default_factory=dict)
    launch_token_env: str | None = None

    @property
    def launch_token(self) -> str | None:
        """The panel's launch token, or ``None`` when the panel has none."""
        return self.env.get(self.launch_token_env) if self.launch_token_env else None

    @contextmanager
    def environment(self) -> Iterator[ServedTree]:
        """Run the block in the container's environment; the process's own is put back after.

        The process's web credentials are read afresh on entry and forgotten on
        exit, so the backend's gate admits this environment's secrets, and its
        unqualified configuration lookups answer from the login's render.
        """
        from osprey.interfaces.web_auth import reset_web_credentials
        from osprey_connectors.config import config_anchored_at

        with _environ(self.env):
            reset_web_credentials()
            try:
                with config_anchored_at(self.config_path):
                    yield self
            finally:
                reset_web_credentials()

    def headers(self, owner: str | None = None) -> dict[str, str]:
        """The operator-secret header naming *owner*, as the terminal's proxy sends it.

        Args:
            owner: An owner the panel's grant admits; may be omitted when it admits one.

        Raises:
            KeyError: The grant admits no such owner.
        """
        from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER

        if owner is None and len(self.secrets) == 1:
            owner = next(iter(self.secrets))
        if owner is None or owner not in self.secrets:
            named = ", ".join(self.secrets) or "none"
            raise KeyError(f"the panel's grant admits no owner {owner!r}; it admits {named}")
        return {OPERATOR_SECRET_HEADER: self.secrets[owner]}

    def record_path(self, owner: str) -> Path:
        """Where *owner*'s control-context record lives in :attr:`control_tree`."""
        from osprey_connectors.control_context import RECORD_FILENAME

        return self.control_tree / owner / RECORD_FILENAME


@contextmanager
def _environ(values: Mapping[str, str | None]) -> Iterator[None]:
    """Set each variable for the block (``None`` unsets it); the process's own are put back."""
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            _put(name, value)
        yield
    finally:
        for name, value in saved.items():
            _put(name, value)


def _put(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else {}


def _environment(service: Mapping[str, Any]) -> dict[str, str]:
    """A compose service's ``environment``, mapping or ``KEY=VALUE`` list, as strings."""
    raw = service.get("environment") or {}
    if isinstance(raw, list):
        raw = dict(entry.split("=", 1) for entry in raw if "=" in entry)
    return {str(name): "" if value is None else str(value) for name, value in raw.items()}


def _test_value(match: re.Match[str]) -> str:
    """The value a test gives an interpolated variable: fixed, and named for it."""
    return f"test-{match.group(1).lower().replace('_', '-')}"


def _mounts(root: Path, volumes: list[Any]) -> list[tuple[str, Path]]:
    """Each bind mount's container path and the host path it binds, longest first."""
    mounts = []
    for volume in volumes:
        if not isinstance(volume, str) or ":" not in volume:
            continue
        source, target = volume.split(":")[:2]
        if source.startswith((".", "/")):
            mounts.append((target.rstrip("/"), (root / source).resolve()))
    return sorted(mounts, key=lambda mount: len(mount[0]), reverse=True)


def _host_value(value: str, mounts: list[tuple[str, Path]]) -> str:
    """*value* with each colon-separated container path read as the host path that backs it."""
    if not value.startswith("/"):
        return value
    return ":".join(_host_path(part, mounts) for part in value.split(":"))


def _host_path(path: str, mounts: list[tuple[str, Path]]) -> str:
    for target, source in mounts:
        if path == target:
            return str(source)
        if path.startswith(target + "/"):
            return str(source / path[len(target) + 1 :])
    return path
