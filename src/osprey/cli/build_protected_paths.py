"""The paths a profile names that executed agent code may not change.

The python executor refuses writes to a fixed list of deployment-root names.
A profile can also name locations the build or a service reads — a plan
directory, a CA bundle, a registry module, a corpus — and a write there by
executed code changes what the next build or the running service trusts. This
module computes those locations from the resolved profile, as the list of
entries ``osprey build`` writes under
:data:`~osprey.utils.workspace.PROTECTED_PATHS_CONFIG_KEY`.

Three feeds make up the list, and each is exported so the decision check can
compare the profile schema and the configuration manifest against them:

1. :data:`INPUT_ENUMERATION_KEYS` — the profile keys the build's own input
   enumeration (:func:`~osprey.cli.build_profile_merge.profile_inputs`) already
   lists; its entries are emitted as that enumeration keys them, minus the
   installed applications, which are no repository content. The enumeration
   also lists the sources the deployment layout declares
   (:func:`~osprey.repo_layout.rendered_sources`), which no key names; they
   are emitted with the rest.
2. :data:`PROFILE_PATH_FIELDS` — path-valued profile fields that enumeration
   does not list.
3. :data:`CONFIG_PATH_KEYS` — path-valued configuration keys, read from a
   merged view of the profile's ``config:`` block (dotted and nested spellings
   folded) and each service block's pass-through settings, re-keyed as
   ``services.<name>.<key>``.

Each row states its reader's shape and anchor. An entry keeps the profile's
own spelling: a project-root value is emitted as stated, ``~`` and ``${VAR}``
verbatim, because the build's environment is not the run-time one. A
render-anchored value is joined to ``build/`` and emitted only when it leaves
the render, which is already protected. A working directory is emitted
relative to the project root. An entry naming the project root itself or an
ancestor of it is never emitted as a tree: a local package there contributes
its ``pyproject.toml``, anything else contributes nothing, and a note names
the key.

A protected path that lands inside the agent data zone — the one place agent
code is meant to write — is refused at build (:class:`BuildProfileError`). A
path-shaped value stated under a configuration key no table decides is
reported, so the build can warn about it; it is not protected.
"""

from __future__ import annotations

import os
import posixpath
import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from osprey.utils.workspace import anchor_protected_entry
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.workspace import BUILD_DIR_NAME, anchored_path

from .build_profile_archiver import _expand_dotted
from .build_profile_load import LoadedProfile
from .build_profile_merge import ProfileInputRole, profile_inputs

__all__ = [
    "CONFIG_PATH_KEYS",
    "EXEMPT_CONFIG_KEYS",
    "INPUT_ENUMERATION_KEYS",
    "PROFILE_PATH_FIELDS",
    "Anchor",
    "PathRow",
    "ProtectedPaths",
    "UndecidedPath",
    "emit_protected_paths",
    "is_path_shaped",
    "root_entry_names",
]


class Anchor(StrEnum):
    """Where a row's reader anchors a relative value."""

    #: The deployment root; the entry is the stated value.
    PROJECT_ROOT = "project-root"
    #: The render directory (``build/``); emitted only when it leaves the render.
    RENDER = "render"
    #: The working directory of a build verb, anchored at the project root
    #: with ``{project_root}`` substituted; emitted relative to the root.
    WORKING_DIRECTORY = "working-directory"


def _single(value: Any) -> list[str]:
    return [value] if isinstance(value, str) and value else []


def _strings(value: Any) -> list[str]:
    """A list of strings, or a bare string the reader coerces to one."""
    if isinstance(value, str):
        return _single(value)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _mapping_values(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    return [item for item in value.values() if isinstance(item, str) and item]


def _panel_files(value: Any) -> list[str]:
    """``phoebus.panels``: a name maps to a file, or to a mapping with ``file``."""
    if not isinstance(value, Mapping):
        return []
    found: list[str] = []
    for item in value.values():
        if isinstance(item, Mapping):
            found.extend(_single(item.get("file")))
        else:
            found.extend(_single(item))
    return found


def _member_strings(*members: str) -> Callable[[Any], list[str]]:
    """A list whose items are strings or mappings carrying the named members."""

    def extract(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        found: list[str] = []
        for item in value:
            if isinstance(item, str):
                found.extend(_single(item))
            elif isinstance(item, Mapping):
                for member in members:
                    found.extend(_single(item.get(member)))
        return found

    return extract


def _mapping_members(*members: str) -> Callable[[Any], list[str]]:
    """A list of mappings; only the named members are paths."""

    def extract(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            path
            for item in value
            if isinstance(item, Mapping)
            for member in members
            for path in _single(item.get(member))
        ]

    return extract


def _python_files(value: Any) -> list[str]:
    """``health.plugins``: dotted module names mixed with ``.py`` paths."""
    return [item for item in _strings(value) if item.endswith(".py")]


def _not_demo(value: Any) -> list[str]:
    """A replay file, unless it names a packaged ``demo:<name>`` replay."""
    return [item for item in _single(value) if not item.startswith("demo:")]


def _not_url(value: Any) -> list[str]:
    """A file path or a URL; only the file path is a location on this host."""
    return [item for item in _single(value) if "://" not in item]


def _interpreter(value: Any) -> list[str]:
    """``python_env``: the words ``project`` and ``build``, or an interpreter path."""
    return [item for item in _single(value) if item not in ("project", "build")]


@dataclass(frozen=True)
class PathRow:
    """One path-valued key the emitter protects.

    Attributes:
        key: The dotted key; ``<name>`` stands for any one segment.
        shape: What the reader opens: ``file``, ``dir``, ``list of files``,
            ``list of dirs``, ``mapping of files``, or a filter on those.
        anchor: Where the reader anchors a relative value.
        extract: The stated value to the path strings it names.
    """

    key: str
    shape: str
    anchor: Anchor
    extract: Callable[[Any], list[str]]


_ROOT = Anchor.PROJECT_ROOT

#: Profile keys whose locations the build's input enumeration
#: (:func:`~osprey.cli.build_profile_merge.profile_inputs`) lists. Feed 1.
INPUT_ENUMERATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "extends",
        "machine",
        "services.<name>.template",
        "dispatch.triggers",
        "dependencies",
        "hooks",
        "rules",
        "skills",
        "agents",
        "output_styles",
        "watchdogs.<name>",
        "apps.<name>",
    }
)

#: Path-valued profile fields the input enumeration does not list. Feed 2.
#: The ``deploy.host.project_path`` row protects only when the deploy host is
#: this host; elsewhere the location is one no code here can reach.
PROFILE_PATH_FIELDS: Final[tuple[PathRow, ...]] = (
    PathRow("environment.python", "file (interpreter)", _ROOT, _single),
    PathRow("python_env", "file when not a word (interpreter)", _ROOT, _interpreter),
    PathRow("lifecycle.pre_build[].cwd", "dir", Anchor.WORKING_DIRECTORY, _single),
    PathRow("bluesky.plan_dir", "dir", _ROOT, _single),
    PathRow("bluesky.external.parameter_schemas", "mapping of files", _ROOT, _mapping_values),
    PathRow("optimizer.environments_dir", "dir", _ROOT, _single),
    PathRow("deploy.host.project_path", "dir (only on this host)", _ROOT, _single),
)

#: Path-valued configuration keys. Feed 3.
CONFIG_PATH_KEYS: Final[tuple[PathRow, ...]] = (
    PathRow("services.<name>.path", "dir", Anchor.RENDER, _single),
    PathRow("services.qmd.models_dir", "dir", _ROOT, _single),
    PathRow(
        "services.qmd.corpora",
        "list of dirs (source, index_dir)",
        _ROOT,
        _mapping_members("source", "index_dir"),
    ),
    PathRow("services.graphdb.ttl_path", "file", Anchor.RENDER, _single),
    PathRow("services.bluesky.plan_dir", "dir", _ROOT, _single),
    PathRow(
        "services.bluesky.external.parameter_schema_mounts",
        "list of files (source)",
        _ROOT,
        _mapping_members("source"),
    ),
    PathRow("services.optimizer.environments_dir", "dir", _ROOT, _single),
    PathRow(
        "services.event_dispatcher.additional_dirs",
        "list of dirs (or src)",
        _ROOT,
        _member_strings("src"),
    ),
    PathRow("registry_path", "file", _ROOT, _single),
    PathRow("application.registry_path", "file", _ROOT, _single),
    PathRow("health.plugins", "list of files (only .py entries)", _ROOT, _python_files),
    PathRow("facility_knowledge.bundle_path", "dir", _ROOT, _single),
    PathRow("ariel.vocabulary.path", "file", _ROOT, _single),
    PathRow(
        "channel_directory.alarm_state.replay_file", "file (unless demo:<name>)", _ROOT, _not_demo
    ),
    PathRow("ariel.ingestion.source_url", "file (unless a URL)", _ROOT, _not_url),
    PathRow("phoebus.panels", "mapping of files (or .file)", _ROOT, _panel_files),
    PathRow("web.panels.canopy.theme_file", "file", _ROOT, _single),
    PathRow("images.site_ca", "file", _ROOT, _single),
    PathRow("archiver.epics_archiver.tls.ca_bundle", "file", _ROOT, _single),
    PathRow("archiver.mongodb_archiver.tls.ca_bundle", "file", _ROOT, _single),
    PathRow("channel_directory.settings.tls.ca_bundle", "file", _ROOT, _single),
    PathRow("channel_directory.alarm_state.settings.tls.ca_bundle", "file", _ROOT, _single),
    PathRow("ariel.ingestion.ca_bundle", "file", _ROOT, _single),
    PathRow("modules.web_terminals.tls.host_cert_dir", "dir", _ROOT, _single),
    PathRow("bluesky.plan_dirs", "list of dirs", _ROOT, _strings),
    PathRow("bluesky.plan_module", "file", _ROOT, _single),
    PathRow("ariel.enhancement_modules.qmd_export.settings.mirror_path", "dir", _ROOT, _single),
    PathRow("ariel.enhancement_modules.qmd_export.mirror_path", "dir", _ROOT, _single),
    PathRow("modules.web_terminals.landing.notices", "list of files", _ROOT, _strings),
)

#: Configuration keys decided as not protected, with the reason. A key here or
#: below one here is never reported as undecided.
EXEMPT_CONFIG_KEYS: Final[Mapping[str, str]] = {
    "project_root": "derived-not-statable",
    "build_dir": "derived-not-statable",
    "agent_data.base_dir": "derived-not-statable (the permitted root)",
    "file_paths": "derived-not-statable",
    "execution.environment.python": "derived-not-statable",
    "control_system.limits_checking.database_path": "derived-not-statable",
    "channel_finder.pipelines.in_context.database.path": "derived-not-statable",
    "channel_finder.pipelines.hierarchical.database.path": "derived-not-statable",
    "channel_finder.pipelines.hierarchical.feedback.store_path": "derived-not-statable",
    "channel_finder.pipelines.middle_layer.database.path": "derived-not-statable",
    "channel_finder.pipelines.middle_layer.database.duckdb_path": "derived-not-statable",
    "channel_finder.benchmark.dataset_path": "already-confined",
    "modules.web_terminals.renders.<name>.project_path": "derived-not-statable",
    "watchdogs": "derived-not-statable",
    "services.<name>.config_file": "derived-not-statable",
    "services.<name>.command": "not-a-path",
    "channel_directory.fleet.roster_sources": "not-a-path",
    "claude_code.servers": "not-a-path",
    "claude_code.hooks": "already-confined",
    "artifact_server.extension_types.<name>.viewer": "already-confined",
    "code_review.prompt": "already-confined",
    "screen_capture.output_dir": "runtime-output",
    "phoebus.snapshot_dir": "runtime-output",
    "phoebus.plot_dir": "runtime-output",
    "simulation.state_dir": "runtime-output",
    "modules.web_terminals.tls.cert": "other-container-filesystem",
    "modules.web_terminals.tls.key": "other-container-filesystem",
    "web.panels.<name>.path": "not-a-path (URL path)",
    "archiver.epics_archiver.retrieval_path": "not-a-path (URL path)",
    "apps.<name>.approval.<name>.state_path": "not-a-path (URL path)",
    "ariel.enhancement_modules.image_caption.prompt_template": "not-a-path (template text)",
    "ariel.enhancement_modules.semantic_processor.prompt_template": "not-a-path (template text)",
    "modules.shared_disk.host_path": "a shared data disk that is meant to be written",
    "config_dir": "derived-not-statable",
    "deployed_config_dir": "derived-not-statable",
    "deployed_build_dir": "derived-not-statable",
}


@dataclass(frozen=True)
class UndecidedPath:
    """A path-shaped value stated under a configuration key no table decides."""

    key: str
    value: str


@dataclass(frozen=True)
class ProtectedPaths:
    """What the emitter found for a deployment.

    Attributes:
        entries: The entries to write, sorted and without duplicates.
        notes: One line per key that named the project root or an ancestor of
            it, which is never protected as a tree.
        undecided: Path-shaped values under keys no table decides.
    """

    entries: tuple[str, ...]
    notes: tuple[str, ...] = ()
    undecided: tuple[UndecidedPath, ...] = ()


def is_path_shaped(value: str) -> bool:
    """Whether a stated string looks like a filesystem path.

    Starts with ``/``, ``./``, ``../`` or ``~``, or is a relative string of
    more than one segment whose last segment has a file extension. Never
    ``n/a``, a URL or a string with spaces.
    """
    if not value or " " in value or "://" in value or value.lower() == "n/a":
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    segments = value.split("/")
    return len(segments) > 1 and all(segments) and bool(PurePosixPath(segments[-1]).suffix)


def root_entry_names(entries: Iterable[str]) -> tuple[str, ...]:
    """The deployment-root entries the root-relative *entries* lie under, once each.

    The first segment of each entry anchored at the project root, in entry
    order. An absolute entry, one that begins with ``~`` or a placeholder, and
    one that climbs out of the root name no root entry.
    """
    names: dict[str, None] = {}
    for entry in entries:
        if entry.startswith(("/", "~", "$")):
            continue
        first = posixpath.normpath(entry).split("/", 1)[0]
        if first not in (".", ".."):
            names[first] = None
    return tuple(names)


def _key_matches(pattern: str, key: str) -> bool:
    """Whether *key* is *pattern* or lies below it; ``<name>`` is one segment."""
    wanted = pattern.split(".")
    parts = key.split(".")
    if len(parts) < len(wanted):
        return False
    return all(w.startswith("<") or w == p for w, p in zip(wanted, parts, strict=False))


def _lookup(tree: Mapping[str, Any], pattern: str) -> list[tuple[str, Any]]:
    """Every ``(dotted key, value)`` *pattern* addresses in a nested tree."""
    found: list[tuple[str, Any]] = [("", tree)]
    for segment in pattern.split("."):
        step: list[tuple[str, Any]] = []
        for prefix, node in found:
            if not isinstance(node, Mapping):
                continue
            names = (
                sorted(k for k in node if isinstance(k, str))
                if segment.startswith("<")
                else [segment]
            )
            step.extend(
                (f"{prefix}.{name}" if prefix else name, node[name])
                for name in names
                if name in node
            )
        found = step
    return found


def _config_views(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``config:`` block and each service block's settings, as nested trees.

    Kept apart rather than merged, so a key stated in both spellings is read
    in both.
    """
    views = [_expand_dotted(raw.get("config"))]
    services = raw.get("services")
    if isinstance(services, Mapping):
        for name, service in services.items():
            settings = service.get("config") if isinstance(service, Mapping) else None
            if isinstance(name, str) and isinstance(settings, Mapping):
                views.append({"services": {name: _expand_dotted(dict(settings))}})
    return views


def _profile_field_values(raw: Mapping[str, Any], row: PathRow) -> list[tuple[str, Any]]:
    if row.key == "lifecycle.pre_build[].cwd":
        lifecycle = raw.get("lifecycle")
        steps = lifecycle.get("pre_build") if isinstance(lifecycle, Mapping) else None
        if not isinstance(steps, list):
            return []
        return [
            (f"lifecycle.pre_build[{index}].cwd", step.get("cwd"))
            for index, step in enumerate(steps)
            if isinstance(step, Mapping)
        ]
    return _lookup(raw, row.key)


def _names_this_host(raw: Mapping[str, Any]) -> bool:
    """Whether the profile's ``deploy.host`` is the host this build runs on."""
    deploy = raw.get("deploy")
    host = deploy.get("host") if isinstance(deploy, Mapping) else None
    if not isinstance(host, Mapping):
        return False
    stated = {
        str(name).lower().split(".")[0]
        for name in (host.get("name"), host.get("fqdn"))
        if isinstance(name, str) and name
    }
    local = {socket.gethostname().lower().split(".")[0], socket.getfqdn().lower().split(".")[0]}
    return bool(stated & local)


class _Collector:
    """Gathers entries, notes and refusals for one deployment."""

    def __init__(self, deployment_root: Path, agent_data_root: Path) -> None:
        self.root = deployment_root.resolve()
        self.agent_data = anchored_path(str(agent_data_root), self.root).resolve()
        self.entries: set[str] = set()
        self.notes: list[str] = []
        self.undecided: list[UndecidedPath] = []

    def _names_root(self, entry: str) -> bool:
        if "$" in entry:
            return False
        location = anchored_path(entry, self.root).resolve()
        return location == self.root or location in self.root.parents

    def add(self, key: str, entry: str, *, package: bool = False) -> None:
        if self._names_root(entry):
            kept = (
                "only 'pyproject.toml' is protected" if package else "nothing is protected for it"
            )
            note = (
                f"{key} names the project root ('{entry}'), which is never protected "
                f"as a tree; {kept}."
            )
            if note not in self.notes:
                self.notes.append(note)
            if not package:
                return
            entry = "pyproject.toml"
        anchored = anchor_protected_entry(entry, self.root, (self.agent_data,))
        if anchored.inside_permitted is not None:
            raise BuildProfileError(
                f"Profile key '{key}' names '{entry}', which lies inside the agent data zone "
                f"'{anchored.inside_permitted}', where executed agent code may write. A path "
                f"the build or a service reads cannot live there: move it out of the zone."
            )
        self.entries.add(entry)

    def add_anchored(self, key: str, entry: str, anchor: Anchor) -> None:
        if anchor is Anchor.RENDER:
            if entry.startswith(("/", "~", "$")):
                self.add(key, entry)
                return
            joined = posixpath.normpath(posixpath.join(BUILD_DIR_NAME, entry))
            if joined == BUILD_DIR_NAME or joined.startswith(f"{BUILD_DIR_NAME}/"):
                return
            self.add(key, joined)
        elif anchor is Anchor.WORKING_DIRECTORY:
            substituted = entry.replace("{project_root}", str(self.root))
            location = Path(os.path.normpath(anchored_path(substituted, self.root)))
            try:
                relative = location.relative_to(self.root).as_posix()
            except ValueError:
                relative = location.as_posix()
            self.add(key, "." if relative == "" else relative)
        else:
            self.add(key, entry)


#: Every configuration key a table decides, protected or exempt.
_DECIDED_CONFIG_KEYS: Final[tuple[str, ...]] = (
    *(row.key for row in CONFIG_PATH_KEYS),
    *EXEMPT_CONFIG_KEYS,
)


def _decided(key: str) -> bool:
    return any(_key_matches(pattern, key) for pattern in _DECIDED_CONFIG_KEYS)


def _leaves(tree: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for name, value in tree.items():
        dotted = f"{prefix}{name}"
        if isinstance(value, Mapping) and value:
            yield from _leaves(value, f"{dotted}.")
        else:
            yield dotted, value


def _emit_one(loaded: LoadedProfile, collector: _Collector) -> None:
    document = loaded.document
    raw = document.raw
    # A bundled preset's inputs are package content, never deployment files.
    in_deployment = document.root_dir.resolve() == collector.root
    if in_deployment:
        for item in profile_inputs(document):
            if item.role is ProfileInputRole.INSTALLED:
                continue
            label = item.label or "extends"
            entry = item.key.rstrip("/") or "."
            collector.add(label, entry, package=item.role is ProfileInputRole.PACKAGE)

    for row in PROFILE_PATH_FIELDS:
        if row.key == "deploy.host.project_path" and not _names_this_host(raw):
            continue
        for key, value in _profile_field_values(raw, row):
            for entry in row.extract(value):
                collector.add_anchored(key, entry, row.anchor)

    for view in _config_views(raw):
        for row in CONFIG_PATH_KEYS:
            for key, value in _lookup(view, row.key):
                for entry in row.extract(value):
                    collector.add_anchored(key, entry, row.anchor)
        for key, value in _leaves(view):
            if _decided(key):
                continue
            for candidate in _strings(value):
                if is_path_shaped(candidate):
                    found = UndecidedPath(key, candidate)
                    if found not in collector.undecided:
                        collector.undecided.append(found)


def emit_protected_paths(
    profiles: Iterable[LoadedProfile],
    *,
    deployment_root: Path,
    agent_data_root: Path,
) -> ProtectedPaths:
    """The protected-path entries of a deployment, across all its profiles.

    Args:
        profiles: The base profile and each of its login files, each as the
            build resolved it (:attr:`LoadedProfile.document` is read).
        deployment_root: The project root entries are relative to.
        agent_data_root: The agent data zone, absolute or relative to
            *deployment_root*; no protected path may lie inside it.

    Returns:
        The union of the entries, the notes for keys naming the project root,
        and the path-shaped values under undecided keys.

    Raises:
        BuildProfileError: When a protected path lies inside the agent data
            zone; the message names the key and the path.
    """
    collector = _Collector(Path(deployment_root), Path(agent_data_root))
    for loaded in profiles:
        _emit_one(loaded, collector)
    return ProtectedPaths(
        entries=tuple(sorted(collector.entries)),
        notes=tuple(collector.notes),
        undecided=tuple(collector.undecided),
    )
