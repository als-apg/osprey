"""Reachability of the paths one deployment's writers and readers must share.

A deployment repo has two path anchors, and they disagree. The compose layer
anchors every relative bind source on the repo root — every invocation is
pinned with ``--project-directory <repo>`` — while the runtime resolvers
(:func:`osprey.utils.config_paths.resolve_config_relative_path` and everything
delegating to it) anchor the same configured strings on the directory holding
``config.yml``, which after a build is ``<repo>/build``. Nothing fails loudly
when the two disagree: the exporter writes a mirror the sidecar never mounts,
a persona's readers open a baked copy while the host bundle is mounted one
directory over, and an audit log lands in a container's writable layer.

So every test here asserts a REACH property on one real built stack: the path
one component writes is the path the other component reads, with both sides
derived through the production functions rather than restated. Each test
fails today, each for a real defect, and each docstring names the existing
test that came closest and why it could not see the gap.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
import yaml

from osprey.deployment.compose_generator import _resolve_qmd_corpora
from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
from osprey.deployment.web_terminals.personas import resolve_personas
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.ariel_search.enhancement.qmd_export.exporter import resolve_mirror_path
from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path
from osprey.utils.workspace import AUDIT_DIR_RELPATH
from tests.cli.test_persona_presets import _build_persona_stack

#: Every test here builds the hosting preset for real — seconds, not
#: milliseconds — because reach is a property of what a build actually writes.
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _mounts(service: dict) -> list[tuple[str, str]]:
    """``(source, target)`` pairs from a compose service's volume entries.

    Normalizes both spellings compose accepts: ``"src:dst[:mode]"`` strings and
    long-form mappings. A long-form entry with no source (a tmpfs) contributes
    an empty source, which matches nothing a test here looks for.
    """
    pairs: list[tuple[str, str]] = []
    for entry in service.get("volumes", []):
        if isinstance(entry, str):
            parts = entry.split(":")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
        elif isinstance(entry, dict):
            pairs.append((str(entry.get("source", "")), str(entry.get("target", ""))))
    return pairs


def _mount_targets(service: dict) -> list[str]:
    """The in-container target of every volume entry on *service*."""
    return [target for _source, target in _mounts(service)]


def _host_path(repo: Path, source: str) -> Path:
    """A compose bind source as the pinned project directory resolves it.

    Every compose invocation is pinned with ``--project-directory <repo>``
    (see :func:`osprey.deployment.compose_generator.compose_base_cmd`), so a
    relative source means ``<repo>/<source>`` and an absolute one means itself.
    """
    path = Path(source)
    if path.is_absolute():
        return path
    return (repo / path).resolve()


def _qmd_export_mirror(config: dict) -> str | None:
    """The mirror path an enabled qmd export writes to, ``settings`` winning.

    The same merge rule ARIEL's loader and ``_resolve_qmd_corpora`` apply: a
    ``settings.mirror_path`` overrides one written directly on the module
    block. ``None`` when the export is absent or disabled.
    """
    ariel = config.get("ariel") or {}
    export = (ariel.get("enhancement_modules") or {}).get("qmd_export") or {}
    if not export.get("enabled"):
        return None
    settings = export.get("settings") or {}
    return settings.get("mirror_path") or export.get("mirror_path")


@pytest.fixture(scope="module")
def built_stack(tmp_path_factory) -> Path:
    """The hosting preset built once, personas included."""
    return _build_persona_stack(tmp_path_factory.mktemp("reach-paths") / "my-facility")


@pytest.fixture(scope="module")
def host_config(built_stack: Path) -> dict:
    """The deployment's own render — what every host-side process runs against."""
    return _load(built_stack / "build" / "config.yml")


@pytest.fixture(scope="module")
def resolved_entries(host_config: dict) -> list[dict]:
    """The roster as render/build resolve it, one entry per per-user service."""
    return resolve_personas(
        host_config["modules"]["web_terminals"],
        host_config.get("registry") or {},
        (host_config.get("facility") or {}).get("prefix") or "",
        strict=True,
    )


@pytest.fixture(scope="module")
def web_compose(built_stack: Path, host_config: dict) -> dict:
    """The multi-user overlay, rendered with exactly the disk-derived inputs the
    deploy path resolves (``artifacts.resolve_render_inputs`` — the one seam
    ``write_web_terminal_artifacts`` hands the render)."""
    artifacts = render_web_terminals(host_config, **resolve_render_inputs(host_config, built_stack))
    return yaml.safe_load(artifacts["docker-compose.web.yml"])


def _persona_config(repo: Path, entry: dict) -> dict:
    """A persona's own rendered ``config.yml`` — what its container's readers load."""
    return _load(repo / "build" / entry["project"] / "config.yml")


# ---------------------------------------------------------------------------
# T1: the host exporter and the qmd sidecar
# ---------------------------------------------------------------------------


def test_host_exporter_writes_where_the_sidecar_reads(built_stack, host_config):
    """The directory the qmd exporter writes must be the directory the sidecar
    bind-mounts, or the sidecar indexes a tree nobody writes to.

    ``tests/deployment/test_qmd_compose_fragment.py::
    test_one_read_only_mount_per_configured_corpus`` came closest: it pins the
    bind SOURCE string the fragment renders, but nothing compares that string
    with where the exporter's own resolver (:func:`resolve_mirror_path`,
    anchored on the config's directory) actually WRITES — so the two anchors
    can disagree while both sides' tests stay green.
    """
    repo = built_stack
    corpora = {c["collection"]: c for c in _resolve_qmd_corpora(host_config, str(repo))}

    raw_mirror = _qmd_export_mirror(host_config)
    assert raw_mirror, "fixture lost its enabled qmd_export — nothing to compare"
    assert "ariel" in corpora, "fixture lost the sidecar's ARIEL corpus — nothing to compare"
    exporter_writes = resolve_mirror_path(raw_mirror, config_dir=repo / "build")
    sidecar_reads = _host_path(repo, corpora["ariel"]["source"])

    assert exporter_writes == sidecar_reads, (
        f"the exporter resolves mirror_path {raw_mirror!r} against the config dir and writes "
        f"{exporter_writes}, but the sidecar bind-mounts {sidecar_reads} (the same string "
        f"anchored on the compose project directory, the repo root) — the sidecar indexes an "
        f"empty directory while the mirror grows inside build/"
    )

    raw_bundle = host_config["facility_knowledge"]["bundle_path"]
    reader_opens = resolve_bundle_path(raw_bundle, config_dir=repo / "build")
    assert "okf" in corpora, "fixture lost the sidecar's OKF corpus — nothing to compare"
    sidecar_bundle = _host_path(repo, corpora["okf"]["source"])

    assert reader_opens == sidecar_bundle, (
        f"the OKF readers resolve bundle_path {raw_bundle!r} against the config dir and open "
        f"{reader_opens}, but the sidecar bind-mounts {sidecar_bundle} — the corpus the "
        f"readers serve is not the corpus the sidecar indexes"
    )


# ---------------------------------------------------------------------------
# T2: a persona's bundle mount and its readers
# ---------------------------------------------------------------------------


def test_persona_bundle_mount_lands_where_its_readers_look(
    built_stack, web_compose, resolved_entries
):
    """A persona service's bundle mount target must be the path that persona's
    own readers resolve from its rendered config, or the mount shadows nothing
    and the readers open the image's stale baked copy.

    ``tests/deployment/web_terminals/test_bundle_mount.py::
    test_target_is_computed_per_persona_from_its_own_project_dir`` came
    closest: it asserts the target equals the render's OWN derivation
    (``/app/<project>/<bundle_path>``), never comparing it with where the
    in-container readers — anchored on the config's directory,
    ``/app/<project>/build`` — resolve the same key.
    """
    checked = 0
    for entry in resolved_entries:
        service = web_compose["services"][f"web-{entry['name']}"]
        persona_config = _persona_config(built_stack, entry)
        raw = (persona_config.get("facility_knowledge") or {}).get("bundle_path")
        bundle_mounts = [
            (source, target)
            for source, target in _mounts(service)
            if raw and target.endswith(f"/{PurePosixPath(str(raw).strip())}")
        ]
        if not bundle_mounts:
            continue
        checked += 1
        reader_opens = resolve_bundle_path(
            str(raw).strip(), config_dir=Path(entry["container_project_dir"]) / "build"
        ).as_posix()
        for _source, target in bundle_mounts:
            assert target == reader_opens, (
                f"persona {entry['persona']!r} mounts the host bundle at {target}, but its "
                f"OKF panel and facility_knowledge MCP server resolve bundle_path {raw!r} "
                f"against the config dir and read {reader_opens} — the baked copy, not the "
                f"live mount"
            )
    assert checked, "no per-user service carried a bundle mount — fixture lost the precondition"


# ---------------------------------------------------------------------------
# T3: a persona's exporter and the host mirror
# ---------------------------------------------------------------------------


def test_persona_mirror_is_the_hosts_mirror(
    built_stack, host_config, web_compose, resolved_entries
):
    """A persona whose render enables ``qmd_export`` must bind-mount the HOST
    mirror directory at the path its in-container exporter writes, or entries
    enhanced inside that container land in its writable layer — unindexed by
    the sidecar and discarded on the next recreate.

    ``tests/deployment/test_qmd_compose_fragment.py::
    test_one_read_only_mount_per_configured_corpus`` covers the sidecar's own
    mount, and ``tests/deployment/web_terminals/test_bundle_mount.py`` covers
    the bundle grant per persona — but no test asks whether the per-user
    services mount the MIRROR at all, so the compose template's silence about
    it (claude-config volume, agent-data volume, optional bundle mount,
    ``extra_mounts`` — no mirror) was never visible.
    """
    repo = built_stack
    corpora = {c["collection"]: c for c in _resolve_qmd_corpora(host_config, str(repo))}
    assert "ariel" in corpora, "fixture lost the sidecar's ARIEL corpus — nothing to compare"
    host_mirror = _host_path(repo, corpora["ariel"]["source"])

    checked = 0
    for entry in resolved_entries:
        persona_config = _persona_config(repo, entry)
        raw = _qmd_export_mirror(persona_config)
        if raw is None:
            continue
        checked += 1
        exporter_writes = resolve_mirror_path(
            raw, config_dir=Path(entry["container_project_dir"]) / "build"
        ).as_posix()
        service = web_compose["services"][f"web-{entry['name']}"]
        matching = [
            (source, target)
            for source, target in _mounts(service)
            if _host_path(repo, source) == host_mirror and target == exporter_writes
        ]
        assert matching, (
            f"persona {entry['persona']!r} enables qmd_export, but its service mounts the "
            f"host mirror {host_mirror} nowhere — its exporter writes {exporter_writes} in "
            f"the container's writable layer, a tree the sidecar never indexes and the next "
            f"recreate discards (volumes: {_mount_targets(service)})"
        )
    assert checked, "no persona render enables qmd_export — fixture lost the precondition"


# ---------------------------------------------------------------------------
# T4: the audit records
# ---------------------------------------------------------------------------


def test_persona_audit_records_are_backed_by_a_volume(web_compose, resolved_entries):
    """Every per-user container's audit subdirectory
    (``<project>/var/audit/<user>``) must be covered by a volume or bind mount,
    or every record the unified writer files there — refused writes, tool
    calls, hook decisions — is silently discarded on the next recreate.

    ``tests/deployment/test_audit_mounts.py`` pins the render's derivation of
    the target; this asks the question from the writer's side: the directory
    :func:`osprey.audit.writer.audit_dir` resolves for this identity, inside
    the compose service the persona actually runs in, is backed by some mount
    — the agent-data volume beside it (``var/agent_data``) is a sibling, not
    an ancestor.
    """
    for entry in resolved_entries:
        service = web_compose["services"][f"web-{entry['name']}"]
        audit_dir = (
            PurePosixPath(entry["container_project_dir"]) / AUDIT_DIR_RELPATH / entry["name"]
        )
        targets = [PurePosixPath(target) for target in _mount_targets(service)]
        covered = any(target == audit_dir or target in audit_dir.parents for target in targets)
        assert covered, (
            f"web-{entry['name']} writes its audit records under {audit_dir}, but no "
            f"volume or bind mount covers that path or an ancestor of it (mount targets: "
            f"{[str(t) for t in targets]}) — the log lands in the container's writable "
            f"layer and are discarded on the next recreate"
        )
