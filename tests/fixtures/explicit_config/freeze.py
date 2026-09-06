#!/usr/bin/env python3
"""Freeze today's rendered ``build/config.yml`` documents as comparison fixtures.

The explicit-profile-config work moves the declarative defaults that live in
``src/osprey/templates/apps/*/config.yml.j2`` into the presets, and rewrites the
framework template to carry only derived keys. Nothing about the *rendered*
config is supposed to change. This script captures what the render produces
before any of that lands, so the render-equivalence test has something to
compare against and the app-template deletion has a gate.

What it does, per cell
----------------------
A **cell** is one ``(preset, channel_finder_mode)`` pair. For each cell the
script runs, in a scratch directory::

    python -m osprey init <PROJECT_NAME> --preset <preset> --no-git \\
        [--set channel_finder_mode=<mode>]
    python -m osprey build

and stores ``yaml.safe_load`` of every ``config.yml`` the build emits, one file
per rendered project::

    tests/fixtures/explicit_config/<preset>/<mode>/root.yml
    tests/fixtures/explicit_config/<preset>/<mode>/<persona>.yml

``root.yml`` is ``build/config.yml``; each ``<persona>.yml`` is
``build/<project>-<persona>/config.yml``.

Completeness
------------
A build writes ``config.yml`` in three more places: ``build/.image/*/build/``
(the container copies), and ``build/services/*/`` (one per deployed service).
Every one of them is a re-serialization of the root or of a persona render, so
capturing root + personas captures the whole surface. That is not assumed — the
script re-reads every ``config.yml`` under ``build/`` and refuses to write a
cell whose extra documents do not each equal a captured one. If a future render
grows a document that is genuinely its own, the freeze fails loudly with the
path rather than quietly producing an incomplete fixture.

Masked and stripped keys
------------------------
Whole documents are compared downstream, so only what is inherently
machine-, path- or catalog-specific is touched. Every one of these is also
listed under ``masked_keys`` in ``meta.json``:

``api.providers``
    Replaced by the sorted list of provider names. The provider catalog is
    moving to a packaged ``providers.yml`` and its *contents* are that task's
    business; which providers reach the render is this fixture's business.
``project_root``
    Deleted. The absolute path of the scratch directory the freeze ran in.
    Verified to be the only machine-specific value in the render: the script
    re-scans every masked document for the scratch root and for the invoking
    user's home directory and fails if either survives.
``execution.environment``
    Deleted (the whole mapping: ``python``, ``packages``, ``inherit_exclude``).
    ``execution.environment.python`` resolves to an interpreter path on a
    machine that has one, so the block cannot be frozen. ``execution_method``
    is kept.
``container_runtime``
    Deleted. The presets ship ``auto``, and ``osprey build`` answers ``auto``
    with the runtime that actually served the build — ``docker`` on a machine
    that has one, and ``auto`` left standing on a machine that has none. So the
    rendered value states a fact about the building host, not about the
    deployment, and cannot be frozen.

The CLI runs with every ``*_API_KEY`` variable removed from the environment, so
which provider keys happen to be exported on the freezing machine cannot reach
the fixtures.

Provenance: the baseline, not the working tree
----------------------------------------------
These fixtures capture the render at one commit, and the script renders at that
commit rather than at whatever is checked out. It exports the baseline itself::

    python tests/fixtures/explicit_config/freeze.py          # --baseline defaults

``git archive <sha> | tar -x`` into a temp directory, whose ``src`` trees go on
``PYTHONPATH`` for the CLI subprocesses — ahead of the editable install, whose
``.pth`` file is only processed with ``site-packages``. The export writes no git
state: no checkout, no stash, no index. So the freeze is correct even while other
work is uncommitted in the same worktree, which is the situation it was written
for.

Rendering at the baseline is the default rather than a flag because the working
tree stops being able to reproduce these renders the moment the feature lands:
the app templates they come from are deleted. The sha is the provenance, and it
is recorded in ``meta.json`` as ``baseline_sha`` alongside the full commit id.
``--osprey-src`` skips the export for somebody who already has one.

Determinism
-----------
Documents are written with ``yaml.safe_dump(sort_keys=True)``, the project name
is a constant, and each cell builds in a fresh scratch directory, so a re-run at
the same baseline is a no-op in git — verified by running two independent
freezes of the same cells and diffing them. Regenerating the fixtures is an
explicit decision, never a side effect of another change.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

#: Where the frozen documents, ``cells.json`` and ``meta.json`` live.
FIXTURE_ROOT = Path(__file__).resolve().parent

#: The repository this file sits in; the ``git archive`` below is taken from it.
REPO_ROOT = FIXTURE_ROOT.parents[2]

#: The commit these fixtures capture: the last one before the feature's own
#: changes. The default rather than a flag, because after this feature lands the
#: checked-out tree can never reproduce these renders — the app templates are
#: deleted — so the sha, not the working tree, is the provenance.
DEFAULT_BASELINE = "246198f1c"

#: Source directories of the baseline export, relative to its root, in the order
#: they go on ``PYTHONPATH``. The connectors package is a second editable install
#: in this repo, so a faithful baseline needs both.
BASELINE_SOURCE_DIRS: tuple[str, ...] = ("src", "packages/osprey-connectors/src")

#: The presets whose renders are frozen. The four the feature converts.
PRESETS: tuple[str, ...] = (
    "hello-world",
    "control-assistant",
    "ariel-standalone",
    "channel-finder-standalone",
)

#: Deployment name used for every cell. A constant, because the repo directory
#: name becomes ``project_name`` and reaches the render in container names and
#: paths — a per-cell name would make the fixtures depend on the scratch layout.
PROJECT_NAME = "osprey-freeze"

#: Directory name for a cell of a preset that runs no channel finder, and so
#: admits no ``channel_finder_mode``. Not a mode name: no ``--set`` is passed.
UNSET_MODE_DIR = "unset"

#: Agent whose presence in a preset makes ``channel_finder_mode`` meaningful.
#: ``osprey build`` requires the mode exactly when this agent is selected
#: (see ``templates/manager.py``), and ignores it otherwise.
CHANNEL_FINDER_AGENT = "channel-finder"

#: Top-level keys deleted from every captured document, with the reason each is
#: unfreezable. Mirrored into ``meta.json``; see the module docstring.
STRIPPED_KEYS: Mapping[str, str] = {
    "project_root": "absolute path of the directory the freeze built in",
    "execution.environment": (
        "execution.environment.python resolves to an interpreter path on a machine that has one"
    ),
    "container_runtime": (
        "osprey build answers the preset's container_runtime: auto with the runtime that "
        "served the build, which is a fact about the building host"
    ),
}

#: Keys replaced rather than deleted, with what replaces them.
REPLACED_KEYS: Mapping[str, str] = {
    "api.providers": "sorted list of provider names, in place of the catalog mapping",
}

#: Build-tree directories that never hold a persona render.
_NON_PERSONA_BUILD_DIRS = frozenset({"services", "data", "docker", "_mcp_servers"})


# ─────────────────────────────────────────────────────────────────────────────
# Cell selection
# ─────────────────────────────────────────────────────────────────────────────


def _packaged_preset(preset: str) -> dict[str, Any]:
    """Load a bundled preset document by name.

    Args:
        preset: Preset name as ``osprey init --preset`` spells it.

    Returns:
        The parsed preset YAML.
    """
    from osprey.profiles import presets as presets_pkg

    path = Path(presets_pkg.__file__).resolve().parent / f"{preset}.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def modes_for_preset(preset: str) -> list[str | None]:
    """The ``channel_finder_mode`` values a preset admits.

    A preset that does not select the channel-finder agent runs no channel
    finder, so the mode has nothing to configure and the build neither requires
    nor reads one: its single cell is the unset one. A preset that does select
    it is offered every member of ``VALID_CHANNEL_FINDER_MODES``; which of those
    it actually accepts is not decided here but by running ``osprey init``, so a
    mode the profile validator refuses (``graph`` without a graph store, say) is
    recorded as a refused cell with the refusal text rather than guessed at.

    Args:
        preset: Preset name as ``osprey init --preset`` spells it.

    Returns:
        Either ``[None]`` or the full list of valid mode names.
    """
    from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES

    agents = _packaged_preset(preset).get("agents") or []
    if CHANNEL_FINDER_AGENT not in agents:
        return [None]
    return list(VALID_CHANNEL_FINDER_MODES)


# ─────────────────────────────────────────────────────────────────────────────
# Masking
# ─────────────────────────────────────────────────────────────────────────────


def mask_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Strip and replace the parts of a rendered config that cannot be frozen.

    Args:
        document: A ``yaml.safe_load`` of one rendered ``config.yml``.

    Returns:
        A copy carrying every other key untouched; see the module docstring for
        what is removed and why. The input is not modified.
    """
    masked = dict(document)
    masked.pop("project_root", None)
    masked.pop("container_runtime", None)

    api = masked.get("api")
    if isinstance(api, Mapping) and isinstance(api.get("providers"), Mapping):
        api = dict(api)
        api["providers"] = sorted(api["providers"])
        masked["api"] = api

    execution = masked.get("execution")
    if isinstance(execution, Mapping) and "environment" in execution:
        execution = dict(execution)
        execution.pop("environment")
        masked["execution"] = execution

    return masked


def _leaf_strings(node: Any) -> Iterable[str]:
    """Every scalar in a document, rendered as text.

    Args:
        node: Any node of a parsed YAML document.

    Yields:
        ``str(value)`` for each leaf reached.
    """
    if isinstance(node, Mapping):
        for value in node.values():
            yield from _leaf_strings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _leaf_strings(value)
    else:
        yield str(node)


def assert_no_machine_paths(document: Mapping[str, Any], scratch: Path, label: str) -> None:
    """Refuse a masked document that still names this machine.

    ``project_root`` is the only path-bearing key the render is known to carry,
    and this is the check that keeps that true: a render that grows a second one
    fails the freeze here instead of baking a scratch directory into a fixture
    that then only ever matches on the machine that wrote it.

    Args:
        document: A masked document.
        scratch: The directory the cell built in.
        label: Document name, for the error message.

    Raises:
        RuntimeError: If any scalar names the scratch tree or the user's home.
    """
    # Both the given and the resolved spelling of each root: on macOS the
    # per-user temp directory reaches the build as `/var/...` and comes back
    # from the render as `/private/var/...`, so checking one spelling would
    # miss a path the other names.
    needles = {
        str(scratch),
        str(scratch.resolve()),
        str(Path.home()),
        str(Path.home().resolve()),
    }
    for text in _leaf_strings(document):
        for needle in needles:
            if needle and needle in text:
                raise RuntimeError(
                    f"{label}: a value still names this machine ({needle!r} in "
                    f"{text[:120]!r}). Add the key to the mask in freeze.py, or "
                    f"the fixture will only ever match on the machine that wrote it."
                )


# ─────────────────────────────────────────────────────────────────────────────
# The baseline export
# ─────────────────────────────────────────────────────────────────────────────


def export_baseline(sha: str, destination: Path) -> list[Path]:
    """Extract the repository at *sha* and return its source directories.

    ``git archive`` writes a tree to stdout and touches no git state — no
    checkout, no stash, no index. It is therefore safe to run from a worktree
    with uncommitted changes, which is the whole point: the fixtures must
    capture the render as of *sha*, not as of whatever the working tree
    currently holds.

    Args:
        sha: Commit-ish to export.
        destination: An existing empty directory to extract into.

    Returns:
        The directories to put on ``PYTHONPATH``, in order. Both editable
        installs in this repo are covered when the export carries them.

    Raises:
        RuntimeError: If the export or the extraction fails, or if the extracted
            tree carries no ``src`` directory.
    """
    archive = subprocess.run(
        ["git", "archive", sha], cwd=REPO_ROOT, capture_output=True, check=False
    )
    if archive.returncode != 0:
        raise RuntimeError(
            f"git archive {sha} failed: {archive.stderr.decode('utf-8', 'replace').strip()}"
        )
    extract = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive.stdout,
        capture_output=True,
        check=False,
    )
    if extract.returncode != 0:
        raise RuntimeError(
            f"extracting the {sha} archive failed: "
            f"{extract.stderr.decode('utf-8', 'replace').strip()}"
        )
    sources = [destination / relative for relative in BASELINE_SOURCE_DIRS]
    present = [path for path in sources if path.is_dir()]
    if not present or not (destination / "src" / "osprey").is_dir():
        raise RuntimeError(f"the {sha} export carries no src/osprey to import")
    return present


# ─────────────────────────────────────────────────────────────────────────────
# Running the CLI
# ─────────────────────────────────────────────────────────────────────────────


def _cli_env(source_dirs: list[Path]) -> dict[str, str]:
    """Environment for the ``osprey`` subprocesses.

    Every ``*_API_KEY`` is dropped so which provider credentials happen to be
    exported cannot reach a fixture, and *source_dirs* is prepended to
    ``PYTHONPATH`` so a pristine source tree shadows the editable install — the
    ``.pth`` file an editable install drops appends its directory when
    ``site-packages`` is processed, which is after ``PYTHONPATH``.

    Args:
        source_dirs: Directories to import OSPREY from, highest priority first.

    Returns:
        The environment mapping to hand to :func:`subprocess.run`.
    """
    env = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
    entries = [str(path) for path in source_dirs]
    if entries:
        existing = env.get("PYTHONPATH")
        if existing:
            entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _run_cli(
    args: list[str], cwd: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run one ``osprey`` command.

    Invoked as ``[sys.executable, "-m", "osprey", ...]`` rather than as a bare
    ``osprey``, which is this repo's rule: PATH may resolve to a different
    install whose presets diverge from the running interpreter's.

    Args:
        args: Arguments after ``osprey``.
        cwd: Working directory for the command.
        env: Environment to run under.

    Returns:
        The completed process, output captured, non-zero exit not raised.
    """
    return subprocess.run(
        [sys.executable, "-m", "osprey", *args],
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _failure_reason(result: subprocess.CompletedProcess[str]) -> str:
    """A one-paragraph reason from a failed CLI run.

    Args:
        result: The completed process.

    Returns:
        The tail of its output, trimmed to something a reader of ``cells.json``
        can act on.
    """
    text = (result.stderr or "").strip() or (result.stdout or "").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return " ".join(lines[-6:])[:900] or f"exited {result.returncode} with no output"


# ─────────────────────────────────────────────────────────────────────────────
# Collecting a build tree
# ─────────────────────────────────────────────────────────────────────────────


def collect_rendered_configs(project: Path) -> dict[str, dict[str, Any]]:
    """The documents a build emitted, keyed by the render they belong to.

    Args:
        project: The deployment repo, containing ``build/``.

    Returns:
        ``{"root": …}`` plus one entry per persona, keyed by the persona name
        (the render directory with the ``<project_name>-`` prefix removed).

    Raises:
        RuntimeError: If ``build/config.yml`` is absent.
    """
    build = project / "build"
    root_path = build / "config.yml"
    if not root_path.is_file():
        raise RuntimeError(f"no build/config.yml under {project}")

    captured = {"root": yaml.safe_load(root_path.read_text(encoding="utf-8"))}
    for candidate in sorted(build.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        if candidate.name in _NON_PERSONA_BUILD_DIRS:
            continue
        config = candidate / "config.yml"
        if not config.is_file():
            continue
        persona = candidate.name
        prefix = f"{project.name}-"
        if persona.startswith(prefix):
            persona = persona[len(prefix) :]
        captured[persona] = yaml.safe_load(config.read_text(encoding="utf-8"))
    return captured


def verify_capture_is_complete(project: Path, masked: Mapping[str, dict[str, Any]]) -> list[str]:
    """Check that every ``config.yml`` in the build tree is one of the captured ones.

    The container copies under ``build/.image/`` and the per-service copies under
    ``build/services/`` are re-serializations of the root or of a persona render.
    This asserts it rather than trusting it, so a render that grows a genuinely
    distinct document is caught here instead of leaving a hole in the fixture.

    Args:
        project: The deployment repo.
        masked: The masked captured documents, keyed by render name.

    Returns:
        The build-relative paths that were checked and matched.

    Raises:
        RuntimeError: If some ``config.yml`` matches no captured document.
    """
    build = project / "build"
    checked: list[str] = []
    unmatched: list[str] = []
    for path in sorted(build.rglob("config.yml")):
        document = mask_document(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        relative = str(path.relative_to(build))
        if any(document == captured for captured in masked.values()):
            checked.append(relative)
        else:
            unmatched.append(relative)
    if unmatched:
        raise RuntimeError(
            "build/ holds config.yml documents that match no captured render: "
            f"{unmatched}. Capture them too, or the fixture is incomplete."
        )
    return checked


# ─────────────────────────────────────────────────────────────────────────────
# Freezing
# ─────────────────────────────────────────────────────────────────────────────


def freeze_cell(
    preset: str,
    mode: str | None,
    scratch: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Build one cell and write its documents under the fixture root.

    Args:
        preset: Preset name.
        mode: ``channel_finder_mode`` to set, or ``None`` to set none.
        scratch: An empty directory to build in.
        env: Environment for the CLI subprocesses.

    Returns:
        The cell's record for ``cells.json``: ``preset``, ``mode``, ``status``
        (``frozen``/``refused``/``failed``), and either the persona names and
        checked paths or the reason it was skipped.
    """
    mode_dir = mode or UNSET_MODE_DIR
    record: dict[str, Any] = {"preset": preset, "mode": mode, "directory": f"{preset}/{mode_dir}"}

    init_args = ["init", PROJECT_NAME, "--preset", preset, "--no-git"]
    if mode is not None:
        init_args += ["--set", f"channel_finder_mode={mode}"]

    result = _run_cli(init_args, cwd=scratch, env=env)
    if result.returncode != 0:
        record.update(status="refused", stage="init", reason=_failure_reason(result))
        return record

    project = scratch / PROJECT_NAME
    result = _run_cli(["build"], cwd=project, env=env)
    if result.returncode != 0:
        record.update(status="failed", stage="build", reason=_failure_reason(result))
        return record

    captured = collect_rendered_configs(project)
    masked = {name: mask_document(document or {}) for name, document in captured.items()}
    for name, document in masked.items():
        assert_no_machine_paths(document, scratch, f"{preset}/{mode_dir}/{name}")
    checked = verify_capture_is_complete(project, masked)

    out_dir = FIXTURE_ROOT / preset / mode_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name, document in sorted(masked.items()):
        (out_dir / f"{name}.yml").write_text(
            yaml.safe_dump(document, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )

    personas = sorted(name for name in masked if name != "root")
    record.update(
        status="frozen",
        personas=personas,
        documents=sorted(f"{name}.yml" for name in masked),
        build_configs_checked=checked,
    )
    return record


def _resolved_baseline(sha: str) -> str | None:
    """The full commit id *sha* names, for the record in ``meta.json``.

    Args:
        sha: The abbreviated commit-ish the freeze exported.

    Returns:
        The 40-character commit id, or ``None`` when this repository cannot
        resolve it (an export handed over by hand, a shallow clone).
    """
    resolved = subprocess.run(
        ["git", "rev-parse", f"{sha}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return resolved.stdout.strip() if resolved.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    """Regenerate every cell under the fixture root.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status: non-zero when a cell that should have frozen did
        not, so a broken freeze cannot pass unnoticed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Commit to export and render with, via `git archive` into a temp "
            f"directory (default: {DEFAULT_BASELINE}). Read-only: no checkout, "
            "no stash, no index write, so it is safe against a dirty worktree."
        ),
    )
    parser.add_argument(
        "--osprey-src",
        type=Path,
        default=os.environ.get("OSPREY_FREEZE_SRC") or None,
        help=(
            "Skip the export and import OSPREY from this src/ directory instead, "
            "for somebody who already has one. Pass --baseline alongside it to "
            "record which commit it came from."
        ),
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=PRESETS,
        help="Freeze only this preset (repeatable). Default: all four.",
    )
    args = parser.parse_args(argv)

    # An explicit --osprey-src means the caller brought their own export, so the
    # baseline sha is only whatever they told us it was; otherwise the default
    # sha is what gets exported and recorded.
    baseline = args.baseline or (None if args.osprey_src else DEFAULT_BASELINE)
    presets = tuple(args.preset) if args.preset else PRESETS

    cells: list[dict[str, Any]] = []
    completeness: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="osprey-freeze-") as tmp:
        tmp_path = Path(tmp)
        if args.osprey_src:
            source_dirs = [args.osprey_src.resolve()]
        else:
            export = tmp_path / "baseline"
            export.mkdir()
            source_dirs = export_baseline(str(baseline), export)
            print(f"exported {baseline} to a temp directory", flush=True)

        # The cell list is read off the same source the renders come from, so a
        # preset's agent list and the valid mode names are the baseline's too.
        for path in reversed(source_dirs):
            sys.path.insert(0, str(path))
        env = _cli_env(source_dirs)

        for preset in presets:
            for mode in modes_for_preset(preset):
                scratch = tmp_path / f"{preset}-{mode or UNSET_MODE_DIR}"
                scratch.mkdir(parents=True)
                record = freeze_cell(preset, mode, scratch, env)
                print(f"{record['status']:>8}  {preset} / {mode or UNSET_MODE_DIR}", flush=True)
                if record["status"] != "frozen":
                    print(f"          {record['reason']}", flush=True)
                else:
                    # The shape the equivalence test asserts against: which
                    # documents belong to the cell, and how many config.yml
                    # files in the build tree were proved to be one of them.
                    completeness[record["directory"]] = {
                        "personas": record.pop("personas"),
                        "documents": record.pop("documents"),
                        "build_configs_checked": len(record.pop("build_configs_checked")),
                    }
                cells.append(record)

    (FIXTURE_ROOT / "cells.json").write_text(
        json.dumps(cells, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    meta = {
        "baseline_sha": baseline,
        "baseline_commit": _resolved_baseline(baseline) if baseline else None,
        "external_source": bool(args.osprey_src),
        "masked_keys": {
            "stripped": dict(STRIPPED_KEYS),
            "replaced": dict(REPLACED_KEYS),
        },
        "project_name": PROJECT_NAME,
        "unset_mode_directory": UNSET_MODE_DIR,
        "api_key_env_stripped": True,
        "cells": [cell["directory"] for cell in cells],
        "completeness": completeness,
        "regenerate": "python tests/fixtures/explicit_config/freeze.py",
    }
    (FIXTURE_ROOT / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    failed = [cell for cell in cells if cell["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
