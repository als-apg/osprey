"""Project-directory persistence for the build pipeline.

Everything that writes into the freshly rendered project directory that isn't a
service injector: the ``--force`` pre-clear (preserving user-owned state),
config overrides, overlay file copies + their scaffold-ownership registration,
persisting profile MCP servers / custom categories into ``config.yml``, and the
initial git commit.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("build")


def _clear_rendered_project_dir(project_path: Path) -> list[str]:
    """Clear a project directory for ``--force``, keeping user-owned state.

    Removes every top-level entry the build renders, but leaves what the user
    owns — ``.env`` (secrets and the service tokens/passwords live docker
    volumes were initialized with), ``_agent_data/`` (agent workspace), and
    ``.git`` (the project's own history) — in place, untouched. This is what
    makes ``--force`` (the staleness advisory's remedy) safe to run on a
    stale project. Mirrors the user-owned exclusion set of
    :func:`osprey.cli.templates.manifest.calculate_file_checksums`.

    Returns:
        Names of the preserved entries that were actually present.
    """
    user_owned = (".env", "_agent_data", ".git")
    preserved: list[str] = []
    for entry in sorted(project_path.iterdir(), key=lambda p: p.name):
        if entry.name in user_owned:
            preserved.append(entry.name)
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return preserved


def _apply_config_overrides(project_path: Path, config_dict: dict[str, Any]) -> None:
    """Apply dot-notation config overrides to the project's config.yml."""
    from osprey.utils.config_writer import config_update_fields

    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found at %s — skipping config overrides", config_path)
        return
    config_update_fields(config_path, config_dict)


def _copy_overlay_files(
    profile_dir: Path, project_path: Path, overlay_dict: dict[str, str]
) -> None:
    """Copy overlay files/directories from profile dir into the project.

    Args:
        profile_dir: Directory containing the profile and overlay sources.
        project_path: Root of the built project.
        overlay_dict: Mapping of source (relative to profile_dir) → destination
            (relative to project_path).
    """
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    with Progress(
        TextColumn("  Copying overlays"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("overlays", total=len(overlay_dict))
        for src_rel, dst_rel in overlay_dict.items():
            src = (profile_dir / src_rel).resolve()
            dst = (project_path / dst_rel).resolve()

            # Path traversal guard
            if not dst.is_relative_to(project_path.resolve()):
                raise ValueError(f"Overlay destination escapes project root: {dst_rel}")

            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

            logger.debug("Overlay: %s → %s", src_rel, dst_rel)
            progress.advance(task)


def _register_overlay_artifacts(project_path: Path, overlay_dict: dict[str, str]) -> int:
    """Register overlay files landing in .claude/ as user_owned in config.yml.

    The Scaffold Gallery flags .claude/ files that aren't in the BuildArtifactCatalog
    or config.yml's scaffold.user_owned as "untracked."  Profile overlay files
    (agents, skills, rules) aren't framework artifacts, so they must be
    registered as user_owned to avoid the untracked warning.
    """
    from osprey.services.build_artifacts.ownership import update_config_add_user_owned

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return 0

    # Subdirectories the Scaffold Gallery scans for untracked files
    # (mirrors ScaffoldGalleryService._scan_dirs)
    scan_prefixes = tuple(
        f".claude/{d}/" for d in ("agents", "commands", "output-styles", "rules", "skills")
    )

    registered = 0
    for _src_rel, dst_rel in overlay_dict.items():
        dst_path = project_path / dst_rel

        if dst_path.is_dir():
            # Directory overlay — find all .md files within
            md_files = [
                str(f.relative_to(project_path)) for f in dst_path.rglob("*.md") if f.is_file()
            ]
        elif dst_path.is_file() and dst_rel.endswith(".md"):
            md_files = [dst_rel]
        else:
            continue

        for rel_path in md_files:
            if not any(rel_path.startswith(p) for p in scan_prefixes):
                continue
            # Derive canonical name: .claude/rules/foo.md → rules/foo
            canonical = rel_path[len(".claude/") : -len(".md")]
            if update_config_add_user_owned(project_path, canonical):
                registered += 1

    return registered


def _persist_mcp_servers(project_path: Path, mcp_servers: dict[str, Any]) -> None:
    """Persist profile MCP server definitions into config.yml's claude_code.servers.

    Servers are written in the format that ``_custom_server_from_spec()`` parses,
    so ``regenerate_claude_code()`` can reconstruct them into the rendered
    ``.mcp.json`` and ``settings.json``.  Placeholders like ``{project_root}``
    are preserved as-is — resolution happens during regen.
    """
    from osprey.utils.config_writer import _load, _save

    from .build_profile import McpServerDef

    config_path = project_path / "config.yml"
    data = _load(config_path)

    # Ensure claude_code.servers section exists
    if "claude_code" not in data:
        from ruamel.yaml import CommentedMap

        data["claude_code"] = CommentedMap()
    cc = data["claude_code"]
    if "servers" not in cc:
        from ruamel.yaml import CommentedMap

        cc["servers"] = CommentedMap()
    servers_section = cc["servers"]

    for name, server in mcp_servers.items():
        if not isinstance(server, McpServerDef):
            continue

        spec: dict[str, Any] = {}
        if server.url:
            spec["transport"] = "http"
            spec["url"] = server.url
        else:
            spec["transport"] = "stdio"
            if server.command:
                spec["command"] = server.command
            if server.args:
                spec["args"] = list(server.args)
            if server.env:
                spec["env"] = dict(server.env)
        if server.port is not None and server.url:
            # Emit a derived network block so non-Claude consumers
            # (compose-port checkers, integration-tests probes) can read
            # host/docker URLs without re-deriving them.
            # NOTE: docker_url uses the MCP server's YAML key (`name`) as the
            # container hostname. This assumes the operator names the
            # docker-compose service identically to the mcp_servers entry
            # (e.g. mcp_servers.matlab → service: matlab). If they diverge,
            # docker_url will point at a non-existent host.
            spec["network"] = {
                "port": int(server.port),
                "host_url": f"http://localhost:{server.port}/mcp",
                "docker_url": f"http://{name}:{server.port}/mcp",
            }
        if server.permissions:
            spec["permissions"] = dict(server.permissions)

        servers_section[name] = spec

    _save(config_path, data)


def _persist_categories(project_path: Path, categories: dict[str, dict[str, str]]) -> None:
    """Persist custom artifact categories into config.yml's ``categories`` section."""
    from osprey.utils.config_writer import _load, _save

    config_path = project_path / "config.yml"
    data = _load(config_path)

    if "categories" not in data:
        from ruamel.yaml import CommentedMap

        data["categories"] = CommentedMap()
    cat_section = data["categories"]

    for key, spec in categories.items():
        from ruamel.yaml import CommentedMap

        entry = CommentedMap()
        entry["label"] = spec["label"]
        entry["color"] = spec["color"]
        cat_section[key] = entry

    _save(config_path, data)


def _git_init_and_commit(project_path: Path) -> None:
    """Initialize a git repo and create an initial commit."""
    import os
    import subprocess

    # Check if project is inside an existing git repo
    inside_existing_repo = False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            parent_root = Path(result.stdout.strip()).resolve()
            if parent_root != project_path.resolve():
                inside_existing_repo = True
    except FileNotFoundError:
        pass

    try:
        subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial project from osprey build"],
            cwd=project_path,
            check=True,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "osprey",
                "GIT_AUTHOR_EMAIL": "osprey@build",
                "GIT_COMMITTER_NAME": "osprey",
                "GIT_COMMITTER_EMAIL": "osprey@build",
            },
        )
        logger.info("  ✓ Initialized git repository")
        if inside_existing_repo:
            logger.warning(
                "  Note: created a nested git repo inside %s.\n"
                "     This is required for Claude Code project isolation (it uses\n"
                "     the git root to discover .claude/ settings). The parent repo\n"
                "     will treat this directory as opaque.",
                parent_root,
            )
    except FileNotFoundError:
        logger.warning(
            "  git not found — project created but not initialized as a git repo.\n"
            "     Claude Code requires git. Run 'git init && git add . && git commit'"
            " manually."
        )
    except subprocess.CalledProcessError:
        logger.warning(
            "  git init succeeded but initial commit failed.\n"
            "     Run 'git add . && git commit' manually."
        )
