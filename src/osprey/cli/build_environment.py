"""Project virtual-environment creation and ``.env`` templating helpers.

The single place the built project's Python environment is set up (one venv,
one install pass over ``osprey`` + profile deps) plus the ``.env`` /
``.env.template`` writers. Kept flat under ``cli/`` so
:func:`_resolve_osprey_spec`'s source-tree fallback (``parents[3]``) still
resolves to the repo root for editable/source checkouts.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

logger = get_logger("build")


def _copy_env_file(profile_dir: Path, project_path: Path, env_file: str) -> None:
    """Copy a profile-provided .env file to the built project.

    An existing project ``.env`` is merged, not clobbered: its values win and
    keys it alone carries are appended (see
    :func:`osprey.utils.dotenv.merge_env_preserving_existing`), so a --force
    re-render never resets user secrets to template defaults.
    """
    src = (profile_dir / env_file).resolve()
    dst = project_path / ".env"
    if dst.exists():
        from osprey.utils.dotenv import merge_env_preserving_existing

        merged = merge_env_preserving_existing(
            src.read_text(encoding="utf-8"), dst.read_text(encoding="utf-8")
        )
        dst.write_text(merged, encoding="utf-8")
        logger.info("  ✓ Merged %s → .env (existing values preserved)", env_file)
        return
    shutil.copy2(src, dst)
    logger.info("  ✓ Copied %s → .env", env_file)


def _generate_env_template(project_path: Path, env_config: Any) -> None:
    """Generate a .env.template file from the profile's env configuration."""
    lines: list[str] = []
    if env_config.required:
        lines.append("# Required")
        for var in env_config.required:
            lines.append(f"{var}=")
    if env_config.defaults:
        if lines:
            lines.append("")
        lines.append("# Defaults")
        for var, value in env_config.defaults.items():
            lines.append(f"{var}={value}")
    lines.append("")  # Trailing newline

    env_path = project_path / ".env.template"
    env_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  ✓ Generated .env.template")
    if not (project_path / ".env").exists():
        logger.info("  Hint: Copy .env.template to .env and fill in required values")


def _resolve_osprey_spec(osprey_install: str) -> tuple[str, str]:
    """Resolve the osprey install spec for the project venv.

    Returns ``(spec, label)`` where ``spec`` is the pip/uv install argument
    and ``label`` is a human-readable identifier used in logs and the
    generated requirements.txt comment.

    The ``osprey_install`` value drives the resolution:
      - ``"local"`` (default): consult ``importlib.metadata``. Editable
        installs (``pip install -e .``, ``uv sync``) install from the source
        tree; non-editable installs (``uv tool install``, wheels from PyPI)
        pin to the running version (``osprey-framework==<version>``).
      - ``"pip"``: install ``osprey-framework`` from PyPI, unpinned.
      - anything else: treated as a PEP 508 spec, passed through verbatim.
    """
    if osprey_install == "local":
        try:
            dist = distribution("osprey-framework")
        except PackageNotFoundError:
            dist = None

        direct_url_text = dist.read_text("direct_url.json") if dist else None
        info = json.loads(direct_url_text) if direct_url_text else {}
        if info.get("dir_info", {}).get("editable"):
            src_path = unquote(urlparse(info["url"]).path)
            return src_path, f"editable: {src_path}"

        if dist is not None:
            spec = f"osprey-framework=={dist.version}"
            return spec, spec

        # Metadata unavailable (rare: e.g. running osprey directly from a
        # source tree without installing it). Fall back to the source root
        # one final time so dev workflows that bypass install still work.
        osprey_root = Path(__file__).resolve().parents[3]
        if (osprey_root / "pyproject.toml").exists():
            return str(osprey_root), f"local: {osprey_root}"
        raise BuildProfileError(
            "Cannot resolve osprey install location: package metadata is "
            f"missing and no source tree is present at {osprey_root}. "
            "Install osprey-framework with `uv tool install osprey-framework` "
            "or set `osprey_install` explicitly in your profile."
        )

    if osprey_install == "pip":
        return "osprey-framework", "osprey-framework"

    return osprey_install, osprey_install


def _normalize_project_name(raw: str) -> str:
    """Normalize a project directory name into a PEP 508-safe project name.

    Project directories are named by the operator (``osprey build "My Rig"``),
    so they carry spaces and punctuation that a ``[project] name`` field may
    not. Runs of invalid characters collapse to a single dash.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._").lower()
    return slug or "osprey-project"


def _osprey_requirement(osprey_spec: str) -> str:
    """Render the resolved osprey install spec as a PEP 508 requirement.

    :func:`_resolve_osprey_spec` returns a value suitable for ``uv pip install``
    on the command line, where a bare filesystem path is legal. A ``[project]
    dependencies`` entry is not a command line: a bare path there is
    unparseable, so source-tree and editable installs are rewritten as PEP 508
    direct references (``osprey-framework @ file:///path``). Version pins and
    plain names pass through untouched.
    """
    if osprey_spec.startswith((".", "~", "/")):
        resolved = Path(osprey_spec).expanduser().resolve()
        return f"osprey-framework @ {resolved.as_uri()}"
    return osprey_spec


def _project_requires_python() -> str:
    """Mirror the framework's own ``Requires-Python`` into the built project.

    A built project runs osprey, so its floor is osprey's floor. Reading it from
    installed metadata keeps the two from drifting when the framework raises its
    minimum.
    """
    try:
        requires = distribution("osprey-framework").metadata["Requires-Python"]
    except PackageNotFoundError:
        requires = None
    return requires or ">=3.11"


def _write_project_pyproject(
    project_path: Path, osprey_spec: str, osprey_label: str, profile: Any
) -> None:
    """Write the built project's ``pyproject.toml`` dependency record.

    Serves two purposes, both load-bearing:

    1. **Dependency record.** Declares the exact set
       :func:`_create_project_venv` installed, so ``uv sync`` rebuilds the
       environment rather than pruning it down to nothing.
    2. **Project anchor.** ``uv run`` locates an environment by walking *up*
       from the working directory looking for a ``pyproject.toml``. Without one,
       ``uv run osprey web`` inside a built project that happens to sit under
       another project (a checkout, a monorepo) silently resolves the
       *ancestor's* venv. This file stops that walk at the project root.

    Written whole on every build — never appended — so ``osprey build --force``
    cannot stack duplicate dependency blocks.

    Only called when a venv is created; ``--skip-deps`` leaves no environment
    for the file to describe, and emitting one would invite ``uv run`` to
    resolve and install in a mode that exists to avoid exactly that.
    """
    deps = [_osprey_requirement(osprey_spec)] + list(profile.dependencies or [])

    lines = [
        "# Generated by `osprey build` — rewritten on every build.",
        "# Edit the build profile, not this file.",
        "#",
        f"# osprey source: {osprey_label}",
        "",
        "[project]",
        f"name = {json.dumps(_normalize_project_name(project_path.name))}",
        'version = "0.1.0"',
        f"requires-python = {json.dumps(_project_requires_python())}",
        "dependencies = [",
        *(f"    {json.dumps(dep)}," for dep in deps),
        "]",
        "",
        "# Intentionally no [build-system]. A built project is a configuration",
        "# directory, not an installable package — it has no importable source",
        "# tree. Declaring a build backend would make every `uv run` invocation",
        "# try to build the project and fail. With a [project] table and no",
        "# [build-system], uv treats this as a virtual project: it anchors the",
        "# environment lookup here without attempting a build.",
        "",
    ]
    (project_path / "pyproject.toml").write_text("\n".join(lines), encoding="utf-8")
    logger.info("  ✓ Recorded %d dependencies in pyproject.toml", len(deps))


def _create_project_venv(project_path: Path, profile: Any) -> None:
    """Create the project venv and install osprey + profile deps.

    This is the single place where the project's Python environment is set up.
    One venv, one install command, one resolver pass. The resolver sees all
    dependencies together (osprey + profile deps) and either succeeds or fails.

    See :func:`_resolve_osprey_spec` for how ``profile.osprey_install`` is
    interpreted.
    """
    import sys

    venv_path = project_path / ".venv"
    uv_path = os.environ.get("UV") or shutil.which("uv")

    # --- Create venv ---
    logger.info("  Creating project virtual environment...")
    if uv_path:
        result = subprocess.run(
            [uv_path, "venv", str(venv_path), "--python", sys.executable, "--quiet"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    else:
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise BuildProfileError(f"Failed to create project venv: {output}")

    # --- Resolve osprey install spec ---
    osprey_install = profile.osprey_install or "local"
    osprey_spec, osprey_label = _resolve_osprey_spec(osprey_install)

    # --- Install osprey + profile deps ---
    all_deps = [osprey_spec] + list(profile.dependencies or [])
    venv_python = venv_path / "bin" / "python"
    dep_count = len(profile.dependencies or [])

    if uv_path:
        cmd = [uv_path, "pip", "install", "--quiet", "-p", str(venv_python), *all_deps]
    else:
        cmd = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            *all_deps,
        ]

    from rich.live import Live
    from rich.spinner import Spinner

    spinner = Spinner("dots", text=f"  Installing osprey ({osprey_label}) + {dep_count} deps...")
    with Live(spinner, transient=True):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0:
        logger.info("  ✓ Installed osprey + %d profile deps into project venv", dep_count)
    elif "litellm" in (result.stdout + result.stderr).lower():
        # ---------------------------------------------------------------
        # TEMPORARY WORKAROUND — litellm supply chain attack (2026-03-24)
        #
        # litellm versions 1.82.7-1.82.8 were compromised with credential-
        # stealing malware (TeamPCP attack chain). PyPI has quarantined the
        # entire package, so uv refuses to resolve it.
        #
        # Workaround: install osprey --no-deps + profile deps into the
        # project venv, then add a .pth file pointing to OSPREY's own
        # site-packages so the project inherits litellm and other
        # transitive deps from the known-good build environment.
        #
        # REVERT THIS when litellm is restored on PyPI:
        #   1. Remove this entire elif block
        #   2. The normal install path above will work again
        # ---------------------------------------------------------------
        logger.warning(
            "  litellm unavailable on PyPI (quarantined) — inheriting from build environment"
        )
        # Install osprey (no transitive deps) + profile deps
        if uv_path:
            cmd_nodeps = [
                uv_path,
                "pip",
                "install",
                "--quiet",
                "-p",
                str(venv_python),
                "--no-deps",
                osprey_spec,
            ]
            cmd_profile = (
                [
                    uv_path,
                    "pip",
                    "install",
                    "--quiet",
                    "-p",
                    str(venv_python),
                    *list(profile.dependencies or []),
                ]
                if profile.dependencies
                else None
            )
        else:
            cmd_nodeps = [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-deps",
                osprey_spec,
            ]
            cmd_profile = (
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--disable-pip-version-check",
                    *list(profile.dependencies or []),
                ]
                if profile.dependencies
                else None
            )

        spinner = Spinner("dots", text="  Installing osprey (--no-deps)...")
        with Live(spinner, transient=True):
            r = subprocess.run(cmd_nodeps, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise BuildProfileError(
                f"Failed to install osprey --no-deps:\n{(r.stdout + r.stderr).strip()}"
            )

        if cmd_profile:
            spinner = Spinner("dots", text=f"  Installing {dep_count} profile deps...")
            with Live(spinner, transient=True):
                r = subprocess.run(cmd_profile, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise BuildProfileError(
                    f"Failed to install profile deps:\n{(r.stdout + r.stderr).strip()}"
                )

        # Add .pth file so project venv can import osprey's transitive deps
        # (litellm, pandas, etc.) from the build environment's site-packages
        build_site_packages = Path(sys.prefix) / "lib"
        # Find the actual site-packages dir (python version varies)
        sp_dirs = list(build_site_packages.glob("python*/site-packages"))
        if sp_dirs:
            pth_path = venv_path / "lib"
            proj_sp = list(pth_path.glob("python*/site-packages"))
            if proj_sp:
                pth_file = proj_sp[0] / "_osprey_build_env.pth"
                pth_file.write_text(f"{sp_dirs[0]}\n")
                logger.info("  ✓ Linked build environment site-packages via .pth")

        logger.info("  ✓ Installed osprey (--no-deps) + %d profile deps", dep_count)
    else:
        output = (result.stdout + result.stderr).strip()
        raise BuildProfileError(
            f"Failed to install project dependencies (exit {result.returncode}):\n{output}"
        )

    # --- Record deps in pyproject.toml ---
    _write_project_pyproject(project_path, osprey_spec, osprey_label, profile)
