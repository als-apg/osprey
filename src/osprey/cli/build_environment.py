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
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

logger = get_logger("build")


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """Whether one provider's API-key env var is available to the built project.

    ``source`` is a human-readable origin (``"project .env"``, ``"shell
    environment"``, ``".env in <dir>"``) and is ``None`` when the key is not
    set. The key's *value* is deliberately never carried — this record is built
    to be logged.
    """

    provider: str
    var: str
    found: bool
    source: str | None = None


def _dotenv_keys(path: Path) -> dict[str, str]:
    """Parse *path* into a dict, dropping empty values. Unreadable → ``{}``.

    Empty values matter: ``.env.template`` ships bare ``VAR=`` lines, and a key
    with no value authenticates nothing.
    """
    if not path.is_file():
        return {}
    from osprey.utils.dotenv import parse_dotenv_file

    try:
        return {key: value for key, value in parse_dotenv_file(path).items() if value}
    except OSError as e:
        # e.g. a 0600 .env owned by another uid — report as "not set" rather
        # than failing a build that is otherwise complete.
        logger.debug("Could not read %s while checking provider credentials: %s", path, e)
        return {}


def detect_provider_credentials(
    project_path: Path, *, cwd: Path | None = None
) -> list[CredentialStatus]:
    """Resolve every keyed provider's API-key env var to a found/not-found status.

    Sources are checked in the order that actually determines whether the built
    project can authenticate:

    1. the built project's ``.env`` — what ships with the project;
    2. the build's working-directory ``.env`` — ``osprey.utils.config`` loads
       this into ``os.environ`` as an import side effect, so it is a real
       source even though nothing in the build reads it explicitly;
    3. the shell environment.

    Keyless providers (ollama, vllm, ds4, asksage) are excluded — they have no
    API-key env var to report on.

    Args:
        project_path: Root of the built project.
        cwd: Directory whose ``.env`` acts as source 2. Defaults to the process
            working directory.

    Returns:
        One :class:`CredentialStatus` per keyed provider, in registry order.
    """
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    cwd = Path.cwd() if cwd is None else cwd
    project_env = _dotenv_keys(project_path / ".env")
    cwd_env_path = cwd / ".env"
    # Skip when the build ran from inside the project: same file, already read.
    cwd_env = (
        {}
        if cwd_env_path.resolve() == (project_path / ".env").resolve()
        else _dotenv_keys(cwd_env_path)
    )

    statuses: list[CredentialStatus] = []
    for provider, var in PROVIDER_API_KEYS.items():
        if var is None:
            continue
        if var in project_env:
            source: str | None = "project .env"
        elif var in cwd_env:
            source = f".env in {cwd}"
        elif os.environ.get(var):
            source = "shell environment"
        else:
            source = None
        statuses.append(
            CredentialStatus(provider=provider, var=var, found=source is not None, source=source)
        )
    return statuses


def report_provider_credentials(
    project_path: Path, provider: str, *, cwd: Path | None = None
) -> list[CredentialStatus]:
    """Log a provider-credentials summary, leading with the selected provider.

    Reports keys that *were* found as well as those that were not. The generic
    ``${VAR}`` resolver in :mod:`osprey.utils.config` can only report misses —
    and knows nothing about which provider the build selected — so a missing key
    for the selected provider would otherwise be indistinguishable from the
    handful of irrelevant misses for providers the project will never use.

    A missing key warns but never aborts: the project is still worth building,
    and the key can be filled into ``.env`` afterwards.

    Args:
        project_path: Root of the built project.
        provider: The provider this project was built for.
        cwd: Passed through to :func:`detect_provider_credentials`.

    Returns:
        The statuses that were reported.
    """
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    statuses = detect_provider_credentials(project_path, cwd=cwd)
    selected = next((s for s in statuses if s.provider == provider), None)

    if selected is None:
        # Keyless provider, or a name outside the registry (custom adapter).
        if provider in PROVIDER_API_KEYS:
            logger.info("  ✓ Provider: %s — no API key required", provider)
        else:
            logger.info("  ✓ Provider: %s — no known API key env var; skipping check", provider)
    elif selected.found:
        logger.info("  ✓ Provider: %s — %s found (%s)", provider, selected.var, selected.source)
    else:
        logger.warning("  ✗ Provider: %s — %s NOT SET", provider, selected.var)
        logger.warning("      The build will complete, but the agent cannot reach the model.")
        logger.warning("      Add %s to %s or export it.", selected.var, project_path / ".env")

    others = [s for s in statuses if s is not selected]
    found_others = [s.var for s in others if s.found]
    missing_others = [s.var for s in others if not s.found]
    if found_others:
        logger.info("      Other keys detected: %s", ", ".join(found_others))
    if missing_others:
        logger.info("      Not set:             %s", ", ".join(missing_others))

    return statuses


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


def _requirement_key(requirement: str) -> tuple[str, bool]:
    """Return *requirement*'s de-duplication key and whether it is explicit.

    The key is the PEP 503 canonical distribution name, so ``Alpha_Pkg==1.0``
    and ``alpha-pkg`` collide as they would in the resolver. A requirement is
    *explicit* when it constrains what gets installed — a version specifier or
    a direct URL — as opposed to a bare name that accepts whatever resolves.

    An unparseable requirement is keyed verbatim and treated as explicit: it
    cannot be matched against anything else, so it passes through untouched
    rather than being silently displaced.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    try:
        req = Requirement(requirement)
    except InvalidRequirement:
        return requirement.strip(), True
    return canonicalize_name(req.name), bool(req.specifier) or req.url is not None


def _merge_project_requirements(
    frozen_base: list[str], dependencies: list[str], packages: list[str]
) -> list[str]:
    """Collapse the project's non-osprey requirements to one entry per distribution.

    The three sources are considered in increasing order of deliberateness:
    the frozen base (inherited from whatever the declared interpreter's
    environment happened to contain), then ``dependencies``, then
    ``environment.packages``. Precedence follows from that ordering:

    * A bare name never displaces anything — an explicit pin always wins over
      it, whichever side it came from.
    * Between two explicit requirements, the later — more deliberately
      declared — one wins. A profile that pins ``numpy==2.1`` overrides the
      ``numpy==2.2.6`` its build host happened to carry.
    * osprey's own requirement is authoritative and is dropped from this list
      entirely; :func:`_write_project_pyproject` records it first, and
      :func:`freeze_base_environment` already excludes it from the freeze.

    The surviving entry keeps its distribution's first-seen position, so the
    frozen block stays contiguous and a rebuild that changes only a pin does
    not reshuffle the recorded list.

    Args:
        frozen_base: ``name==version`` pins from :func:`freeze_base_environment`.
        dependencies: The profile's ``dependencies``.
        packages: The profile's ``environment.packages``.

    Returns:
        One requirement string per distribution, in first-seen order.
    """
    from packaging.utils import canonicalize_name

    reserved = canonicalize_name(_OSPREY_DIST_NAME)
    winner: dict[str, str] = {}
    order: list[str] = []

    for requirement in [*frozen_base, *dependencies, *packages]:
        key, explicit = _requirement_key(requirement)
        if key == reserved:
            continue
        if key not in winner:
            winner[key] = requirement
            order.append(key)
        elif explicit:
            # Only an explicit requirement displaces an incumbent; a bare name
            # asks for less than whatever is already recorded.
            winner[key] = requirement

    return [winner[key] for key in order]


def _write_project_pyproject(
    project_path: Path,
    osprey_spec: str,
    osprey_label: str,
    extra_deps: list[str],
    base_python: str,
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

    *extra_deps* is the very list that was installed, not a re-derivation of
    it: the environment on the host and the record that rebuilds it elsewhere
    cannot disagree, because there is only one list.

    Written whole on every build — never appended — so ``osprey build --force``
    cannot stack duplicate dependency blocks.

    Only called when a venv is created; ``--skip-deps`` leaves no environment
    for the file to describe, and emitting one would invite ``uv run`` to
    resolve and install in a mode that exists to avoid exactly that.

    Args:
        project_path: Built project root; the file lands directly inside it.
        osprey_spec: osprey requirement as handed to the installer.
        osprey_label: Human-readable description of where osprey came from.
        extra_deps: Every non-osprey requirement that was installed, already
            merged and de-duplicated by :func:`_merge_project_requirements`.
        base_python: Interpreter the project venv was based on. Recorded as a
            comment for provenance only — nothing ever reads it back, so a
            path that is meaningless on another machine cannot mislead a
            rebuild there.
    """
    deps = [_osprey_requirement(osprey_spec)] + list(extra_deps)

    lines = [
        "# Generated by `osprey build` — rewritten on every build.",
        "# Edit the build profile, not this file.",
        "#",
        f"# osprey source: {osprey_label}",
        f"# base interpreter: {base_python}",
        "# (provenance only — never read back; the runtime resolves its own interpreter)",
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


_OSPREY_DIST_NAME = "osprey-framework"

_VENV_PROBE = "import sys; print(1 if sys.prefix != sys.base_prefix else 0)"

_DISTRIBUTIONS_PROBE = """\
import json
import sys
from importlib.metadata import distributions

records = []
for dist in distributions():
    try:
        direct_url = dist.read_text("direct_url.json")
    except OSError:
        direct_url = None
    records.append(
        {
            "name": dist.metadata["Name"],
            "version": dist.version,
            "direct_url": direct_url,
        }
    )
json.dump(records, sys.stdout)
"""


def _run_base_probe(python_path: Path, script: str, *, what: str, timeout: int) -> str:
    """Run *script* under the base interpreter and return its stdout.

    Always isolated (``-I``): no ``PYTHONPATH``, no user site-packages. The
    freeze must describe what the base venv itself installs, not whatever the
    build's ambient environment injects into it.

    Args:
        python_path: Interpreter to run the probe with.
        script: Source passed to ``python -c``.
        what: Gerund-free description of the probe, used in error messages
            (e.g. ``"enumerate the base environment"``).
        timeout: Seconds before the probe is abandoned.

    Returns:
        The probe's stdout.

    Raises:
        BuildProfileError: The interpreter could not be run, timed out, or
            exited non-zero.
    """
    try:
        result = subprocess.run(
            [str(python_path), "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise BuildProfileError(f"Cannot {what} with base interpreter {python_path}: {e}") from e
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise BuildProfileError(
            f"Cannot {what} with base interpreter {python_path} "
            f"(exit {result.returncode}):\n{output}"
        )
    return result.stdout


def _base_is_venv(python_path: Path) -> bool:
    """Return True when *python_path* is the interpreter of a virtual environment.

    Asked of the interpreter itself (``sys.prefix != sys.base_prefix``) rather
    than inferred from the filesystem, so the answer comes from the same
    interpreter whose packages would be frozen.
    """
    out = _run_base_probe(python_path, _VENV_PROBE, what="probe", timeout=30)
    lines = out.strip().splitlines()
    return bool(lines) and lines[-1].strip() == "1"


def _probe_base_distributions(python_path: Path) -> list[dict[str, Any]]:
    """Enumerate the base environment's installed distributions.

    ``importlib.metadata`` is run *inside* the base interpreter and the result
    is marshalled back as JSON. Reading the base venv's metadata from this
    process would describe the build's environment, not the base's.

    Returns:
        One ``{"name", "version", "direct_url"}`` record per installed
        distribution. ``direct_url`` is the raw ``direct_url.json`` text, or
        ``None`` for an index install.

    Raises:
        BuildProfileError: The probe failed or returned unreadable output.
    """
    raw = _run_base_probe(
        python_path, _DISTRIBUTIONS_PROBE, what="enumerate the base environment", timeout=120
    )
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as e:
        raise BuildProfileError(
            f"Base interpreter {python_path} returned unreadable package metadata: {e}"
        ) from e
    return [record for record in records if isinstance(record, dict)]


def _osprey_requirement_specifiers() -> dict[str, Any]:
    """Map canonical distribution name → osprey's own version specifier.

    Only requirements that apply to a plain ``osprey-framework`` install are
    included: extras-gated and platform-gated requirements are dropped by
    evaluating their environment marker. Several requirements naming the same
    distribution are intersected.

    Returns:
        ``{canonical_name: packaging.specifiers.SpecifierSet}``. Empty when
        osprey's metadata is unavailable, which disables conflict detection
        rather than failing the build.
    """
    from packaging.markers import UndefinedEnvironmentName
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    try:
        requires = distribution(_OSPREY_DIST_NAME).requires or []
    except PackageNotFoundError:
        return {}

    pins: dict[str, Any] = {}
    for raw in requires:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.marker is not None:
            try:
                applies = req.marker.evaluate()
            except UndefinedEnvironmentName:
                applies = True  # marker we cannot decide — assume it applies
            if not applies:
                continue
        if not req.specifier:
            continue
        name = canonicalize_name(req.name)
        pins[name] = pins[name] & req.specifier if name in pins else req.specifier
    return pins


def _direct_url_origin(raw: str | None) -> str | None:
    """Describe a distribution's non-index origin, or ``None`` if it has one.

    A ``direct_url.json`` means the distribution was installed from a path, a
    VCS URL or a bare archive URL — there is no index coordinate that would
    reinstall it elsewhere, so it cannot be reproduced by ``name==version``.
    """
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return "installed from a direct URL"
    url = info.get("url", "")
    if info.get("dir_info", {}).get("editable"):
        return f"editable install from {unquote(urlparse(url).path) or url}"
    return f"direct URL install from {url or 'an unrecorded location'}"


def _freeze_offender_message(
    python_path: Path,
    unreproducible: list[tuple[str, str]],
    conflicts: list[tuple[str, str]],
) -> str:
    """Render the failure message naming every package that blocks the freeze.

    Both offender classes are reported together, in full. Naming one offender
    per build would make the user resolve them one build at a time.
    """
    total = len(unreproducible) + len(conflicts)
    lines = [
        f"Cannot freeze the base environment at {python_path}: "
        f"{total} package(s) cannot be reproduced in the built project."
    ]
    if unreproducible:
        lines += ["", "Not installed from a package index (no reproducible source):"]
        lines += [f"  - {detail}" for _, detail in unreproducible]
    if conflicts:
        lines += ["", f"Conflicts with {_OSPREY_DIST_NAME}'s own requirements (osprey wins):"]
        lines += [f"  - {detail}" for _, detail in conflicts]
    offenders = sorted({name for name, _ in unreproducible + conflicts}, key=str.lower)
    lines += [
        "",
        "Every package above must be resolved. Name one in the build profile's",
        "`environment.inherit_exclude` to leave it out of the freeze:",
        "",
        "  environment:",
        "    inherit_exclude:",
        *(f"      - {name}" for name in offenders),
    ]
    return "\n".join(lines)


def freeze_base_environment(python_path: Path, inherit_exclude: list[str]) -> list[str]:
    """Pin the packages installed in a venv base so a build can reproduce them.

    ``uv venv --python <venv>/bin/python`` does *not* inherit the base venv's
    packages — it yields an empty environment. This freeze is what carries the
    base's installed set into the built project, so a project (or a container
    image built from it) matches the host environment it was based on.

    Reproducibility is checked, not assumed. A package with no index
    coordinate, or one whose version osprey itself forbids, cannot be carried
    over; every such package is reported at once so a single edit to
    ``inherit_exclude`` clears them all.

    Args:
        python_path: Interpreter the project environment is based on. A bare
            (non-venv) interpreter has no installed set to freeze.
        inherit_exclude: Distribution names to leave out of the freeze.
            Matched case- and separator-insensitively (PEP 503 canonical form).

    Returns:
        ``name==version`` requirement strings, sorted by canonical name. Empty
        when *python_path* is not a venv interpreter.

    Raises:
        BuildProfileError: The base interpreter could not be probed, or one or
            more packages cannot be reproduced (all of them are named).
    """
    import time

    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion

    started = time.perf_counter()
    if not _base_is_venv(python_path):
        logger.debug("Base interpreter %s is not a venv — nothing to freeze", python_path)
        return []

    excluded = {canonicalize_name(name) for name in inherit_exclude}
    excluded.add(canonicalize_name(_OSPREY_DIST_NAME))

    survivors: dict[str, tuple[str, str, str | None]] = {}
    for record in _probe_base_distributions(python_path):
        name, version = record.get("name"), record.get("version")
        if not name or not version:
            # Metadata too damaged to pin by name; nothing to carry over.
            logger.debug("Skipping base distribution with incomplete metadata: %r", record)
            continue
        key = canonicalize_name(str(name))
        if key in excluded or key in survivors:
            continue
        survivors[key] = (str(name), str(version), record.get("direct_url"))

    pins = _osprey_requirement_specifiers()
    unreproducible: list[tuple[str, str]] = []
    conflicts: list[tuple[str, str]] = []
    for dist_key, (name, version, direct_url) in sorted(survivors.items()):
        origin = _direct_url_origin(direct_url)
        if origin is not None:
            unreproducible.append((name, f"{name} {version} — {origin}"))
            continue
        specifier = pins.get(dist_key)
        if specifier is None:
            continue
        try:
            satisfied = specifier.contains(version, prereleases=True)
        except InvalidVersion:
            continue  # unparseable version — nothing to compare against
        if not satisfied:
            conflicts.append(
                (name, f"{name} {version} installed, {_OSPREY_DIST_NAME} requires {specifier}")
            )

    if unreproducible or conflicts:
        raise BuildProfileError(_freeze_offender_message(python_path, unreproducible, conflicts))

    pinned = [f"{name}=={version}" for _, (name, version, _) in sorted(survivors.items())]
    logger.info(
        "  ✓ Froze %d packages from base venv %s (%.2fs)",
        len(pinned),
        python_path,
        time.perf_counter() - started,
    )
    return pinned


# Ceiling on the project-venv dependency install.
#
# This install pulls osprey's whole transitive tree into a fresh venv — around
# 1.4 GB installed, from ~2.2 GB of wheels, on a cold cache. How long that takes
# is a property of the network, not of the project, so the cap has to bound an
# installer that is *stuck* rather than one that is merely slow. A tight cap
# turns an ordinary cold-cache download into an abort that fires intermittently
# and reads as flakiness rather than as a limit set too low.
_VENV_INSTALL_TIMEOUT_S = 1800


def _create_project_venv(project_path: Path, profile: Any) -> list[str]:
    """Create the project venv and install osprey + profile deps.

    This is the single place where the project's Python environment is set up.
    One venv, one install command, one resolver pass. The resolver sees all
    dependencies together (osprey + profile deps + ``environment.packages``) and
    either succeeds or fails.

    The interpreter the venv is based on comes from
    ``profile.environment.python`` when set, and from the build's own
    interpreter otherwise. A bare-interpreter base and a venv-interpreter base
    take the identical path — a venv's python *is* an interpreter — so nothing
    here branches on venv-ness. Basing the venv on another venv's interpreter
    does **not** inherit that venv's installed packages; only the requirements
    assembled here are installed. :func:`freeze_base_environment` is what turns
    a declared venv base's packages into such requirements, so a declared base
    is carried over deliberately, as pins, rather than by inheritance.

    The same assembled list is installed here, recorded by
    :func:`_write_project_pyproject`, and returned to the caller, which renders
    it into the generated Dockerfile's install line. Three consumers, one list:
    the host venv, its record, and the image built from it cannot disagree
    about what the project depends on, because none of them re-derives it.

    See :func:`_resolve_osprey_spec` for how ``profile.osprey_install`` is
    interpreted.

    Returns:
        Every non-osprey requirement that was installed, already merged and
        de-duplicated by :func:`_merge_project_requirements`. osprey's own
        requirement is not in it — the image installs osprey from its own
        pinned spec, exactly as :func:`_write_project_pyproject` records it
        separately.
    """
    import sys

    venv_path = project_path / ".venv"
    uv_path = os.environ.get("UV") or shutil.which("uv")
    base_python = profile.environment.resolved_python()
    base_python_arg = str(base_python) if base_python is not None else sys.executable

    # --- Create venv ---
    logger.info("  Creating project virtual environment...")
    if base_python is not None:
        logger.info("  Base interpreter: %s", base_python)
    if uv_path:
        result = subprocess.run(
            [uv_path, "venv", str(venv_path), "--python", base_python_arg, "--quiet"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    else:
        result = subprocess.run(
            [base_python_arg, "-m", "venv", str(venv_path)],
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

    # --- Freeze the declared base environment ---
    # Computed exactly once per build: the probe enumerates a whole environment,
    # and a second call would be a second chance for the installed set and the
    # recorded set to disagree — the divergence this freeze exists to close.
    # Only a *declared* base is frozen. With no `environment.python` the base is
    # the interpreter osprey itself happens to be installed into: an accident of
    # how the operator installed the framework, not a curated environment, and
    # carrying its packages (test tooling included) into every built project
    # would be inheritance nobody asked for.
    frozen_base = (
        freeze_base_environment(base_python, list(profile.environment.inherit_exclude or []))
        if base_python is not None
        else []
    )

    # --- Install osprey + frozen base + profile deps + environment packages ---
    # `extra_deps` is the single non-osprey requirement list: everything the
    # profile asks for, resolved in one pass alongside osprey itself, and the
    # same list `_write_project_pyproject` records below.
    extra_deps = _merge_project_requirements(
        frozen_base,
        list(profile.dependencies or []),
        list(profile.environment.packages or []),
    )
    all_deps = [osprey_spec] + extra_deps
    venv_python = venv_path / "bin" / "python"
    dep_count = len(extra_deps)

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
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_VENV_INSTALL_TIMEOUT_S
            )
        except subprocess.TimeoutExpired as e:
            # A bare TimeoutExpired escapes to build_cmd's catch-all and prints
            # "Unexpected error", naming neither the step nor a way forward.
            raise BuildProfileError(
                f"installing osprey + {dep_count} deps into the project venv did not "
                f"finish within {_VENV_INSTALL_TIMEOUT_S // 60} minutes.\n"
                "        This install downloads osprey's full dependency tree (over a "
                "gigabyte on a cold cache), so a slow or interrupted connection is the "
                "usual cause.\n"
                "        Re-running the build resumes from what already downloaded — "
                "completed packages are served from the local cache.\n"
                "        On a slow or restricted link, point the build at a closer "
                "package mirror (UV_INDEX_URL, or PIP_INDEX_URL without uv)."
            ) from e

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
                    *extra_deps,
                ]
                if extra_deps
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
                    *extra_deps,
                ]
                if extra_deps
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
    # The recorded list is the installed list — same object, no re-derivation.
    _write_project_pyproject(project_path, osprey_spec, osprey_label, extra_deps, base_python_arg)

    # Handed back for the image's install line — the third consumer of this one
    # list. Returned only after the install and the record both succeeded, so a
    # failed build never yields a list for an image to be built from.
    return extra_deps
