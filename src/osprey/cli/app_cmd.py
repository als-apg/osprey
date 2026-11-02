"""Make and check a deployment's applications without building anything.

``osprey app new`` copies the shipped example under a new name and lists it in the profile.
``osprey app check`` reports what ``osprey build`` reports of them, through the same
:func:`osprey.apps.loader.check_deployment`, then what only the application's own source
can tell: the import boundary over each of its Python packages, ESLint over ``panel/`` when
it holds JS, and Vitest over ``tests/**/*.test.{js,mjs}`` when that matches a file. A
deployment repository has no JS toolchain, so the command installs the one pinned in
``osprey/apps/js_check/`` into the user's cache once and runs it from there; an application
with no JS needs no Node.

Note: the loader is imported inside the command body, so ``osprey --help`` does not load it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn

import click
import yaml

from osprey.apps.boundary import check_package
from osprey.apps.errors import AppManifestError, Violation
from osprey.apps.manifest import APP_NAME_PATTERN
from osprey.utils.workspace import APPS_DIRNAME

from .application_material import app_build_output
from .output import fail, report
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option

_OSPREY_DIR = Path(__file__).resolve().parents[1]
#: The shipped example ``osprey app new`` copies.
EXAMPLE_APP_DIR = _OSPREY_DIR / "profiles" / "presets" / APPS_DIRNAME / "knob-example"
#: Every app-scoped identifier of the example is its name in one of these two forms.
_EXAMPLE_NAMES = ("knob-example", "knob_example")
#: The anchored ``.gitignore`` entry for the files a build fetches into a panel.
_VENDOR_IGNORE = f"/{APPS_DIRNAME}/*/panel/vendor/"
#: The pinned toolchain's manifest, lockfile and configs, copied beside its ``node_modules``.
JS_CHECK_DIR = _OSPREY_DIR / "apps" / "js_check"
#: What ``/design-system/js`` resolves to under a panel's tests.
_DESIGN_SYSTEM_JS = _OSPREY_DIR / "interfaces" / "design_system" / "static" / "js"

_JS_SUFFIXES = (".js", ".mjs")
_JS_TEST_SUFFIXES = (".test.js", ".test.mjs")
#: Third-party directories; what the build fetches is told apart by the manifest.
_NOT_AUTHORED_DIRS = frozenset({"vendor", "node_modules"})


class _ToolchainUnavailable(Exception):
    """The pinned JS toolchain is neither cached nor fetchable."""


@click.group()
def app() -> None:
    """Make and check a deployment's applications."""


@app.command("new")
@click.argument("name")
@repo_option
@click.option("--force", is_flag=True, help="Replace an existing apps/NAME/ directory.")
def new(name: str, repo: Path | None, force: bool) -> None:
    """Copy the shipped example application into apps/NAME/ and list it.

    Every app-scoped name of the example is rewritten to NAME (its Python package to
    NAME with underscores); no port is pinned. The profile's apps: block gains
    NAME: {}, and .gitignore the entry that keeps fetched panel files out of git. An
    existing apps/NAME/ is left alone unless --force replaces it.
    """
    from .build_profile import write_back_cli_overrides

    package = name.replace("-", "_")
    if not re.fullmatch(APP_NAME_PATTERN, name) or not package.isidentifier():
        raise click.BadParameter(
            "lower-case letters, digits and dashes, starting with a letter", param_hint="NAME"
        )
    root = find_repo_root(repo)
    target = root / APPS_DIRNAME / name
    if target.exists():
        if not force:
            raise click.UsageError(f"{APPS_DIRNAME}/{name}/ exists; --force replaces it.")
        shutil.rmtree(target)
    shutil.copytree(
        EXAMPLE_APP_DIR, target, ignore=shutil.ignore_patterns("__pycache__", "*.py[co]")
    )
    _rename(target, dict(zip(_EXAMPLE_NAMES, (name, package), strict=True)))

    profile_path = root / PROFILE_FILENAME
    listed = (yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}).get("apps") or {}
    if name not in listed:
        write_back_cli_overrides(profile_path, (f"apps.{name}={{}}",))
    _ignore_vendor_files(root / ".gitignore")
    report(f"✓ {APPS_DIRNAME}/{name}/ written and listed; run `osprey app check`.")


def _rename(root: Path, names: Mapping[str, str]) -> None:
    """Rewrite every *names* key to its value in the paths and texts under *root*."""

    def renamed(text: str) -> str:
        for old, replacement in names.items():
            text = text.replace(old, replacement)
        return text

    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file():
            path.write_text(renamed(path.read_text(encoding="utf-8")), encoding="utf-8")
        if renamed(path.name) != path.name:
            path.rename(path.with_name(renamed(path.name)))


def _ignore_vendor_files(gitignore: Path) -> None:
    """Make sure *gitignore* carries :data:`_VENDOR_IGNORE`."""
    text = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    if _VENDOR_IGNORE in text.splitlines():
        return
    separator = "\n" if text and not text.endswith("\n") else ""
    gitignore.write_text(f"{text}{separator}{_VENDOR_IGNORE}\n", encoding="utf-8")


@app.command("check")
@repo_option
@click.pass_context
def check(ctx: click.Context, repo: Path | None) -> None:
    """Check the applications the profile's apps: block names, as `osprey build` would.

    Reports every violation with its application, field and rule, and adds the import
    boundary and each application's own JS checks. The JS checks execute the
    application's JS tests, with no secret of this shell's environment. Exits 1 on any
    violation.
    """
    from osprey.apps.loader import check_deployment
    from osprey_connectors.errors import BuildProfileError

    root = find_repo_root(repo)
    try:
        deployment = check_deployment(root)
    except AppManifestError as error:
        _refuse(ctx, error.violations)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)

    for name in deployment.unnamed:
        report(
            f"{APPS_DIRNAME}/{name}/ not checked: the apps: block of {PROFILE_FILENAME} "
            "does not name it"
        )
    if not deployment.manifests:
        if not deployment.unnamed:
            report(f"Nothing to check: no application under {APPS_DIRNAME}/.")
        return

    app_dirs = [root / APPS_DIRNAME / manifest.name for manifest in deployment.manifests]
    violations = [*_boundary_violations(app_dirs), *_js_violations(app_dirs)]
    if violations:
        _refuse(ctx, violations)
    report(f"✓ {len(app_dirs)} application(s) pass.")


def _refuse(ctx: click.Context, violations: Sequence[Violation]) -> NoReturn:
    """Print every violation, then stop with exit status 1."""
    for violation in violations:
        report(f"{violation} [{violation.rule}]")
    fail(f"{len(violations)} violation(s).")
    ctx.exit(1)


def _boundary_violations(app_dirs: Iterable[Path]) -> list[Violation]:
    """The import boundary over each importable package of each application."""
    found: list[Violation] = []
    for app_dir in app_dirs:
        packages = _packages(app_dir)
        if not packages:
            report(
                f"{APPS_DIRNAME}/{app_dir.name}/: no Python package, no import boundary to check"
            )
        found += [
            dataclasses.replace(
                violation, source=f"{APPS_DIRNAME}/{app_dir.name}/{violation.source}"
            )
            for package in packages
            for violation in check_package(package, app=app_dir.name)
        ]
    return found


def _packages(app_dir: Path) -> list[Path]:
    """The top-level directories of an application that hold an ``__init__.py``."""
    return [
        entry
        for entry in sorted(app_dir.iterdir(), key=lambda path: path.name)
        if not entry.name.startswith(".")
        and entry.name != "__pycache__"
        and (entry / "__init__.py").is_file()
    ]


def _js_violations(app_dirs: Iterable[Path]) -> list[Violation]:
    """ESLint and Vitest over each application that has something for them to read."""
    work = [(app_dir, _panel_js(app_dir), _js_tests(app_dir)) for app_dir in app_dirs]
    work = [item for item in work if item[1] or item[2]]
    if not work:
        return []
    if shutil.which("node") is None:
        return [
            Violation(
                app_dir.name,
                "",
                "node_required",
                "Node required: this application has panel JS or a JS test; install Node",
                source=f"{APPS_DIRNAME}/{app_dir.name}",
            )
            for app_dir, _panel, _tests in work
        ]
    try:
        toolchain = _toolchain()
    except _ToolchainUnavailable as error:
        return [Violation("", "", "js_toolchain", str(error))]
    from osprey.deployment.wheel_build import _wheel_build_env

    # The tools run the application's own code, so they get what a wheel build gets,
    # less the proxies: fetching is the install's business alone.
    env = {name: value for name, value in _wheel_build_env().items() if "proxy" not in name.lower()}
    found: list[Violation] = []
    for app_dir, panel, tests in work:
        if panel:
            found += _tool_violation(app_dir, "eslint", _eslint_command(toolchain, panel), env)
        if tests:
            found += _tool_violation(
                app_dir,
                "vitest",
                _vitest_command(toolchain),
                {**env, "OSPREY_DESIGN_SYSTEM_JS": str(_DESIGN_SYSTEM_JS)},
            )
    return found


def _authored(paths: Iterable[Path], app_dir: Path) -> list[str]:
    """Those of *paths* the application wrote, relative to *app_dir*, sorted."""
    built = app_build_output(app_dir)
    relative = (path.relative_to(app_dir) for path in paths if path.is_file())
    return sorted(
        rel.as_posix()
        for rel in relative
        if not built(rel.as_posix())
        and not rel.name.endswith(".min.js")
        and not any(part in _NOT_AUTHORED_DIRS or part.startswith(".") for part in rel.parts[:-1])
    )


def _panel_js(app_dir: Path) -> list[str]:
    """The JS the application wrote under ``panel/``."""
    panel = app_dir / "panel"
    files = (path for path in panel.rglob("*") if path.name.endswith(_JS_SUFFIXES))
    return _authored(files, app_dir)


def _js_tests(app_dir: Path) -> list[str]:
    """The files ``tests/**/*.test.{js,mjs}`` matches."""
    tests = app_dir / "tests"
    files = (path for path in tests.rglob("*") if path.name.endswith(_JS_TEST_SUFFIXES))
    return _authored(files, app_dir)


def _eslint_command(toolchain: Path, files: Sequence[str]) -> list[str]:
    """ESLint over *files*, which its base path, the working directory, must hold."""
    return [_bin(toolchain, "eslint"), "--config", str(toolchain / "eslint.config.js"), *files]


def _vitest_command(toolchain: Path) -> list[str]:
    """Vitest with its working directory as the root, where the config's glob reads from."""
    return [
        _bin(toolchain, "vitest"),
        "run",
        "--root",
        ".",
        "--config",
        str(toolchain / "vitest.config.js"),
    ]


def _bin(toolchain: Path, tool: str) -> str:
    return str(toolchain / "node_modules" / ".bin" / tool)


def _tool_violation(
    app_dir: Path, rule: str, argv: Sequence[str], env: Mapping[str, str]
) -> list[Violation]:
    """One JS tool over one application: nothing when it passes, its output when not."""
    done = _run(argv, app_dir, env)
    if done.returncode == 0:
        return []
    return [
        Violation(
            app_dir.name,
            "",
            rule,
            f"{rule} exited {done.returncode}:\n{done.stdout}{done.stderr}".rstrip(),
            source=f"{APPS_DIRNAME}/{app_dir.name}",
        )
    ]


def _toolchain() -> Path:
    """The pinned toolchain's directory, installed into the cache on first use.

    Installed beside a copy of the configs, because a config's own imports resolve from
    where the file is. A directory is keyed by what it was installed from and moved into
    place whole, so one that exists is complete.

    Raises:
        _ToolchainUnavailable: It is not cached and cannot be fetched.
    """
    sources = sorted(path for path in JS_CHECK_DIR.iterdir() if path.is_file())
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    cache = _cache_root()
    target = cache / digest.hexdigest()[:16]
    if target.is_dir():
        return target
    if shutil.which("npm") is None:
        raise _ToolchainUnavailable(f"npm is required to install the JS toolchain into {target}")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".install-", dir=cache))
    except OSError as error:
        raise _ToolchainUnavailable(f"cannot write the JS toolchain cache: {error}") from error
    try:
        for path in sources:
            shutil.copy2(path, staging / path.name)
        done = _run(
            ("npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"), staging, dict(os.environ)
        )
        if done.returncode != 0:
            raise _ToolchainUnavailable(
                "The JS toolchain cannot be fetched and none is cached; run once with network "
                f"access:\n{done.stdout}{done.stderr}".rstrip()
            )
        try:
            staging.replace(target)
        except OSError:
            if not target.is_dir():  # Otherwise another run finished the same install first.
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def _cache_root() -> Path:
    """Where installed toolchains live: the user's cache, by the XDG rule."""
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "osprey" / "js_check"


def _run(
    argv: Sequence[str], cwd: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a JS tool in *cwd* with exactly *env*, its output captured for the report."""
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
