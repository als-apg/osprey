"""Shared fixtures for hook tests.

Provides a hook runner that executes hook scripts as subprocesses (matching
how Claude Code invokes them), plus config file factories.

Hook subprocesses run with a *curated* environment rather than a copy of
``os.environ``: only the interpreter/OS essentials and the switches the hooks
themselves read are forwarded, so a hook run cannot pick up ``OSPREY_CONFIG``,
``CONFIG_FILE`` or similar state leaked into the process environment by a test
that ran earlier in the same worker.

Unit tests that need a hook's *helpers* rather than its end-to-end behaviour
import the module in-process through :func:`import_hook`, which contains the
first of the two side effects that come with doing so: the ``sys.path`` entry
most hooks insert for their own directory, stripped again on the way out. The
second — ``osprey_hook_log``'s process-wide config memos — outlives any single
import, so an autouse fixture blanks it around every test instead.
"""

import importlib
import json
import os
import subprocess
import sys
from itertools import count
from pathlib import Path

import pytest
import yaml

HOOKS_DIR = (
    Path(__file__).parents[2] / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
)

# Interpreter and OS essentials the subprocess needs to start at all.
_SYSTEM_VARS = (
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONHASHSEED",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    # Subprocess coverage. pytest-cov 7 dropped its own subprocess measurement;
    # coverage.py's ``patch = ["subprocess"]`` now sets COVERAGE_PROCESS_CONFIG,
    # which the site-packages .pth bootstrap reads to start coverage in the child.
    # COVERAGE_PROCESS_START is the older switch for driving subprocess coverage
    # manually. COVERAGE_CORE names the measurement core, so forwarding it keeps
    # the child measuring with the same tracer as the parent.
    "COVERAGE_PROCESS_CONFIG",
    "COVERAGE_PROCESS_START",
    "COVERAGE_CORE",
)

# Hook behaviour switches. Tests that need one set it with ``monkeypatch.setenv``
# (reverted at teardown), so forwarding them cannot leak across tests. Config
# location vars are deliberately absent: they are set only from an explicit
# ``config_path`` argument, never inherited.
_HOOK_VARS = (
    "CLAUDE_PROJECT_DIR",
    "OSPREY_HOOK_CONFIG",
    "OSPREY_HOOK_DEBUG",
    "OSPREY_DISPATCH_RUN",
    "OSPREY_WEB_PORT",
    "BLUESKY_BRIDGE_URL",
)

_hook_config_seq = count()

# The hooks ship as data inside the templates tree, but that tree is reachable
# as a namespace package, so a hook can be imported by dotted path without
# putting its directory on ``sys.path`` at import time.
_HOOK_PACKAGE = "osprey.templates.claude_code.claude.hooks"

#: Hooks whose module body only defines things (the work sits behind a
#: ``__main__`` guard), so importing one in-process is inert.
_SEAM_IMPORTABLE_HOOKS = frozenset(
    {
        "osprey_approval",
        "osprey_config_drift",
        "osprey_error_guidance",
        "osprey_focus_validate",
        "osprey_hook_log",
        "osprey_limits",
        "osprey_memory_guard",
        "osprey_notebook_update",
        "osprey_panels_context",
        "osprey_writes_check",
    }
)

#: Hooks that must never be imported in-process, and why.
_UNIMPORTABLE_HOOKS = {
    "osprey_cf_feedback_capture": (
        "its module body runs the hook end to end and finishes with a "
        "module-level sys.exit(0), which would tear down the importing test"
    ),
}


def import_hook(name):
    """Import a hook module in-process and hand back the module object.

    For tests that exercise a hook's helper functions directly; anything that
    depends on hook *behaviour* should go through :func:`hook_runner` instead,
    which runs the script the way Claude Code does.

    ``name`` is the module name, with or without the ``.py`` suffix. Only the
    hooks in :data:`_SEAM_IMPORTABLE_HOOKS` may be imported; asking for one that
    is not on the list raises :class:`ValueError` rather than executing it.

    The import happens here rather than at module scope for two reasons. Most
    hooks prepend their own directory to ``sys.path`` so they can import
    ``osprey_hook_log`` as a sibling — that entry is removed again on the way
    out, since leaving it there lets any later ``import osprey_hook_log``
    anywhere in the worker resolve to the template copy. The strip is
    unconditional, so it is simply a no-op for the few that never add it. And
    the audit in ``tests/infrastructure/test_import_time_audit.py`` forbids
    exactly this kind of import-time mutation in a test module, which is the
    same rule stated as a test.

    The module is cached by :mod:`importlib` for the life of the worker, so every
    call hands back the *same* object and anything written onto it outlives the
    test that wrote it. Mutate a hook module only through ``monkeypatch.setattr``,
    which puts the original back at teardown. The sole state restored for you is
    ``osprey_hook_log``'s three config memos, blanked around every test by the
    :func:`_reset_hook_log_config_caches` autouse fixture.

    ``import_hook("osprey_hook_log")`` carries one trap. It returns the copy under
    :data:`_HOOK_PACKAGE`, which is *not* the top-level ``osprey_hook_log`` that a
    hook binds when its own ``sys.path`` shim imports the sibling. Those are two
    module objects with two sets of globals, so patching the one returned here
    does not change what a hook sees. Exercise logging behaviour that has to cross
    that boundary through :func:`hook_runner`.
    """
    module_name = name[:-3] if name.endswith(".py") else name
    if module_name in _UNIMPORTABLE_HOOKS:
        raise ValueError(
            f"{module_name} cannot be imported in-process: "
            f"{_UNIMPORTABLE_HOOKS[module_name]}. Run it with the hook_runner "
            f"fixture instead."
        )
    if module_name not in _SEAM_IMPORTABLE_HOOKS:
        raise ValueError(
            f"Unknown hook {name!r}. Seam-importable hooks: "
            f"{', '.join(sorted(_SEAM_IMPORTABLE_HOOKS))}."
        )
    try:
        return importlib.import_module(f"{_HOOK_PACKAGE}.{module_name}")
    finally:
        hooks_dir = os.path.abspath(str(HOOKS_DIR))
        sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != hooks_dir]


@pytest.fixture
def hook_module():
    """Give a test :func:`import_hook` without importing anything itself."""
    return import_hook


#: Module-level memos in ``osprey_hook_log``, all of them ``None`` when cold.
_HOOK_LOG_CACHES = ("_hook_config_cache", "_osprey_config_cache", "_debug_from_config")


def _reset_hook_log_caches():
    """Blank the config memos on every loaded copy of ``osprey_hook_log``.

    There are two: the one under the package path, and the top-level
    ``osprey_hook_log`` a hook's own ``sys.path`` shim creates. They are separate
    module objects with separate caches, and the hooks bind the top-level one.
    """
    for module_name in (f"{_HOOK_PACKAGE}.osprey_hook_log", "osprey_hook_log"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for cache_name in _HOOK_LOG_CACHES:
            setattr(module, cache_name, None)


@pytest.fixture(autouse=True)
def _reset_hook_log_config_caches():
    """Isolate every test from ``osprey_hook_log``'s process-wide config memos.

    ``load_hook_config``, ``load_osprey_config`` and the debug switch each read
    their file once and keep the answer in a module global for the life of the
    process. Any test that imports a hook in-process therefore pins whatever
    config was visible on the first call — a later test pointing
    ``OSPREY_HOOK_CONFIG`` at its own file would silently get the earlier one,
    and a test that expects the shipped defaults would get a fixture's config.
    Subprocess hook runs are unaffected (they start cold), but they share the
    module with the in-process tests, so the reset is unconditional.
    """
    _reset_hook_log_caches()
    yield
    _reset_hook_log_caches()


@pytest.fixture
def hook_home(tmp_path_factory):
    """Throwaway ``$HOME`` for hook tests.

    Hooks resolve ``~/.claude/...`` from ``$HOME``; tests build their expected
    paths from this fixture rather than from the developer's real home.
    """
    return tmp_path_factory.mktemp("hook-home")


@pytest.fixture(autouse=True)
def _isolated_home(hook_home, monkeypatch):
    """Point ``$HOME`` at :func:`hook_home` for this test and its subprocesses.

    Leak guarded: without this, ``Path.home()`` in the test process and in the
    hook subprocess both resolve to the real home directory, which the memory
    guard hook is asked to write into. ``monkeypatch`` restores the previous
    values after the test.
    """
    monkeypatch.setenv("HOME", str(hook_home))
    monkeypatch.setenv("USERPROFILE", str(hook_home))
    yield


def _curated_env(config_path=None, hook_config_path=None):
    """Build the subprocess environment from the allowlists above."""
    env = {name: os.environ[name] for name in _SYSTEM_VARS + _HOOK_VARS if name in os.environ}
    if config_path:
        env["OSPREY_CONFIG"] = str(config_path)
        # Also set CONFIG_FILE so osprey.utils.config picks it up.
        env["CONFIG_FILE"] = str(config_path)
    if hook_config_path is not None:
        env["OSPREY_HOOK_CONFIG"] = str(hook_config_path)
    return env


def _write_hook_config(tmp_path, hook_config):
    """Serialise a hook_config dict into a per-test temp file."""
    path = tmp_path / f"hook_config_{next(_hook_config_seq)}.json"
    path.write_text(json.dumps(hook_config))
    return path


@pytest.fixture
def hook_runner(tmp_path):
    """Factory to run hook scripts as subprocesses.

    Mirrors the real Claude Code hook execution: stdin receives JSON with
    tool_name and tool_input, stdout receives JSON output (or empty for allow).
    """

    def run(
        hook_name,
        tool_name,
        tool_input,
        config_path=None,
        cwd=None,
        tool_response=None,
        hook_input_extra=None,
        hook_config=None,
    ):
        hook_script = HOOKS_DIR / hook_name
        payload = {
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        if tool_response is not None:
            payload["tool_response"] = tool_response
        if hook_input_extra:
            payload.update(hook_input_extra)
        stdin_data = json.dumps(payload)

        hook_config_path = None
        if hook_config is not None:
            hook_config_path = _write_hook_config(tmp_path, hook_config)
        env = _curated_env(config_path, hook_config_path)

        result = subprocess.run(
            [sys.executable, str(hook_script)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        assert result.returncode == 0, f"Hook failed (exit {result.returncode}): {result.stderr}"
        stdout = result.stdout.strip()
        if not stdout:
            return None  # Hook allowed (no output = pass through)
        # Find the JSON object in stdout (skip any log/warning lines)
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        # Try the full stdout as JSON
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return None  # Non-JSON output = treat as pass through

    return run


@pytest.fixture
def hook_runner_raw(tmp_path):
    """Factory to run hook scripts without asserting returncode.

    Same as hook_runner but returns a (returncode, stdout, stderr) tuple
    instead of parsing the JSON output. Used for crash resilience tests.
    """

    def run(
        hook_name,
        tool_name,
        tool_input,
        config_path=None,
        cwd=None,
        tool_response=None,
        hook_input_extra=None,
        stdin_override=None,
        hook_config=None,
    ):
        hook_script = HOOKS_DIR / hook_name
        if stdin_override is not None:
            stdin_data = stdin_override
        else:
            payload = {
                "tool_name": tool_name,
                "tool_input": tool_input,
            }
            if tool_response is not None:
                payload["tool_response"] = tool_response
            if hook_input_extra:
                payload.update(hook_input_extra)
            stdin_data = json.dumps(payload)

        hook_config_path = None
        if hook_config is not None:
            hook_config_path = _write_hook_config(tmp_path, hook_config)
        env = _curated_env(config_path, hook_config_path)

        result = subprocess.run(
            [sys.executable, str(hook_script)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        return result.returncode, result.stdout, result.stderr

    return run


@pytest.fixture
def make_config(tmp_path):
    """Factory for creating test config.yml files from dicts."""

    def _make(config_dict):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config_dict))
        return config_path

    return _make
