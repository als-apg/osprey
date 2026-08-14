"""Importing an ``osprey`` module must not rewrite ``os.environ`` from ``.env``.

The environment counterpart to ``test_logging_guard.py``: importing the
framework leaves the host application's logging alone, and it leaves the host
application's *environment* alone too. Loading ``.env`` is something a process
entry point does deliberately — the CLI, an MCP server's ``main()``, a Claude
Code launch path — never something an import does on the way past.

Two independent writers used to break that, and each has its own case below:

* OSPREY's own config loader, reached because ``osprey/utils/__init__.py``
  imports ``config``, which built the default ``Config`` at module scope.
  It reads ``./.env`` — so which file wins depended on the working directory.
* LiteLLM, which calls a bare ``dotenv.load_dotenv()`` at import when
  ``LITELLM_MODE`` is unset. That form searches *upward*, so it needs no
  ``.env`` in the working directory at all — a worktree inherited the parent
  checkout's. ``osprey.models.providers.litellm_adapter`` pins the mode to
  disable it.

Every case runs in a subprocess. Import side effects fire once per interpreter,
so they cannot be re-observed in-process, and a leaked ``os.environ`` write
would outlive ``monkeypatch`` and contaminate the rest of the session.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Deliberately not a name anything reads. A real key (TZ, an API key) would let
# a passing test mean "nothing consumed it yet" rather than "nothing loaded it".
SENTINEL = "OSPREY_IMPORT_HYGIENE_SENTINEL"


def _write_dotenv(directory: Path) -> Path:
    """Write a ``.env`` carrying the sentinel, plus filler that looks routine."""
    directory.mkdir(parents=True, exist_ok=True)
    env_file = directory / ".env"
    env_file.write_text(
        f"{SENTINEL}=leaked\nTZ=Antarctica/Troll\nOSPREY_FAKE_API_KEY=sk-not-a-real-key\n"
    )
    return env_file


def _probe(snippet: str, *, cwd: Path) -> str | None:
    """Run ``snippet`` in a fresh interpreter under ``cwd``; return the sentinel it saw.

    The child must not inherit the sentinel, or every assertion below passes
    for the wrong reason.
    """
    env = {k: v for k, v in os.environ.items() if k != SENTINEL}
    code = textwrap.dedent(snippet) + textwrap.dedent(
        f"""
        import json as _json, os as _os
        print("SENTINEL_RESULT " + _json.dumps(_os.environ.get({SENTINEL!r})))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"probe failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    for line in result.stdout.splitlines():
        if line.startswith("SENTINEL_RESULT "):
            return json.loads(line.removeprefix("SENTINEL_RESULT "))
    raise AssertionError(f"probe printed no result\n--- stdout ---\n{result.stdout}")


# ===================================================================
# Non-vacuity: the fixture .env is real, readable, and loadable
# ===================================================================


def test_explicit_load_does_publish_the_sentinel(tmp_path):
    """The control. Without this, every assertion below could pass on a typo.

    Also pins the other half of the contract: entry points that *do* call
    ``load_project_dotenv()`` still get the full ``.env`` passthrough, which
    feeds the ``${VAR}`` references Claude Code expands in ``.mcp.json``.
    """
    _write_dotenv(tmp_path)

    seen = _probe(
        """
        from osprey.utils.config import load_project_dotenv
        load_project_dotenv()
        """,
        cwd=tmp_path,
    )

    assert seen == "leaked"


def test_explicit_load_overrides_an_existing_value(tmp_path):
    """``override=True`` is load-bearing for API keys: ``.env`` beats a stale export."""
    _write_dotenv(tmp_path)

    seen = _probe(
        """
        import os
        os.environ["OSPREY_IMPORT_HYGIENE_SENTINEL"] = "from-the-shell"
        from osprey.utils.config import load_project_dotenv
        load_project_dotenv()
        """,
        cwd=tmp_path,
    )

    assert seen == "leaked"


def test_explicit_load_records_what_it_overrode(tmp_path, monkeypatch):
    """The shell's own differing value survives the override, by name and value.

    The deploy path needs it: compose gives a shell export precedence over
    ``--env-file``, so the shadow preflight and the env handed to compose both
    reconstruct the shell from this record. A key the shell never set, or set
    to the file's own value, is not recorded — nothing was shadowed.
    """
    import osprey.utils.config as config

    (tmp_path / ".env").write_text(
        "OSPREY_HYGIENE_DIFFERS=from-the-file\n"
        "OSPREY_HYGIENE_AGREES=same-both-sides\n"
        "OSPREY_HYGIENE_FILE_ONLY=file-only\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OSPREY_HYGIENE_DIFFERS", "from-the-shell")
    monkeypatch.setenv("OSPREY_HYGIENE_AGREES", "same-both-sides")
    monkeypatch.setattr(config, "_dotenv_shell_overrides", {})
    monkeypatch.chdir(tmp_path)

    import os

    try:
        config.load_project_dotenv()

        assert config.dotenv_shell_overrides() == {"OSPREY_HYGIENE_DIFFERS": "from-the-shell"}
        # The override itself still happened — .env stays the in-process winner.
        assert os.environ["OSPREY_HYGIENE_DIFFERS"] == "from-the-file"

        # The load runs more than once per process; after the first, the
        # environment already matches the file — the record of what the SHELL
        # held must survive the repeat, not be erased by it.
        config.load_project_dotenv()
        assert config.dotenv_shell_overrides() == {"OSPREY_HYGIENE_DIFFERS": "from-the-shell"}
    finally:
        # load_dotenv wrote the file-only key straight into os.environ, outside
        # monkeypatch's bookkeeping; the differing/agreeing keys are restored by
        # the setenv registrations above.
        os.environ.pop("OSPREY_HYGIENE_FILE_ONLY", None)


# ===================================================================
# Importing publishes nothing
# ===================================================================


@pytest.mark.parametrize(
    "module",
    [
        # The config machinery itself, and the package __init__ that pulls it in.
        "osprey.utils.config",
        "osprey.utils",
        # A module-level `logger = get_logger(...)` site. ~70 of these exist;
        # each one used to build a Config to resolve a colour nothing read.
        "osprey.connectors.archiver.mock_archiver_connector",
        # A connector an application imports without asking for any config.
        "osprey.connectors.control_system.mock_connector",
    ],
)
def test_import_does_not_publish_dotenv(tmp_path, module):
    _write_dotenv(tmp_path)

    seen = _probe(f"import {module}", cwd=tmp_path)

    assert seen is None, f"importing {module} published .env into os.environ"


def test_import_does_not_clobber_a_caller_value(tmp_path):
    """The monkeypatch failure mode: a later import must not undo a caller's write."""
    _write_dotenv(tmp_path)

    seen = _probe(
        """
        import os
        os.environ["OSPREY_IMPORT_HYGIENE_SENTINEL"] = "set-by-the-caller"
        import osprey.utils.config  # noqa: F401
        """,
        cwd=tmp_path,
    )

    assert seen == "set-by-the-caller"


# ===================================================================
# The upward walk: a .env the process never chose
# ===================================================================


def test_provider_import_does_not_walk_up_to_a_parent_dotenv(tmp_path):
    """LiteLLM's bare ``load_dotenv()`` searches parent directories.

    This is the worktree case: ``.claude/worktrees/<name>/`` has no ``.env``,
    the walk reaches the parent checkout's, and importing a provider module
    absorbs it. The child's cwd deliberately has no ``.env`` of its own, so
    only an upward search can find one.
    """
    _write_dotenv(tmp_path)
    child = tmp_path / "nested" / "workdir"
    child.mkdir(parents=True)
    assert not (child / ".env").exists()

    seen = _probe("import osprey.models.providers.litellm_adapter", cwd=child)

    assert seen is None, "a parent directory's .env reached os.environ via the provider import"


def test_litellm_mode_is_pinned_but_respects_an_explicit_setting():
    """The mechanism, and its escape hatch. Pinning gates nothing else in LiteLLM."""
    env = {k: v for k, v in os.environ.items() if k != "LITELLM_MODE"}
    probe = (
        "import os; import osprey.models.providers.litellm_adapter; "
        "print('MODE ' + os.environ['LITELLM_MODE'])"
    )

    default = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=180
    )
    assert default.returncode == 0, default.stderr
    assert "MODE PRODUCTION" in default.stdout

    explicit = subprocess.run(
        [sys.executable, "-c", probe],
        env={**env, "LITELLM_MODE": "DEV"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert explicit.returncode == 0, explicit.stderr
    assert "MODE DEV" in explicit.stdout


# ===================================================================
# The CLI is an entry point, so it does load .env
# ===================================================================


def test_cli_entry_point_loads_dotenv(tmp_path):
    """`osprey <command>` must still see `.env` from its first line.

    Nothing else populates the environment for a CLI process, since import must
    not, so a subcommand reading ``os.environ`` directly — before anything
    builds a config — depends on this. The invocation runs inside the probe's
    subprocess because the CLI writes straight to ``os.environ``, which would
    otherwise outlive the test.
    """
    _write_dotenv(tmp_path)

    seen = _probe(
        """
        from click.testing import CliRunner
        from osprey.cli.main import cli
        # The group callback runs before the subcommand is dispatched. It may
        # then fail on the missing config.yml — irrelevant, .env loads first,
        # and that ordering is the point.
        CliRunner().invoke(cli, ["config", "--help"])
        """,
        cwd=tmp_path,
    )

    assert seen == "leaked"
