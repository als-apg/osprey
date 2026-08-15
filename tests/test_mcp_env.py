"""The env chain as an MCP server sees it at startup.

:func:`osprey.mcp_env.load_dotenv_from_project` is the one loader in OSPREY
that reads the chain with ``override=False`` — first file to supply a key wins,
and anything the shell already exports outranks both files. The chain contract
is the same as everywhere else (``.env.shared`` defaults below the host-local
``.env``), so realizing it here means loading the files in the *opposite* order
from a compose ``env_file:`` list. These tests pin the contract rather than the
order, so the day someone "corrects" the file order to match compose, the
conflict-key test is what fails.

The other half of this loader — which directory the chain is read from, across
the repo/render split — is pinned in ``tests/registry/test_mcp_env_rendering.py``
alongside what the registry renders into ``.mcp.json``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osprey.mcp_env import load_dotenv_from_project

#: Keys the fixtures below write into the chain files and read back out of the
#: process environment. Prefixed so a leak is obvious in a failure message.
LOCAL_KEY = "OSPREY_TEST_CHAIN_LOCAL"
SHARED_KEY = "OSPREY_TEST_CHAIN_SHARED"
CONFLICT_KEY = "OSPREY_TEST_CHAIN_CONFLICT"


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot and restore the whole environment around every test.

    The loader writes into ``os.environ`` and — by design — *pops* every
    empty-string variable it finds there, so nothing narrower than a full
    snapshot puts the process back the way it was found.
    """
    saved = os.environ.copy()
    os.environ.pop("OSPREY_CONFIG", None)
    for key in (LOCAL_KEY, SHARED_KEY, CONFLICT_KEY):
        os.environ.pop(key, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _write_chain(root: Path, *, shared: str | None = None, local: str | None = None) -> None:
    """Write the requested chain members under *root*; omitted ones stay absent."""
    root.mkdir(parents=True, exist_ok=True)
    if shared is not None:
        (root / ".env.shared").write_text(shared, encoding="utf-8")
    if local is not None:
        (root / ".env").write_text(local, encoding="utf-8")


def _repo_with_render(root: Path) -> Path:
    """Build a repo with a rendered config and return that config's path."""
    build = root / "build"
    build.mkdir(parents=True, exist_ok=True)
    config = build / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    return config


def _load_at(root: Path) -> None:
    """Point ``OSPREY_CONFIG`` at *root*'s render and run the loader."""
    os.environ["OSPREY_CONFIG"] = str(_repo_with_render(root))
    load_dotenv_from_project()


class TestChainPrecedence:
    """Ascending ``.env.shared`` < ``.env`` < shell, under first-wins loading."""

    def test_a_key_set_in_both_files_resolves_to_the_local_value(self, tmp_path) -> None:
        """The conflict key: local wins, which is why ``.env`` is loaded first."""
        _write_chain(
            tmp_path,
            shared=f"{CONFLICT_KEY}=from-shared\n",
            local=f"{CONFLICT_KEY}=from-local\n",
        )

        _load_at(tmp_path)

        assert os.environ[CONFLICT_KEY] == "from-local"

    def test_a_key_only_the_shared_file_sets_is_still_loaded(self, tmp_path) -> None:
        """Local-wins is per key — it does not shadow the defaults file wholesale."""
        _write_chain(
            tmp_path,
            shared=f"{SHARED_KEY}=default\n",
            local=f"{LOCAL_KEY}=local\n",
        )

        _load_at(tmp_path)

        assert os.environ[SHARED_KEY] == "default"
        assert os.environ[LOCAL_KEY] == "local"

    def test_a_shell_value_outranks_the_whole_chain(self, tmp_path) -> None:
        """``override=False``: an exported variable is the top of the chain."""
        _write_chain(
            tmp_path,
            shared=f"{CONFLICT_KEY}=from-shared\n",
            local=f"{CONFLICT_KEY}=from-local\n",
        )
        os.environ[CONFLICT_KEY] = "from-shell"

        _load_at(tmp_path)

        assert os.environ[CONFLICT_KEY] == "from-shell"

    def test_an_empty_string_variable_is_filled_from_the_chain(self, tmp_path) -> None:
        """Claude Code's ``${VAR:-}`` passthrough must read as absent, not as a value."""
        _write_chain(tmp_path, shared=f"{SHARED_KEY}=default\n", local=f"{LOCAL_KEY}=local\n")
        os.environ[SHARED_KEY] = ""
        os.environ[LOCAL_KEY] = ""

        _load_at(tmp_path)

        assert os.environ[SHARED_KEY] == "default"
        assert os.environ[LOCAL_KEY] == "local"


class TestChainResolution:
    """Which directory the chain is read from, and what counts as a chain."""

    def test_a_shared_file_alone_is_enough(self, tmp_path) -> None:
        """A deployment carrying only committed defaults is a valid shape."""
        _write_chain(tmp_path, shared=f"{SHARED_KEY}=default\n")

        _load_at(tmp_path)

        assert os.environ[SHARED_KEY] == "default"

    def test_the_repo_root_chain_wins_and_is_not_stitched_to_the_render(self, tmp_path) -> None:
        """One directory supplies the whole chain; a file in the render is never merged in."""
        config = _repo_with_render(tmp_path)
        _write_chain(tmp_path, local=f"{LOCAL_KEY}=repo-root\n")
        _write_chain(
            tmp_path / "build",
            shared=f"{SHARED_KEY}=stale-render\n",
            local=f"{LOCAL_KEY}=stale-render\n",
        )
        os.environ["OSPREY_CONFIG"] = str(config)

        load_dotenv_from_project()

        assert os.environ[LOCAL_KEY] == "repo-root"
        assert SHARED_KEY not in os.environ

    def test_falls_back_to_the_config_directory(self, tmp_path) -> None:
        """The container layout: the project directory *is* the render."""
        config = tmp_path / "config.yml"
        config.write_text("{}\n", encoding="utf-8")
        _write_chain(
            tmp_path,
            shared=f"{CONFLICT_KEY}=from-shared\n",
            local=f"{CONFLICT_KEY}=from-local\n",
        )
        os.environ["OSPREY_CONFIG"] = str(config)

        load_dotenv_from_project()

        assert os.environ[CONFLICT_KEY] == "from-local"

    def test_without_osprey_config_it_reads_the_working_directory(
        self, tmp_path, monkeypatch
    ) -> None:
        _write_chain(
            tmp_path,
            shared=f"{SHARED_KEY}=default\n",
            local=f"{CONFLICT_KEY}=from-local\n",
        )
        monkeypatch.chdir(tmp_path)

        load_dotenv_from_project()

        assert os.environ[SHARED_KEY] == "default"
        assert os.environ[CONFLICT_KEY] == "from-local"

    def test_no_chain_files_at_all_is_not_an_error(self, tmp_path) -> None:
        _load_at(tmp_path)  # must not raise

        assert SHARED_KEY not in os.environ
        assert LOCAL_KEY not in os.environ


class TestLocalOnlyLayoutIsUnchanged:
    """A deployment with no ``.env.shared`` behaves exactly as it did before."""

    def test_the_repo_root_env_is_loaded_over_one_in_the_render(self, tmp_path) -> None:
        config = _repo_with_render(tmp_path)
        _write_chain(tmp_path, local=f"{CONFLICT_KEY}=repo-root\n")
        _write_chain(tmp_path / "build", local=f"{CONFLICT_KEY}=stale-render\n")
        os.environ["OSPREY_CONFIG"] = str(config)

        load_dotenv_from_project()

        assert os.environ[CONFLICT_KEY] == "repo-root"

    def test_a_shell_value_still_outranks_it(self, tmp_path) -> None:
        _write_chain(tmp_path, local=f"{CONFLICT_KEY}=from-local\n")
        os.environ[CONFLICT_KEY] = "from-shell"

        _load_at(tmp_path)

        assert os.environ[CONFLICT_KEY] == "from-shell"
