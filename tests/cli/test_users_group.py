"""Tests for the ``osprey users`` web-terminal roster group.

The group is a thin CLI over ``osprey.deployment.web_terminals``: those engines
do the work, called with their own signatures. The verbs take no
``--project``/``--config`` — they walk up to the enclosing deployment repo and
act through its rendered ``build/config.yml``.

So these tests pin two things. First the seam: which file each verb hands the
engine, what the engine is handed alongside it, and which directory it runs in.
Second the gates the CLI must not loosen — the typed
confirmations guarding volume destruction are exercised through the CLI against
the real lifecycle engine, with the container runtime mocked (no container or
volume is ever created or removed here).
"""

from __future__ import annotations

import logging
import re
import stat
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.users_cmd import users
from osprey.deployment.compose_generator import resolve_user_volume_names
from osprey.deployment.web_terminals import lifecycle

#: Every verb the group ships, mapped to the exact set of option spellings it
#: accepts. Pinned as a whole rather than flag-by-flag: a verb must carry the
#: options it acts on and no others, and the retired ``--project``/``--config``
#: pair must not survive the move in any verb.
VERB_OPTIONS = {
    "remove": {"--repo", "--archive", "--purge", "--yes", "-y"},
    "prune": {"--repo", "--archive", "--purge", "--yes", "-y", "--dry-run"},
    "seed": {"--repo"},
    "passwd": {"--repo"},
    "env-production": {"--repo", "--env-file", "--output", "-o"},
}


#: Roster config the engines read. Mirrors the shape a rendered
#: ``build/config.yml`` has for a multi-user deployment: every field
#: ``decommission_user``/``prune_users`` touch, and nothing else.
def _config(users_list, *, project_name="demo-project", facility_prefix="dls"):
    return {
        "project_name": project_name,
        "facility": {"name": "Demo Light Source", "prefix": facility_prefix, "timezone": "UTC"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 8080,
                "web_base_port": 9000,
                "artifact_base_port": 9100,
                "ariel_base_port": 9200,
                "lattice_base_port": 9300,
                "users": users_list,
            }
        },
    }


#: A config exercising the ``.env.production`` render: one provider credential
#: that belongs in a web terminal, one build-time token that never does.
ENV_PRODUCTION_CONFIG = textwrap.dedent(
    """
    project_name: demo-project
    facility:
      name: Demo Light Source
      prefix: dls
      timezone: UTC
    llm:
      provider: cborg
      api_key_env_var: CBORG_API_KEY
    registry:
      url: registry.example.org
      token_env_var: TEST_REGISTRY_TOKEN
    modules:
      web_terminals:
        enabled: true
        image_source: local
    """
)


@pytest.fixture
def cli_runner():
    return CliRunner()


#: A profile that spells the roster the way the presets do: ONE literal dotted
#: key holding the whole ``modules.web_terminals`` subtree, with the roster
#: nested inside it and each entry carrying its persona.
PROFILE_WITH_ROSTER = textwrap.dedent(
    """
    preset: control-assistant
    config:
      facility.prefix: dls
      modules.web_terminals:
        enabled: true
        default_persona: readonly
        # Who gets a web terminal. `index` pins each user's ports.
        users:
          - name: alice
            index: 0
            persona: readonly
          - name: bob
            index: 1
            persona: readwrite
    """
)


def _make_repo(
    tmp_path: Path,
    config: dict | str | None = None,
    *,
    name: str = "repo",
    profile: str = "preset: control-assistant\n",
) -> Path:
    """Create a three-zone deployment repo: ``profile.yml`` + rendered ``build/``."""
    repo_root = tmp_path / name
    (repo_root / "build").mkdir(parents=True)
    (repo_root / "profile.yml").write_text(profile, encoding="utf-8")

    if config is None:
        config = _config([{"name": "alice", "index": 0}])
    rendered = config if isinstance(config, str) else yaml.safe_dump(config)
    (repo_root / "build" / "config.yml").write_text(rendered, encoding="utf-8")
    return repo_root.resolve()


def _fake_web_terminals(**attrs):
    """Patch the lazily imported engine modules with recording stand-ins.

    Returns ``(patcher, mocks)``: the engines are imported inside the command
    bodies, so replacing the modules in ``sys.modules`` is what intercepts them
    without importing the real deploy stack.
    """
    pkg = types.ModuleType("osprey.deployment.web_terminals")
    fake_lifecycle = types.ModuleType("osprey.deployment.web_terminals.lifecycle")
    fake_seeding = types.ModuleType("osprey.deployment.web_terminals.seeding")
    mocks = {}
    for name in ("decommission_user", "prune_users", "rotate_user_password"):
        mocks[name] = MagicMock()
        setattr(fake_lifecycle, name, mocks[name])
    mocks["seed_web_terminals"] = MagicMock()
    fake_seeding.seed_web_terminals = mocks["seed_web_terminals"]
    for name, value in attrs.items():
        setattr(fake_lifecycle, name, value)

    patcher = patch.dict(
        sys.modules,
        {
            "osprey.deployment.web_terminals": pkg,
            "osprey.deployment.web_terminals.lifecycle": fake_lifecycle,
            "osprey.deployment.web_terminals.seeding": fake_seeding,
        },
    )
    return patcher, mocks


def _flat(output: str) -> str:
    """Collapse Rich's line wrapping so help text can be matched as one string.

    ANSI style codes are stripped first: a wrap point re-opens the style at the
    start of the next line, which would otherwise glue an escape sequence onto
    the middle of a matched phrase.
    """
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).split())


@pytest.fixture
def fake_runtime(monkeypatch):
    """Patch the container runtime away and record every argv the engine emits."""
    calls: list[list[str]] = []
    listing: dict[str, list[str]] = {"containers": [], "volumes": []}

    def _fake_run(argv, capture_output=True, text=True, env=None, check=False):
        calls.append(list(argv))
        if argv[1:3] == ["ps", "-a"]:
            stdout = "\n".join(listing["containers"])
        elif argv[1:3] == ["volume", "ls"]:
            stdout = "\n".join(listing["volumes"])
        else:
            stdout = ""
        return subprocess.CompletedProcess(argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(lifecycle, "verify_runtime_is_running", lambda config=None: (True, ""))
    return calls, listing


def _roster(repo_root: Path) -> list:
    data = yaml.safe_load((repo_root / "build" / "config.yml").read_text(encoding="utf-8"))
    return data["modules"]["web_terminals"]["users"]


class TestCommandSurface:
    """What the group offers, and what it deliberately does not accept."""

    def test_group_help_lists_every_verb(self, cli_runner):
        result = cli_runner.invoke(users, ["--help"])

        assert result.exit_code == 0
        listed = _flat(result.output)
        for verb in VERB_OPTIONS:
            assert verb in listed

    @pytest.mark.parametrize("verb", sorted(VERB_OPTIONS))
    def test_verb_carries_exactly_its_own_options(self, cli_runner, verb):
        """A flag a verb does not act on must be a parse error, not a no-op."""
        result = cli_runner.invoke(users, [verb, "--help"])

        assert result.exit_code == 0
        offered = {
            token.strip(" ,")
            for token in _flat(result.output).split()
            if token.startswith("-") and token != "--help"
        }
        assert offered == VERB_OPTIONS[verb]

    @pytest.mark.parametrize("verb", sorted(VERB_OPTIONS))
    @pytest.mark.parametrize("retired", ["--project", "-p", "--config", "-c"])
    def test_retired_location_flags_are_rejected(self, cli_runner, tmp_path, verb, retired):
        """Discovery is the repo walk; there are no location flags."""
        argv = [verb, retired, str(tmp_path)]
        if verb in ("remove", "passwd"):
            argv.insert(1, "alice")

        result = cli_runner.invoke(users, argv)

        assert result.exit_code == 2
        assert "no such option" in result.output.lower()

    def test_remove_requires_a_user(self, cli_runner):
        result = cli_runner.invoke(users, ["remove", "--archive"])

        assert result.exit_code == 2
        assert "user" in result.output.lower()

    def test_archive_and_purge_are_mutually_exclusive(self, cli_runner):
        """Rejected before any repo is resolved: it is a usage mistake."""
        result = cli_runner.invoke(users, ["remove", "alice", "--archive", "--purge"])

        assert result.exit_code == 2
        assert "mutually exclusive" in result.output.lower()

    def test_prune_rejects_archive_with_purge_too(self, cli_runner):
        result = cli_runner.invoke(users, ["prune", "--archive", "--purge"])

        assert result.exit_code == 2
        assert "mutually exclusive" in result.output.lower()


class TestRepoDiscovery:
    """Every verb acts on the repo it is standing in, through its build/."""

    def test_verb_acts_on_the_enclosing_repo_from_a_subdirectory(
        self, cli_runner, tmp_path, monkeypatch
    ):
        repo_root = _make_repo(tmp_path)
        nested = repo_root / "data" / "channels"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["seed", "alice"])

        assert result.exit_code == 0
        (config_path, user), _ = mocks["seed_web_terminals"].call_args
        assert Path(config_path) == repo_root / "build" / "config.yml"
        assert user == "alice"

    def test_repo_flag_substitutes_the_walk_start(self, cli_runner, tmp_path, monkeypatch):
        """--repo moves where the walk begins; a subdirectory still finds the repo."""
        repo_root = _make_repo(tmp_path)
        nested = repo_root / "var"
        nested.mkdir()
        monkeypatch.chdir(tmp_path)

        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["seed", "--repo", str(nested)])

        assert result.exit_code == 0
        (config_path, _user), _ = mocks["seed_web_terminals"].call_args
        assert Path(config_path) == repo_root / "build" / "config.yml"

    def test_outside_a_repo_the_verb_refuses(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code != 0
        assert "profile.yml" in _flat(result.output)
        assert not mocks["seed_web_terminals"].called

    def test_a_repo_with_no_build_refuses_and_names_the_remedy(
        self, cli_runner, tmp_path, monkeypatch
    ):
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (repo_root / "profile.yml").write_text("preset: control-assistant\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)

        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["prune"])

        assert result.exit_code != 0
        flat = _flat(result.output)
        assert "No build found" in flat
        assert "osprey build" in flat
        assert not mocks["prune_users"].called

    def test_the_engine_runs_with_the_repo_root_as_the_working_directory(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """The engines resolve .env, .env.auth and archives against the cwd."""
        repo_root = _make_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        seen: dict[str, Path] = {}

        patcher, mocks = _fake_web_terminals()
        mocks["prune_users"].side_effect = lambda *a, **k: seen.update(cwd=Path.cwd().resolve())
        with patcher:
            result = cli_runner.invoke(users, ["prune", "--repo", str(repo_root)])

        assert result.exit_code == 0
        assert seen["cwd"] == repo_root

    def test_the_working_directory_is_restored_afterwards(self, cli_runner, tmp_path, monkeypatch):
        repo_root = _make_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        before = Path.cwd().resolve()

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            cli_runner.invoke(users, ["prune", "--repo", str(repo_root)])

        assert Path.cwd().resolve() == before

    def test_an_engine_failure_is_reported_without_a_traceback(
        self, cli_runner, tmp_path, monkeypatch
    ):
        repo_root = _make_repo(tmp_path)
        monkeypatch.chdir(repo_root)

        patcher, mocks = _fake_web_terminals()
        mocks["seed_web_terminals"].side_effect = RuntimeError("seed image missing")
        with patcher:
            result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code != 0
        assert "seed image missing" in _flat(result.output)


class TestEngineWiring:
    """Each verb hands the engine exactly what it handed it before the move."""

    @pytest.fixture
    def repo_root(self, tmp_path, monkeypatch):
        root = _make_repo(tmp_path)
        monkeypatch.chdir(root)
        return root

    def test_remove_targets_one_user(self, cli_runner, repo_root):
        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice", "--archive"])

        assert result.exit_code == 0
        (config_path, user), kwargs = mocks["decommission_user"].call_args
        assert Path(config_path) == repo_root / "build" / "config.yml"
        assert user == "alice"
        assert kwargs == {"archive": True, "purge": False, "assume_yes": False}

    def test_remove_forwards_yes_to_the_typed_gate(self, cli_runner, repo_root):
        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice", "--purge", "--yes"])

        assert result.exit_code == 0
        _args, kwargs = mocks["decommission_user"].call_args
        assert kwargs == {"archive": False, "purge": True, "assume_yes": True}

    def test_prune_forwards_every_flag(self, cli_runner, repo_root):
        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["prune", "--purge", "--dry-run", "-y"])

        assert result.exit_code == 0
        (config_path,), kwargs = mocks["prune_users"].call_args
        assert Path(config_path) == repo_root / "build" / "config.yml"
        assert kwargs == {"dry_run": True, "archive": False, "purge": True, "assume_yes": True}

    def test_prune_defaults_leave_the_gate_armed(self, cli_runner, repo_root):
        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["prune"])

        assert result.exit_code == 0
        _args, kwargs = mocks["prune_users"].call_args
        assert kwargs == {"dry_run": False, "archive": False, "purge": False, "assume_yes": False}

    def test_seed_without_a_user_seeds_the_whole_roster(self, cli_runner, repo_root):
        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code == 0
        (config_path, user), _kwargs = mocks["seed_web_terminals"].call_args
        assert Path(config_path) == repo_root / "build" / "config.yml"
        assert user is None


class TestPasswd:
    """The password never reaches argv, the terminal, or a log."""

    @pytest.fixture
    def repo_root(self, tmp_path, monkeypatch):
        root = _make_repo(tmp_path)
        monkeypatch.chdir(root)
        return root

    def test_the_prompted_password_reaches_the_engine_verbatim(self, cli_runner, repo_root):
        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["passwd", "alice"], input="s3cret\ns3cret\n")

        assert result.exit_code == 0
        (config_path, user, password), _kwargs = mocks["rotate_user_password"].call_args
        assert Path(config_path) == repo_root / "build" / "config.yml"
        assert (user, password) == ("alice", "s3cret")

    def test_the_password_is_never_echoed(self, cli_runner, repo_root):
        patcher, _mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["passwd", "alice"], input="s3cret\ns3cret\n")

        assert "s3cret" not in result.output

    def test_a_mistyped_confirmation_rotates_nothing(self, cli_runner, repo_root):
        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["passwd", "alice"], input="s3cret\ntypo\n")

        assert result.exit_code != 0
        assert not mocks["rotate_user_password"].called


class TestTypedGates:
    """The confirmations guarding destruction are pinned verbatim.

    These run the real ``lifecycle`` engine through the CLI with the
    container runtime mocked, so what is pinned is the gate an operator
    actually meets: the exact word it demands, and that anything else is a
    true no-op.
    """

    def test_remove_purge_demands_the_username_typed_out(
        self, cli_runner, tmp_path, monkeypatch, fake_runtime
    ):
        repo_root = _make_repo(tmp_path, _config([{"name": "alice", "index": 0}]))
        monkeypatch.chdir(repo_root)
        prompts: list[str] = []
        monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or "")

        result = cli_runner.invoke(users, ["remove", "alice", "--purge"])

        assert result.exit_code != 0
        assert prompts and "Type 'alice' to confirm" in prompts[0]

    def test_remove_purge_declined_leaves_the_roster_and_volumes_untouched(
        self, cli_runner, tmp_path, monkeypatch, fake_runtime
    ):
        calls, _listing = fake_runtime
        repo_root = _make_repo(tmp_path, _config([{"name": "alice", "index": 0}]))
        monkeypatch.chdir(repo_root)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")  # blank, non-matching

        result = cli_runner.invoke(users, ["remove", "alice", "--purge"])

        assert result.exit_code != 0
        assert _roster(repo_root) == [{"name": "alice", "index": 0}]
        assert [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"]] == []

    def test_remove_purge_is_not_confirmed_by_a_generic_yes(
        self, cli_runner, tmp_path, monkeypatch, fake_runtime
    ):
        calls, _listing = fake_runtime
        repo_root = _make_repo(tmp_path, _config([{"name": "alice", "index": 0}]))
        monkeypatch.chdir(repo_root)
        monkeypatch.setattr("builtins.input", lambda prompt="": "yes")

        result = cli_runner.invoke(users, ["remove", "alice", "--purge"])

        assert result.exit_code != 0
        assert _roster(repo_root) == [{"name": "alice", "index": 0}]
        assert [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"]] == []

    def test_prune_demands_the_word_prune(self, cli_runner, tmp_path, monkeypatch, fake_runtime):
        calls, listing = fake_runtime
        config = _config([])
        repo_root = _make_repo(tmp_path, config)
        monkeypatch.chdir(repo_root)
        eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
        listing["containers"] = ["dls-web-eve"]
        listing["volumes"] = [eve_claude, eve_agent]
        prompts: list[str] = []
        monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or "yes")

        result = cli_runner.invoke(users, ["prune", "--purge"])

        assert result.exit_code != 0
        assert prompts and "Type 'prune' to confirm" in prompts[0]
        assert [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"]] == []

    def test_archived_workspaces_land_under_var_not_in_the_source_zone(
        self, cli_runner, tmp_path, monkeypatch, fake_runtime
    ):
        """An archived volume is durable state, so it belongs in the var/ zone.

        At the repo root the tarballs would sit in the tracked source zone, one
        ``git add .`` away from committing a multi-gigabyte volume dump.
        """
        calls, _listing = fake_runtime
        config = _config([{"name": "alice", "index": 0}, {"name": "bob", "index": 1}])
        repo_root = _make_repo(tmp_path, config)
        monkeypatch.chdir(repo_root)
        alice_claude, alice_agent = resolve_user_volume_names(config, "alice")

        result = cli_runner.invoke(users, ["remove", "alice", "--archive", "--yes"])

        assert result.exit_code == 0
        archive_dir = repo_root / "var" / "web_terminal_archives"
        assert archive_dir.is_dir()
        assert not (repo_root / "web_terminal_archives").exists()

        # Each volume is streamed into that directory and only then removed.
        # Keyed off the tarball name, not the bind mount: both volumes share one
        # destination, so matching on the mount would find the same first
        # archive call twice and the per-volume ordering claim would be vacuous.
        mount = f"type=bind,source={archive_dir},destination=/to"
        for volume in (alice_claude, alice_agent):
            archived_at = next(
                i
                for i, c in enumerate(calls)
                if c[1] == "run" and f"/to/{volume}.tar.gz" in c and mount in c
            )
            removed_at = next(
                i for i, c in enumerate(calls) if c[1:3] == ["volume", "rm"] and c[3] == volume
            )
            assert archived_at < removed_at, f"{volume} must be archived before it is removed"

    def test_prune_dry_run_needs_no_confirmation_and_removes_nothing(
        self, cli_runner, tmp_path, monkeypatch, fake_runtime
    ):
        calls, listing = fake_runtime
        config = _config([])
        repo_root = _make_repo(tmp_path, config)
        monkeypatch.chdir(repo_root)
        eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
        listing["containers"] = ["dls-web-eve"]
        listing["volumes"] = [eve_claude, eve_agent]

        def _unexpected(prompt=""):
            raise AssertionError(f"dry-run must not prompt, got: {prompt}")

        monkeypatch.setattr("builtins.input", _unexpected)

        result = cli_runner.invoke(users, ["prune", "--dry-run"])

        assert result.exit_code == 0
        assert [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"]] == []


class TestProfileRosterWrite:
    """`users remove` writes the roster change into the source of truth.

    Editing only the rendered ``build/config.yml`` would be undone by the next
    ``osprey build``, which re-renders the departed user back onto the roster.
    """

    def _profile_roster(self, repo_root: Path) -> list:
        doc = yaml.safe_load((repo_root / "profile.yml").read_text(encoding="utf-8"))
        return doc["config"]["modules.web_terminals"]["users"]

    def test_the_removed_user_is_dropped_from_the_profile(self, cli_runner, tmp_path, monkeypatch):
        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice"])

        assert result.exit_code == 0
        assert [entry["name"] for entry in self._profile_roster(repo_root)] == ["bob"]

    def test_the_user_stays_gone_when_the_profile_is_re_rendered(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """The render rule itself, not just the file: the emitter's collapse."""
        from osprey.cli.build_profile_emit import _collapse_config_prefixes

        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            cli_runner.invoke(users, ["remove", "alice"])

        config_block = yaml.safe_load((repo_root / "profile.yml").read_text())["config"]
        rendered = _collapse_config_prefixes(config_block)
        roster = rendered["modules.web_terminals"]["users"]
        assert [entry["name"] for entry in roster] == ["bob"]

    def test_survivors_keep_their_persona(self, cli_runner, tmp_path, monkeypatch):
        """The trap this fix must not fall into.

        ``normalize_users`` — what the engine writes into the rendered config —
        keeps only name/index plus three optional fields and drops ``persona``.
        Writing its output back into the profile would silently re-assign every
        surviving user to ``default_persona``.
        """
        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            cli_runner.invoke(users, ["remove", "alice"])

        (bob,) = self._profile_roster(repo_root)
        assert bob["persona"] == "readwrite", "a removal must not re-assign a survivor's persona"

    def test_no_second_roster_key_is_added_beside_the_first(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """One roster in the file, edited in place.

        A `config.modules.web_terminals.users` key written beside the subtree
        renders the same config, but leaves the removed user listed in one place
        and absent in the other — and deleting the apparently-redundant key, the
        obvious tidy-up, would silently bring them back.
        """
        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            cli_runner.invoke(users, ["remove", "alice"])

        config_block = yaml.safe_load((repo_root / "profile.yml").read_text())["config"]
        roster_keys = [key for key in config_block if "web_terminals" in key]
        assert roster_keys == ["modules.web_terminals"]
        assert "alice" not in (repo_root / "profile.yml").read_text()

    def test_the_roster_comment_survives_the_edit(self, cli_runner, tmp_path, monkeypatch):
        """profile.yml is hand-edited; a roster edit must not strip its comments."""
        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            cli_runner.invoke(users, ["remove", "alice"])

        assert "# Who gets a web terminal" in (repo_root / "profile.yml").read_text()

    def test_a_bare_string_roster_has_its_indices_frozen_first(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Removing an earlier entry must not shift a later user's ports."""
        profile = textwrap.dedent(
            """
            preset: control-assistant
            config:
              modules.web_terminals:
                users:
                  - alice
                  - bob
            """
        )
        repo_root = _make_repo(tmp_path, profile=profile)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            cli_runner.invoke(users, ["remove", "alice"])

        assert self._profile_roster(repo_root) == [{"name": "bob", "index": 1}]

    def test_the_deeper_key_spelling_is_edited_in_place_too(
        self, cli_runner, tmp_path, monkeypatch
    ):
        profile = textwrap.dedent(
            """
            preset: control-assistant
            config:
              modules.web_terminals.users:
                - name: alice
                  index: 0
                - name: bob
                  index: 1
            """
        )
        repo_root = _make_repo(tmp_path, profile=profile)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice"])

        assert result.exit_code == 0
        config_block = yaml.safe_load((repo_root / "profile.yml").read_text())["config"]
        assert list(config_block) == ["modules.web_terminals.users"]
        assert [entry["name"] for entry in config_block["modules.web_terminals.users"]] == ["bob"]

    def test_a_profile_that_does_not_spell_the_roster_says_so(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Silence would leave the operator believing the removal was complete."""
        repo_root = _make_repo(tmp_path)  # profile carries no config: block
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice"])

        assert result.exit_code == 0
        flat = _flat(result.output)
        assert "does not spell the web-terminal roster" in flat
        assert "next build will restore them" in flat

    def test_the_drift_hint_is_printed_after_the_write(self, cli_runner, tmp_path, monkeypatch):
        """Same closing line `osprey set` prints: the build is now out of date."""
        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice"])

        assert "build:" in _flat(result.output)

    def test_the_engine_runs_before_the_profile_is_written(self, cli_runner, tmp_path, monkeypatch):
        """Ordering is load-bearing: the typed gate lives inside the engine.

        Writing the profile first would commit the roster edit before the
        operator is asked to type the username, so a decline would leave the
        source of truth already changed.
        """
        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)
        order: list[str] = []

        patcher, mocks = _fake_web_terminals()
        mocks["decommission_user"].side_effect = lambda *a, **k: order.append("engine")
        with patcher:
            cli_runner.invoke(users, ["remove", "alice"])

        # The profile write is observable by its result, so record it after.
        order.append("profile" if "alice" not in (repo_root / "profile.yml").read_text() else "?")
        assert order == ["engine", "profile"]

    def test_an_interrupted_removal_converges_on_re_run(self, cli_runner, tmp_path, monkeypatch):
        """Engine done, profile write missed: re-running finishes the job.

        The user is off the rendered roster (the engine got that far) but still
        on the profile's. A re-run must not fail with "not present" — it must
        complete the half-done removal.
        """
        rendered = _config([{"name": "bob", "index": 1}])  # alice already gone from build/
        repo_root = _make_repo(tmp_path, rendered, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice"])

        assert result.exit_code == 0
        # The engine is skipped: it has nothing left to act on and would raise.
        assert not mocks["decommission_user"].called
        assert [entry["name"] for entry in self._profile_roster(repo_root)] == ["bob"]
        assert "osprey users prune" in _flat(result.output)

    def test_a_failed_profile_write_reports_what_actually_happened(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """The destruction succeeded; only the bookkeeping failed.

        The generic handler would say "remove failed", which is the opposite of
        the truth and would send the operator looking for workspaces that are
        already gone.
        """
        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)
        monkeypatch.setattr(
            "osprey.cli.users_cmd._drop_user_from_profile_roster",
            lambda *a, **k: (_ for _ in ()).throw(OSError("Read-only file system")),
        )

        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice", "--purge", "--yes"])

        assert result.exit_code != 0
        flat = _flat(result.output)
        assert mocks["decommission_user"].called
        assert "WAS removed" in flat
        assert "Read-only file system" in flat
        assert "still lists alice" in flat
        assert "osprey users remove alice" in flat  # the converging remedy
        assert "by hand" in flat  # the manual remedy

    def test_an_unparseable_profile_gets_the_same_true_state_message(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """The corrupt-profile path lands in the same handler, not the generic one."""
        broken = "preset: control-assistant\nconfig:\n  modules.web_terminals:\n   users: [\n"
        repo_root = _make_repo(tmp_path, profile=broken)
        monkeypatch.chdir(repo_root)

        patcher, mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice", "--purge", "--yes"])

        assert result.exit_code != 0
        assert mocks["decommission_user"].called, "the engine ran, so the workspace is gone"
        assert "WAS removed" in _flat(result.output)

    def test_expanding_a_bare_roster_says_why_the_file_gained_lines(
        self, cli_runner, tmp_path, monkeypatch
    ):
        profile = textwrap.dedent(
            """
            preset: control-assistant
            config:
              modules.web_terminals:
                users:
                  - alice
                  - bob
            """
        )
        repo_root = _make_repo(tmp_path, profile=profile)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice"])

        assert result.exit_code == 0
        assert "explicit 'index:' values" in _flat(result.output)

    def test_an_already_explicit_roster_does_not_mention_indices(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """The note is for files that gained lines, not every removal."""
        repo_root = _make_repo(tmp_path, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, _mocks = _fake_web_terminals()
        with patcher:
            result = cli_runner.invoke(users, ["remove", "alice"])

        assert "explicit 'index:' values" not in _flat(result.output)

    def test_without_the_resume_branch_the_real_engine_would_refuse(
        self, tmp_path, monkeypatch, fake_runtime
    ):
        """Why the branch above exists, pinned against the real engine.

        In the interrupted state the engine has nothing to act on and raises,
        so a re-run that called it unconditionally could never finish the
        profile write.
        """
        rendered = _config([{"name": "bob", "index": 1}])
        repo_root = _make_repo(tmp_path, rendered, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        with pytest.raises(ValueError, match="not present"):
            lifecycle.decommission_user(str(repo_root / "build" / "config.yml"), "alice")

    def test_an_unknown_user_still_gets_the_engines_own_error(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Absent from both rosters is a typo, not a resumed removal."""
        rendered = _config([{"name": "bob", "index": 1}])
        repo_root = _make_repo(tmp_path, rendered, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)

        patcher, mocks = _fake_web_terminals()
        mocks["decommission_user"].side_effect = ValueError(
            "User 'nobody' is not present in modules.web_terminals.users; "
            "nothing was decommissioned."
        )
        with patcher:
            result = cli_runner.invoke(users, ["remove", "nobody"])

        assert result.exit_code != 0
        assert mocks["decommission_user"].called
        assert "not present" in _flat(result.output)

    def test_a_declined_typed_gate_leaves_the_profile_untouched(
        self, cli_runner, tmp_path, monkeypatch, fake_runtime
    ):
        """The gate guards the profile write too, because the engine runs first."""
        config = _config([{"name": "alice", "index": 0}, {"name": "bob", "index": 1}])
        repo_root = _make_repo(tmp_path, config, profile=PROFILE_WITH_ROSTER)
        monkeypatch.chdir(repo_root)
        before = (repo_root / "profile.yml").read_text(encoding="utf-8")
        monkeypatch.setattr("builtins.input", lambda prompt="": "")  # blank, non-matching

        result = cli_runner.invoke(users, ["remove", "alice", "--purge"])

        assert result.exit_code != 0
        assert (repo_root / "profile.yml").read_text(encoding="utf-8") == before


class TestEnvProduction:
    """Rendering the file the web terminals run with."""

    def test_renders_the_subset_to_stdout(self, cli_runner, tmp_path, monkeypatch):
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        (repo_root / ".env").write_text(
            "CBORG_API_KEY=llm-secret\nTEST_REGISTRY_TOKEN=registry-secret\n", encoding="utf-8"
        )
        monkeypatch.chdir(repo_root)

        result = cli_runner.invoke(users, ["env-production"])

        assert result.exit_code == 0
        assert "CBORG_API_KEY=llm-secret" in result.output
        assert "TEST_REGISTRY_TOKEN" not in result.output

    def test_the_root_env_is_the_default_source(self, cli_runner, tmp_path, monkeypatch):
        """The single root .env, not anything under build/."""
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        (repo_root / "build" / ".env").write_text("CBORG_API_KEY=from-build\n", encoding="utf-8")
        (repo_root / ".env").write_text("CBORG_API_KEY=from-root\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)

        result = cli_runner.invoke(users, ["env-production"])

        assert result.exit_code == 0
        assert "CBORG_API_KEY=from-root" in result.output

    def test_env_file_and_output_are_relative_to_where_the_command_was_typed(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """Not to the repo root the verb chdirs into."""
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "prod.env").write_text("CBORG_API_KEY=from-elsewhere\n", encoding="utf-8")
        monkeypatch.chdir(elsewhere)

        result = cli_runner.invoke(
            users,
            [
                "env-production",
                "--repo",
                str(repo_root),
                "--env-file",
                "prod.env",
                "--output",
                "rendered.env",
            ],
        )

        assert result.exit_code == 0
        written = elsewhere / "rendered.env"
        assert written.is_file()
        assert "CBORG_API_KEY=from-elsewhere" in written.read_text(encoding="utf-8")
        assert not (repo_root / "rendered.env").exists()

    def test_the_output_file_is_created_at_mode_0600(self, cli_runner, tmp_path, monkeypatch):
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        (repo_root / ".env").write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)

        result = cli_runner.invoke(users, ["env-production", "--output", "out.env"])

        assert result.exit_code == 0
        mode = stat.S_IMODE((repo_root / "out.env").stat().st_mode)
        assert mode == 0o600

    def test_a_persona_whose_project_is_missing_is_named_not_skipped(
        self, cli_runner, tmp_path, monkeypatch, caplog
    ):
        """Silence would read as "that persona needs no secret" — a claim we can't make.

        A persona whose project cannot be found contributes no auth secret, so
        its terminals come up healthy and fail authentication on the first
        prompt. The render still proceeds (seeing the gap is the point), but it
        says which persona and which path.
        """
        config = yaml.safe_load(ENV_PRODUCTION_CONFIG)
        config["modules"]["web_terminals"]["users"] = [{"name": "alice", "persona": "readonly"}]
        config["modules"]["web_terminals"]["personas"] = {
            "readonly": {"project": "ca-readonly", "project_path": "../ca-readonly"}
        }
        repo_root = _make_repo(tmp_path, config)
        (repo_root / ".env").write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)

        with caplog.at_level(logging.WARNING):
            result = cli_runner.invoke(users, ["env-production"])

        assert result.exit_code == 0
        assert "readonly" in caplog.text
        # The literal path searched, `..` and all: it is what the operator has
        # to go and look at, and normalizing it would hide the project_path
        # that produced it.
        assert f"{repo_root}/../ca-readonly/config.yml" in caplog.text
        # The render still happens — the warning reports the gap, it is not a refusal.
        assert "CBORG_API_KEY=llm-secret" in result.output

    def test_stdout_carries_the_file_and_nothing_else(self, cli_runner, tmp_path, monkeypatch):
        """`osprey users env-production > .env.production` has to produce a usable
        file, so nothing but assignments may reach stdout — no banner, no summary."""
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        (repo_root / ".env").write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)

        result = cli_runner.invoke(users, ["env-production"])

        assert result.exit_code == 0
        for line in result.stdout.splitlines():
            assert "=" in line, f"non-assignment line on stdout: {line!r}"

    def test_long_values_are_not_wrapped(self, cli_runner, tmp_path, monkeypatch):
        """Rendered lines are file bytes, not console output — a 200-character
        secret must arrive on one line rather than re-wrapped by Rich."""
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        long_secret = "sk-" + "x" * 200
        (repo_root / ".env").write_text(f"CBORG_API_KEY={long_secret}\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)

        result = cli_runner.invoke(users, ["env-production"])

        assert result.exit_code == 0
        assert f"CBORG_API_KEY={long_secret}" in result.stdout.splitlines()

    def test_ambient_environment_is_not_a_source(self, cli_runner, tmp_path, monkeypatch):
        """Only the handed secrets file counts. The CLI bulk-loads `.env` into
        os.environ at startup, so a variable living solely in the environment
        must not be picked up — otherwise the render would depend on the
        caller's shell rather than on the file it names."""
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        (repo_root / ".env").write_text("CBORG_API_KEY=from-dotenv\n", encoding="utf-8")
        monkeypatch.setenv("TEST_REGISTRY_TOKEN", "from-shell-only")
        monkeypatch.chdir(repo_root)

        result = cli_runner.invoke(users, ["env-production"])

        assert result.exit_code == 0
        assert "CBORG_API_KEY=from-dotenv" in result.output
        assert "from-shell-only" not in result.output

    def test_output_keeps_stdout_clean(self, cli_runner, tmp_path, monkeypatch, caplog):
        """With --output the file is the product, so stdout stays empty and the
        path is reported through the log instead."""
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        (repo_root / ".env").write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)

        with caplog.at_level(logging.INFO):
            result = cli_runner.invoke(users, ["env-production", "--output", "out.env"])

        assert result.exit_code == 0
        assert result.stdout == ""
        assert "CBORG_API_KEY=llm-secret" in (repo_root / "out.env").read_text(encoding="utf-8")
        assert "out.env" in caplog.text

    def test_output_overwrites_an_existing_file(self, cli_runner, tmp_path, monkeypatch):
        """An explicit --output is an instruction, so a re-render replaces it —
        unlike a deploy, which never overwrites an existing .env.production."""
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        (repo_root / ".env").write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")
        (repo_root / "out.env").write_text("STALE=leftover\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)

        result = cli_runner.invoke(users, ["env-production", "--output", "out.env"])

        assert result.exit_code == 0
        assert "STALE" not in (repo_root / "out.env").read_text(encoding="utf-8")

    def test_a_missing_secrets_file_aborts(self, cli_runner, tmp_path, monkeypatch, caplog):
        repo_root = _make_repo(tmp_path, ENV_PRODUCTION_CONFIG)
        monkeypatch.chdir(repo_root)

        result = cli_runner.invoke(users, ["env-production"])

        assert result.exit_code != 0
        assert "does not exist" in caplog.text
