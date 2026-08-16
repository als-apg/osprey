"""One chain, every consumer: the env chain's cross-cutting matrix.

The deployment's environment is a chain of two files at the repo root —
``.env.shared`` (committed defaults) below ``.env`` (host-local secrets) — and
roughly a dozen unrelated mechanisms deliver it: compose reads it as an
``env_file:`` list, the CLI verbs load it into ``os.environ``, the build
lifecycle hands it to a subprocess, an MCP server reads it at startup, and the
web-terminal writer derives a file from it. Each of those has its own tests
next to its own code. What none of them can assert is the property the chain
actually promises: that **every** path answers the same question the same way.

So this module holds one fixture and asks it of all of them:

* the presence matrix — neither file, only ``.env``, only ``.env.shared``,
  both — across the three delivery shapes (what the render tells the templates,
  the ``--env-file`` argv a docker-shaped invocation carries, and the single
  merged file a podman-compose-shaped one is handed), plus the membership
  marker the render leaves behind for the deploy to check against;
* a conflict key set in BOTH files, resolved through every loader in the
  codebase. The expected answer is the same string on every row, so the table
  fails the day one loader disagrees — including the one that reaches it by
  loading the files in the opposite order.

The mechanics differ per path and are supposed to: a ``--env-file`` list and an
``override=True`` loader both let the LAST file win, an ``override=False``
loader lets the FIRST one win and therefore reads the chain reversed, and a
provider that keeps only the last ``--env-file`` it is handed gets one
pre-merged file instead of a list. This suite is deliberately blind to all of
that. It asserts the contract — ``.env`` wins, ``.env.shared`` is still
delivered — so a future "correction" that makes one mechanism's order look
like another's fails here rather than silently dropping the defaults half.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import (
    _inject_project_metadata,
    compose_env_file_args,
    env_chain_names,
    read_rendered_env_chain,
    record_env_chain_membership,
)
from osprey.deployment.container_lifecycle import ENV_MERGED_RELPATH, _env_file_args
from osprey.deployment.runtime_helper import ComposeProvider
from osprey.utils.dotenv import (
    ENV_LOCAL_FILENAME,
    ENV_SHARED_FILENAME,
    parse_dotenv_file,
    parse_dotenv_text,
)

# ---------------------------------------------------------------------------
# The one fixture every path is asked about
# ---------------------------------------------------------------------------
# Three keys, because three things can go wrong and each has its own witness: a
# path that reads only `.env` drops SHARED_ONLY, a path that reads only
# `.env.shared` drops LOCAL_ONLY, and a path that gets the precedence backwards
# resolves CONFLICT to the shared value. Prefixed so a leak into a real
# process environment is obvious in a failure message.

SHARED_ONLY = "OSPREY_CHAIN_SHARED_ONLY"
LOCAL_ONLY = "OSPREY_CHAIN_LOCAL_ONLY"
CONFLICT = "OSPREY_CHAIN_CONFLICT"
CHAIN_KEYS = (SHARED_ONLY, LOCAL_ONLY, CONFLICT)

SHARED_TEXT = f"{SHARED_ONLY}=shared-only\n{CONFLICT}=from-shared\n"
LOCAL_TEXT = f"{LOCAL_ONLY}=local-only\n{CONFLICT}=from-local\n"

#: What a complete chain resolves to, on every path there is.
BOTH_FILES_RESOLVE_TO = {
    SHARED_ONLY: "shared-only",
    LOCAL_ONLY: "local-only",
    CONFLICT: "from-local",
}


def write_chain(root: Path, *, shared: bool = False, local: bool = False) -> Path:
    """Lay down the requested chain members under *root*; omitted ones stay absent."""
    root.mkdir(parents=True, exist_ok=True)
    if shared:
        (root / ENV_SHARED_FILENAME).write_text(SHARED_TEXT, encoding="utf-8")
    if local:
        (root / ENV_LOCAL_FILENAME).write_text(LOCAL_TEXT, encoding="utf-8")
    return root


def expected_values(*, shared: bool, local: bool) -> dict[str, str]:
    """The merged view of the chain members that exist, for one matrix cell."""
    values: dict[str, str] = {}
    if shared:
        values.update(parse_dotenv_text(SHARED_TEXT))
    if local:
        values.update(parse_dotenv_text(LOCAL_TEXT))
    return values


def chain_keys_of(values) -> dict[str, str]:
    """Just this suite's keys, out of whatever mapping a path resolved to.

    Every path returns a different kind of mapping — ``os.environ``, a merged
    dict, a parsed file — and most carry unrelated entries. Narrowing to the
    three keys is what lets one expected dict be asserted against all of them.
    """
    return {key: values[key] for key in CHAIN_KEYS if key in values}


#: The four presence shapes: which files exist, and the chain they make.
PRESENCE_MATRIX = [
    pytest.param(False, False, [], id="neither"),
    pytest.param(True, False, [ENV_SHARED_FILENAME], id="shared-only"),
    pytest.param(False, True, [ENV_LOCAL_FILENAME], id="local-only"),
    pytest.param(True, True, [ENV_SHARED_FILENAME, ENV_LOCAL_FILENAME], id="both"),
]


@pytest.fixture
def chain_repo(tmp_path: Path) -> Path:
    """A deployment repo whose chain is complete: both files, one key in both."""
    return write_chain(tmp_path / "repo", shared=True, local=True)


# ---------------------------------------------------------------------------
# Presence matrix x delivery shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("shared", "local", "expected_names"), PRESENCE_MATRIX)
class TestPresenceMatrix:
    """What each delivery mechanism makes of each presence shape.

    Written as one parametrized class rather than four sets of cases because
    the point is the comparison ACROSS the mechanisms: the render context, the
    docker argv, the podman-compose merged file and the membership marker are
    four consumers of one probe, and a cell where they disagree is a deployment
    whose compose file describes an environment it will not be started with.
    """

    def test_the_render_lists_the_chain_lowest_precedence_first(
        self, tmp_path: Path, shared: bool, local: bool, expected_names: list[str]
    ) -> None:
        """The ordered list a worker's ``env_file:`` block is emitted from.

        Spelled ``./<name>`` against the pinned compose project directory, and
        ordered so the later entry wins — which is the chain's precedence
        expressed as compose reads it.
        """
        repo = write_chain(tmp_path, shared=shared, local=local)

        context = _inject_project_metadata({"project_root": str(repo)})

        assert context["osprey_env_chain"] == [f"./{name}" for name in expected_names]

    def test_the_per_file_flags_agree_with_the_list(
        self, tmp_path: Path, shared: bool, local: bool, expected_names: list[str]
    ) -> None:
        """Templates gate on the flags and iterate the list; one probe answers both."""
        repo = write_chain(tmp_path, shared=shared, local=local)

        context = _inject_project_metadata({"project_root": str(repo)})

        assert context["osprey_env_shared_present"] is (ENV_SHARED_FILENAME in expected_names)
        assert context["osprey_env_present"] is (ENV_LOCAL_FILENAME in expected_names)

    def test_the_docker_argv_repeats_the_flag_in_chain_order(
        self, tmp_path: Path, shared: bool, local: bool, expected_names: list[str]
    ) -> None:
        """``--env-file A --env-file B``, absolute, with ``.env`` last.

        Compose v2 applies a later ``--env-file`` over an earlier one, so the
        order IS the precedence here too. An empty chain gets the empty
        fragment rather than a path to a file that is not there — compose hard
        fails on that.
        """
        repo = write_chain(tmp_path, shared=shared, local=local)

        args = compose_env_file_args(repo)

        assert args == [arg for name in expected_names for arg in ("--env-file", str(repo / name))]

    def test_podman_compose_is_handed_one_merged_file(
        self, tmp_path: Path, shared: bool, local: bool, expected_names: list[str]
    ) -> None:
        """The provider that keeps only the LAST ``--env-file`` gets a pre-merged one.

        Handing it the list would deliver ``.env`` alone and drop every
        committed default with nothing on screen to say so. With no chain file
        at all there is nothing to merge and the shape falls through to the
        default fragment.
        """
        repo = write_chain(tmp_path, shared=shared, local=local)

        args = _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)

        if not expected_names:
            assert args == compose_env_file_args(repo)
            assert not (repo / ENV_MERGED_RELPATH).exists()
            return
        assert args == ["--env-file", str(repo / ENV_MERGED_RELPATH)]
        merged = parse_dotenv_file(repo / ENV_MERGED_RELPATH)
        assert chain_keys_of(merged) == expected_values(shared=shared, local=local)

    def test_the_marker_records_the_membership_the_render_used(
        self, tmp_path: Path, shared: bool, local: bool, expected_names: list[str]
    ) -> None:
        """The record the deploy compares the chain on disk against.

        An empty chain records the empty list — a real answer, and distinct
        from the ``None`` a render that never wrote a marker leaves.
        """
        repo = write_chain(tmp_path, shared=shared, local=local)

        record_env_chain_membership(repo / "build", env_chain_names(repo))

        assert read_rendered_env_chain(repo) == expected_names

    def test_all_four_consumers_name_the_same_files(
        self, tmp_path: Path, shared: bool, local: bool, expected_names: list[str]
    ) -> None:
        """The cross-check the per-mechanism suites cannot make.

        A render that lists one chain in a compose file, passes another on the
        command line and records a third is the failure this whole marker
        exists to catch, so it is asserted directly.
        """
        repo = write_chain(tmp_path, shared=shared, local=local)
        record_env_chain_membership(repo / "build", env_chain_names(repo))

        context = _inject_project_metadata({"project_root": str(repo)})
        from_render = [entry.removeprefix("./") for entry in context["osprey_env_chain"]]
        argv = compose_env_file_args(repo)
        from_argv = [Path(value).name for flag, value in zip(argv[::2], argv[1::2], strict=True)]

        assert from_render == expected_names
        assert from_argv == expected_names
        assert read_rendered_env_chain(repo) == expected_names


# ---------------------------------------------------------------------------
# The conflict key, through every loader
# ---------------------------------------------------------------------------
# Each resolver below runs one real loader against the fixture chain and
# returns what that loader made of this suite's three keys. They share a
# signature so the table can be parametrized: every row must produce the same
# dict, whatever the loader's own mechanics are.

#: A deploy config whose declared secret names ARE this suite's chain keys, so
#: the ``.env.users`` subset generator — which copies only the vars the config
#: names — carries them into the file it writes.
ENV_USERS_CONFIG = {
    "project_name": "chain-matrix",
    "facility": {"name": "Demo Light Source", "prefix": "dls", "timezone": "UTC"},
    "llm": {"provider": "cborg", "api_key_env_var": CONFLICT},
    "modules": {
        "web_terminals": {"enabled": True, "image_source": "local"},
        "olog": {
            "enabled": True,
            "username_env_var": SHARED_ONLY,
            "password_env_var": LOCAL_ONLY,
        },
    },
}


def _render_repo(repo: Path) -> Path:
    """Give *repo* the rendered zone the repo-scoped verbs expect to find."""
    build = repo / "build"
    build.mkdir(parents=True, exist_ok=True)
    (repo / "profile.yml").write_text("preset: control-assistant\n", encoding="utf-8")
    (build / "config.yml").write_text(yaml.safe_dump(ENV_USERS_CONFIG), encoding="utf-8")
    return build / "config.yml"


def resolve_via_load_project_dotenv(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The framework's entry-point loader: the chain over ``os.environ``, cwd-rooted."""
    import osprey.utils.config as config

    monkeypatch.setattr(config, "_dotenv_shell_overrides", {})
    monkeypatch.chdir(repo)
    config.load_project_dotenv()
    return chain_keys_of(os.environ)


def resolve_via_chat_overlay(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """``osprey chat``'s overlay, which runs before the provider spec resolves."""
    from osprey.cli.chat_cmd import _overlay_repo_env

    _overlay_repo_env(repo)
    return chain_keys_of(os.environ)


def resolve_via_query_overlay(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """``osprey query``'s overlay — the same contract on the one-shot verb."""
    from osprey.cli.query_cmd import _overlay_repo_env

    _overlay_repo_env(repo)
    return chain_keys_of(os.environ)


def resolve_via_build_lifecycle(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """A lifecycle step's subprocess environment — asked of the process itself.

    Not the merge helper the phase runner calls, but what a profile's own
    command actually sees: this is the only path whose consumer is a program
    OSPREY did not write.
    """
    from osprey.cli.build_lifecycle import _run_lifecycle_phase
    from osprey.cli.build_profile_schema import LifecycleStep

    dump = repo / "lifecycle-env.txt"
    # The pipe is what makes the runner use a shell, which is what makes the
    # redirect-free `tee` land the environment in a file we can read back.
    step = LifecycleStep(name="dump environment", run=f"env | tee {dump}")
    _run_lifecycle_phase("post_build", [step], repo, repo)
    return chain_keys_of(parse_dotenv_text(dump.read_text(encoding="utf-8")))


def resolve_via_inject_provider_env(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The launch-path passthrough that feeds ``.mcp.json``'s ``${VAR}`` expansion.

    Every key of the chain is copied, not a declared subset — narrowing it
    would silently mis-address the control system rather than fail — so this
    suite's keys ride along like any other.
    """
    from osprey.build.claude_code_resolver import ClaudeCodeModelSpec, inject_provider_env

    environ: dict[str, str] = {}
    inject_provider_env(environ, ClaudeCodeModelSpec(provider="test"), project_dir=repo)
    return chain_keys_of(environ)


def resolve_via_resolver_env_lookup(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The ``${VAR}`` view a provider spec is resolved against (chain over shell)."""
    from osprey.build.claude_code_resolver import _env_lookup

    return chain_keys_of(_env_lookup(repo))


def resolve_via_mcp_env(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The MCP server startup loader — the one that reads the chain REVERSED.

    ``override=False`` means the first file to supply a key keeps it, so the
    same local-wins contract requires the opposite file order. Asserting the
    contract rather than the order is the point: this row is what fails if
    someone "fixes" the order into agreement with the compose lists.
    """
    from osprey.mcp_env import load_dotenv_from_project

    monkeypatch.setenv("OSPREY_CONFIG", str(_render_repo(repo)))
    load_dotenv_from_project()
    return chain_keys_of(os.environ)


def resolve_via_users_env_file(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The ``.env.users`` a local-mode deploy derives for the web terminals."""
    from osprey.deployment.web_terminals import env_production

    written = env_production.ensure_env_production(ENV_USERS_CONFIG, repo)
    return chain_keys_of(parse_dotenv_file(written))


def resolve_via_users_cli(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """``osprey users env``, whose stdout IS the rendered file."""
    from click.testing import CliRunner

    from osprey.cli.users_cmd import env_production, users

    _render_repo(repo)
    result = CliRunner().invoke(users, [env_production.name, "--repo", str(repo)])

    assert result.exit_code == 0, result.output
    return chain_keys_of(parse_dotenv_text(result.output))


def resolve_via_health_cli(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """``osprey health --project <repo>``'s pre-check env load, from ANOTHER cwd.

    The anchors resolve through the config path, never the working directory —
    this row runs without chdir-ing into the repo, which is exactly the stance
    that used to drop ``.env.shared`` (the cwd-rooted chain load never saw the
    target repo, and the repo-anchored reload knew only ``.env``).
    """
    from osprey.cli.health_cmd import _load_project_env, _resolve_anchors

    _render_repo(repo)
    _config_path, _repo_root, env_paths = _resolve_anchors(repo)
    _load_project_env(env_paths)
    return chain_keys_of(os.environ)


def resolve_via_health_loader(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The long-lived health surface's refresh-cycle loader.

    The one loader that also WATCHES what it reads: the chain it loads here is
    the same file set ``signatures.disk_signature`` stats, so an ``.env.shared``
    edit both reloads and invalidates.
    """
    from osprey.health.loader import HealthConfigLoader

    HealthConfigLoader(config_path=_render_repo(repo)).load()
    return chain_keys_of(os.environ)


LOADERS = [
    pytest.param(resolve_via_load_project_dotenv, id="load_project_dotenv"),
    pytest.param(resolve_via_chat_overlay, id="chat-overlay"),
    pytest.param(resolve_via_query_overlay, id="query-overlay"),
    pytest.param(resolve_via_build_lifecycle, id="build-lifecycle"),
    pytest.param(resolve_via_inject_provider_env, id="inject_provider_env"),
    pytest.param(resolve_via_resolver_env_lookup, id="resolver-env-lookup"),
    pytest.param(resolve_via_mcp_env, id="mcp_env-reversed"),
    pytest.param(resolve_via_users_env_file, id="env-users-derivation"),
    pytest.param(resolve_via_users_cli, id="users-env-cli"),
    pytest.param(resolve_via_health_cli, id="health-cli-cross-cwd"),
    pytest.param(resolve_via_health_loader, id="health-loader"),
]


@pytest.mark.parametrize("resolve", LOADERS)
class TestEveryLoaderResolvesTheChainTheSameWay:
    """The conflict key, asked of every loader there is."""

    def test_the_local_file_wins_a_key_both_files_set(
        self, chain_repo: Path, monkeypatch: pytest.MonkeyPatch, resolve
    ) -> None:
        """The whole contract in one assertion, on every path.

        ``.env`` wins the conflict, and neither file's exclusive keys are lost
        on the way — a path that reads one file only fails on the missing key
        rather than on the conflict, which is why all three are asserted here
        together.
        """
        assert resolve(chain_repo, monkeypatch) == BOTH_FILES_RESOLVE_TO

    def test_a_shared_only_chain_still_delivers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve
    ) -> None:
        """Committed defaults with no host-local file is a deployment shape.

        A path that treats ``.env`` as the chain's mandatory member delivers
        nothing at all here.
        """
        repo = write_chain(tmp_path / "repo", shared=True)

        assert resolve(repo, monkeypatch) == parse_dotenv_text(SHARED_TEXT)

    def test_a_local_only_chain_behaves_as_it_did_before_the_chain_existed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve
    ) -> None:
        """The overwhelmingly common shape, and the one that must not have moved."""
        repo = write_chain(tmp_path / "repo", local=True)

        assert resolve(repo, monkeypatch) == parse_dotenv_text(LOCAL_TEXT)


# ---------------------------------------------------------------------------
# `osprey users env` — the chain branch of the CLI
# ---------------------------------------------------------------------------
# The verb renders the same subset a deploy would generate, from the same
# source: the merged chain by default, or ONE named file when the operator
# points at one. The two are different questions, and answering the second with
# the first would hand back secrets the operator did not ask about.


class TestUsersEnvChainBranch:
    """Which files the verb reads, and which it deliberately does not."""

    def _run(self, repo: Path, *args: str):
        from click.testing import CliRunner

        from osprey.cli.users_cmd import env_production, users

        _render_repo(repo)
        return CliRunner().invoke(users, [env_production.name, "--repo", str(repo), *args])

    def test_the_default_source_is_the_merged_chain(self, chain_repo: Path) -> None:
        result = self._run(chain_repo)

        assert result.exit_code == 0, result.output
        assert chain_keys_of(parse_dotenv_text(result.output)) == BOTH_FILES_RESOLVE_TO

    def test_the_shared_half_alone_is_a_source(self, tmp_path: Path) -> None:
        """No host-local ``.env``: there is still a chain, so there is still output."""
        repo = write_chain(tmp_path / "repo", shared=True)

        result = self._run(repo)

        assert result.exit_code == 0, result.output
        assert chain_keys_of(parse_dotenv_text(result.output)) == parse_dotenv_text(SHARED_TEXT)

    def test_an_explicit_env_file_bypasses_the_chain(self, chain_repo: Path) -> None:
        """The operator pointed at a file; rendering from anything else answers
        a different question than the one asked.

        The chain is complete and sets all three keys, so a verb that merged
        the named file WITH it would still emit them — the assertion is that
        the keys the named file does not carry are simply absent.
        """
        named = chain_repo / "elsewhere.env"
        named.write_text(f"{CONFLICT}=from-the-named-file\n", encoding="utf-8")

        result = self._run(chain_repo, "--env-file", str(named))

        assert result.exit_code == 0, result.output
        assert chain_keys_of(parse_dotenv_text(result.output)) == {CONFLICT: "from-the-named-file"}

    def test_no_chain_file_at_all_is_refused_naming_both(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nothing to render from, and the refusal says which files were looked for.

        Read from the log rather than stdout on purpose: in the default mode
        stdout IS the rendered file, so a diagnostic printed there would
        corrupt a ``> .env.production`` redirect.
        """
        repo = tmp_path / "repo"
        repo.mkdir()

        result = self._run(repo)

        assert result.exit_code != 0
        assert ENV_SHARED_FILENAME in caplog.text
        assert ENV_LOCAL_FILENAME in caplog.text
        assert result.output.strip() == "Aborted!"


# ---------------------------------------------------------------------------
# What the entry-load overrode: the shell record, across two files
# ---------------------------------------------------------------------------
# `load_project_dotenv` overrides `os.environ` from the chain and records what
# each override replaced, so the deploy path can reconstruct the operator's
# shell — compose gives a shell export precedence over `--env-file`, and the
# shadow preflight is built on this record. With two files in play the record
# has to be taken against the value the chain ENDS on, once, rather than once
# per file against whatever the previous file said.


class TestShellOverrideRecordAcrossTheChain:
    """One record per key, judged against the winning value."""

    @pytest.fixture(autouse=True)
    def _fresh_record(self, monkeypatch: pytest.MonkeyPatch):
        """The record accumulates process-wide; start every test from empty."""
        import osprey.utils.config as config

        monkeypatch.setattr(config, "_dotenv_shell_overrides", {})
        return config

    def test_the_shell_value_is_recorded_not_the_intermediate_shared_one(
        self, chain_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recorded before any file loads, so no file's value can be mistaken
        for the shell's.

        A record taken per file would see ``os.environ`` already holding
        ``from-shared`` by the time ``.env`` loaded, and would report the
        defaults file as the operator's own export.
        """
        import osprey.utils.config as config

        monkeypatch.setenv(CONFLICT, "from-the-shell")
        monkeypatch.chdir(chain_repo)

        config.load_project_dotenv()

        assert config.dotenv_shell_overrides() == {CONFLICT: "from-the-shell"}
        assert os.environ[CONFLICT] == "from-local"

    def test_a_shell_export_agreeing_with_the_winner_shadowed_nothing(
        self, chain_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The judgement is against the value the chain delivers — the local
        one — however the shared defaults spelled it."""
        import osprey.utils.config as config

        monkeypatch.setenv(CONFLICT, "from-local")
        monkeypatch.chdir(chain_repo)

        config.load_project_dotenv()

        assert config.dotenv_shell_overrides() == {}
        assert os.environ[CONFLICT] == "from-local"

    def test_a_key_only_the_shared_half_sets_is_recorded_too(
        self, chain_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defaults file overrides a shell export just as the local one does."""
        import osprey.utils.config as config

        monkeypatch.setenv(SHARED_ONLY, "from-the-shell")
        monkeypatch.chdir(chain_repo)

        config.load_project_dotenv()

        assert config.dotenv_shell_overrides() == {SHARED_ONLY: "from-the-shell"}
        assert os.environ[SHARED_ONLY] == "shared-only"

    def test_a_second_load_does_not_erase_the_record(
        self, chain_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The load runs more than once per process.

        After the first, ``os.environ`` already matches the chain — so a
        re-derived record would be empty, and the shadow preflight would report
        an operator's export as absent.
        """
        import osprey.utils.config as config

        monkeypatch.setenv(CONFLICT, "from-the-shell")
        monkeypatch.chdir(chain_repo)

        config.load_project_dotenv()
        config.load_project_dotenv()

        assert config.dotenv_shell_overrides() == {CONFLICT: "from-the-shell"}
