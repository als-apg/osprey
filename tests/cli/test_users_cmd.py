"""Terminal-secret provisioning through the ``osprey users`` verbs.

A per-user web-terminal secret is what lets a terminal refuse every request that
did not come through nginx. It lives in the deploy ``.env`` — not ``.env.auth``,
which only the auth sidecar mounts — so the roster verbs that make a user's
terminal real (``seed``, ``env``) must establish it, and the verbs that take a
user away (``remove``, ``prune``) must take it with them: the engines' own purge
reaches ``.env.auth`` alone, so without this a re-added same-name user would
inherit a value their predecessor still knows.

The lifecycle engines themselves are stubbed here. What is under test is the
CLI's own half — which file it writes, when relative to the engine, and what an
operator is told when it cannot — not the container work those engines do, which
``tests/cli/test_users_group.py`` covers.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.users_cmd import users
from osprey.deployment.web_terminals import auth_credentials, lifecycle, seeding
from osprey.deployment.web_terminals.auth_credentials import TERMINAL_SECRET_VAR_PREFIX
from osprey.utils.dotenv import ENV_LOCAL_FILENAME, parse_dotenv_file

pytestmark = pytest.mark.unit

ALICE_SECRET = f"{TERMINAL_SECRET_VAR_PREFIX}ALICE"
BOB_SECRET = f"{TERMINAL_SECRET_VAR_PREFIX}BOB"

#: A rendered ``build/config.yml`` for a two-user multi-user deployment. Carries
#: the provider credential the ``env`` verb renders from, so one config serves
#: every verb here.
RENDERED_CONFIG = textwrap.dedent(
    """
    project_name: demo-project
    facility:
      name: Demo Light Source
      prefix: dls
      timezone: UTC
    llm:
      provider: cborg
      api_key_env_var: CBORG_API_KEY
    modules:
      web_terminals:
        enabled: true
        image_source: local
        nginx_port: 8080
        users:
          - name: alice
            index: 0
          - name: bob
            index: 1
    """
)

#: The same deployment after bob has been hand-edited off the roster — what
#: ``prune`` sees when it goes looking for orphans.
RENDERED_CONFIG_ALICE_ONLY = RENDERED_CONFIG.replace("      - name: bob\n        index: 1\n", "")

PROFILE_WITH_ROSTER = textwrap.dedent(
    """
    preset: control-assistant
    config:
      facility.prefix: dls
      modules.web_terminals:
        enabled: true
        users:
          - name: alice
            index: 0
          - name: bob
            index: 1
    """
)


@pytest.fixture
def cli_runner():
    return CliRunner()


def _make_repo(tmp_path: Path, rendered: str = RENDERED_CONFIG) -> Path:
    """A deployment repo the verbs can resolve: ``profile.yml`` + rendered build."""
    repo_root = tmp_path / "repo"
    (repo_root / "build").mkdir(parents=True)
    (repo_root / "profile.yml").write_text(PROFILE_WITH_ROSTER, encoding="utf-8")
    (repo_root / "build" / "config.yml").write_text(rendered, encoding="utf-8")
    return repo_root.resolve()


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    root = _make_repo(tmp_path)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def stub_engines(monkeypatch):
    """Replace the container-touching engines with recorders.

    Patched on the real modules rather than by shadowing the package, because
    the verbs import them inside their own bodies: the attribute lookup happens
    at call time, so the stub is what the command receives.
    """
    calls: dict[str, list] = {"decommission_user": [], "prune_users": [], "seed_web_terminals": []}

    def _record(name):
        def _call(*args, **kwargs):
            calls[name].append((args, kwargs))

        return _call

    monkeypatch.setattr(lifecycle, "decommission_user", _record("decommission_user"))
    monkeypatch.setattr(lifecycle, "prune_users", _record("prune_users"))
    monkeypatch.setattr(seeding, "seed_web_terminals", _record("seed_web_terminals"))
    return calls


def read_env(repo_root: Path) -> dict[str, str]:
    path = repo_root / ENV_LOCAL_FILENAME
    return parse_dotenv_file(path) if path.is_file() else {}


class TestSeedMintsTerminalSecrets:
    """``osprey users seed`` prepares a workspace; a workspace needs a front door."""

    def test_seeding_mints_a_secret_for_every_roster_user(
        self, cli_runner, repo_root, stub_engines
    ):
        result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code == 0
        stored = read_env(repo_root)
        assert stored[ALICE_SECRET]
        assert stored[BOB_SECRET]
        assert stored[ALICE_SECRET] != stored[BOB_SECRET]

    def test_seeding_one_user_still_provisions_the_whole_roster(
        self, cli_runner, repo_root, stub_engines
    ):
        """The mint is idempotent, and a secret missing for somebody else is a gap
        `osprey up` refuses on either way — so narrowing it would only defer that."""
        result = cli_runner.invoke(users, ["seed", "alice"])

        assert result.exit_code == 0
        assert BOB_SECRET in read_env(repo_root)

    def test_the_mint_runs_before_the_seeding_engine(self, cli_runner, repo_root, monkeypatch):
        """A workspace must never be prepared for a terminal nothing can reach."""
        seen: list[bool] = []

        def _seed(*_args, **_kwargs):
            seen.append(ALICE_SECRET in read_env(repo_root))

        monkeypatch.setattr(seeding, "seed_web_terminals", _seed)

        result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code == 0
        assert seen == [True]

    def test_reseeding_never_rotates_an_established_secret(
        self, cli_runner, repo_root, stub_engines
    ):
        cli_runner.invoke(users, ["seed"])
        before = read_env(repo_root)

        result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code == 0
        assert read_env(repo_root) == before

    def test_an_operators_own_secret_survives_a_seed(self, cli_runner, repo_root, stub_engines):
        (repo_root / ENV_LOCAL_FILENAME).write_text(
            f"{ALICE_SECRET}=set-by-hand\n", encoding="utf-8"
        )

        result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code == 0
        assert read_env(repo_root)[ALICE_SECRET] == "set-by-hand"


class TestEnvMintsTerminalSecrets:
    """``osprey users env`` renders the file the terminals run with."""

    def test_rendering_mints_the_missing_secrets(self, cli_runner, repo_root):
        (repo_root / ENV_LOCAL_FILENAME).write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")

        result = cli_runner.invoke(users, ["env"])

        assert result.exit_code == 0
        stored = read_env(repo_root)
        assert stored[ALICE_SECRET]
        assert stored[BOB_SECRET]

    def test_stdout_stays_a_pure_dotenv_document(self, cli_runner, repo_root):
        """In the default mode stdout IS the rendered file, so the mint's promoted
        fact goes to stderr — one line of prose there would corrupt a redirect."""
        (repo_root / ENV_LOCAL_FILENAME).write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")

        result = cli_runner.invoke(users, ["env"])

        assert result.exit_code == 0
        assert "Provisioned" not in result.stdout
        for line in result.stdout.splitlines():
            assert not line or line.startswith("#") or "=" in line

    def test_a_terminal_secret_never_reaches_the_rendered_env_users(self, cli_runner, repo_root):
        """Every per-user container is handed this file WHOLE, so a secret in it
        would give each user every other user's front-door key."""
        (repo_root / ENV_LOCAL_FILENAME).write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")

        result = cli_runner.invoke(users, ["env"])

        assert result.exit_code == 0
        assert TERMINAL_SECRET_VAR_PREFIX not in result.stdout
        for value in read_env(repo_root).values():
            assert value not in result.stdout or value == "llm-secret"


class TestRemovePurgesTheDepartedSecret:
    """The engines' own purge reaches ``.env.auth`` only."""

    def test_the_removed_users_secret_is_dropped_from_the_deploy_env(
        self, cli_runner, repo_root, stub_engines
    ):
        cli_runner.invoke(users, ["seed"])
        bob_before = read_env(repo_root)[BOB_SECRET]

        result = cli_runner.invoke(users, ["remove", "alice", "--yes"])

        assert result.exit_code == 0
        stored = read_env(repo_root)
        assert ALICE_SECRET not in stored
        assert stored[BOB_SECRET] == bob_before

    def test_a_re_added_user_does_not_inherit_the_previous_holders_secret(
        self, cli_runner, repo_root, stub_engines
    ):
        cli_runner.invoke(users, ["seed"])
        original = read_env(repo_root)[ALICE_SECRET]
        cli_runner.invoke(users, ["remove", "alice", "--yes"])

        result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code == 0
        assert read_env(repo_root)[ALICE_SECRET] != original

    def test_an_engine_that_refuses_leaves_the_secret_in_place(
        self, cli_runner, repo_root, monkeypatch
    ):
        """The engine owns the typed confirmation, so the purge must sit after it:
        a declined gate has to leave the deployment exactly as it was."""
        cli_runner.invoke(users, ["seed"])

        def _refuse(*_args, **_kwargs):
            raise RuntimeError("declined")

        monkeypatch.setattr(lifecycle, "decommission_user", _refuse)

        result = cli_runner.invoke(users, ["remove", "alice"])

        assert result.exit_code != 0
        assert ALICE_SECRET in read_env(repo_root)

    def test_an_unwritable_env_warns_and_names_the_variable(
        self, cli_runner, repo_root, stub_engines, monkeypatch
    ):
        """Everything else about the removal already succeeded, so this cannot
        abort the verb — it has to say what is left over and how to finish it."""
        cli_runner.invoke(users, ["seed"])

        def _fail(*_args, **_kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(auth_credentials, "purge_terminal_secret", _fail)

        result = cli_runner.invoke(users, ["remove", "alice", "--yes"])

        assert result.exit_code == 0, result.output
        assert ALICE_SECRET in result.output
        assert "read-only file system" in result.output

    def test_a_secret_shared_with_a_surviving_user_is_left_alone(
        self, cli_runner, tmp_path, monkeypatch, stub_engines
    ):
        """`alice-b` and `alice_b` key one variable. Such a roster can never be
        minted for — the mint refuses it — so this only fires on a hand-placed
        value, but deleting it would lock out a user who is still on the roster:
        worse than the stale value it would clean up."""
        colliding = yaml.safe_load(RENDERED_CONFIG)
        colliding["modules"]["web_terminals"]["users"] = [
            {"name": "alice-b", "index": 0},
            {"name": "alice_b", "index": 1},
        ]
        root = _make_repo(tmp_path, yaml.safe_dump(colliding))
        monkeypatch.chdir(root)
        shared = f"{TERMINAL_SECRET_VAR_PREFIX}ALICE_B"
        (root / ENV_LOCAL_FILENAME).write_text(f"{shared}=set-by-hand\n", encoding="utf-8")

        result = cli_runner.invoke(users, ["remove", "alice-b", "--yes"])

        assert result.exit_code == 0, result.output
        assert read_env(root)[shared] == "set-by-hand"
        assert "alice_b" in result.output


class TestPrunePurgesOrphanSecrets:
    """Prune discovers orphans in the runtime and reports no names back."""

    def test_secrets_for_users_off_the_roster_are_removed(
        self, cli_runner, tmp_path, monkeypatch, stub_engines
    ):
        root = _make_repo(tmp_path)
        monkeypatch.chdir(root)
        cli_runner.invoke(users, ["seed"])
        alice_before = read_env(root)[ALICE_SECRET]
        # Bob leaves the roster the way prune's docstring describes: hand-edited
        # out of the rendered config without a decommission.
        (root / "build" / "config.yml").write_text(RENDERED_CONFIG_ALICE_ONLY, encoding="utf-8")

        result = cli_runner.invoke(users, ["prune", "--yes"])

        assert result.exit_code == 0
        stored = read_env(root)
        assert BOB_SECRET not in stored
        assert stored[ALICE_SECRET] == alice_before

    def test_dry_run_removes_nothing(self, cli_runner, tmp_path, monkeypatch, stub_engines):
        root = _make_repo(tmp_path)
        monkeypatch.chdir(root)
        cli_runner.invoke(users, ["seed"])
        (root / "build" / "config.yml").write_text(RENDERED_CONFIG_ALICE_ONLY, encoding="utf-8")
        before = (root / ENV_LOCAL_FILENAME).read_text(encoding="utf-8")

        result = cli_runner.invoke(users, ["prune", "--dry-run"])

        assert result.exit_code == 0
        assert (root / ENV_LOCAL_FILENAME).read_text(encoding="utf-8") == before

    def test_an_intact_roster_keeps_every_secret(
        self, cli_runner, repo_root, stub_engines, monkeypatch
    ):
        cli_runner.invoke(users, ["seed"])
        before = (repo_root / ENV_LOCAL_FILENAME).read_text(encoding="utf-8")

        result = cli_runner.invoke(users, ["prune", "--yes"])

        assert result.exit_code == 0
        assert (repo_root / ENV_LOCAL_FILENAME).read_text(encoding="utf-8") == before

    def test_an_unwritable_env_warns_without_failing_the_prune(
        self, cli_runner, repo_root, stub_engines, monkeypatch
    ):
        def _fail(*_args, **_kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(auth_credentials, "purge_orphan_terminal_secrets", _fail)

        result = cli_runner.invoke(users, ["prune", "--yes"])

        assert result.exit_code == 0, result.output
        assert "read-only file system" in result.output


class TestTheDeployEnvItself:
    """Where the secrets land, and what they must not disturb."""

    def test_the_deploy_env_is_created_0600(self, cli_runner, repo_root, stub_engines):
        cli_runner.invoke(users, ["seed"])

        assert (repo_root / ENV_LOCAL_FILENAME).stat().st_mode & 0o777 == 0o600

    def test_nothing_is_written_to_env_auth(self, cli_runner, repo_root, stub_engines):
        cli_runner.invoke(users, ["seed"])

        assert not (repo_root / ".env.auth").exists()

    def test_an_operators_other_variables_survive(self, cli_runner, repo_root, stub_engines):
        (repo_root / ENV_LOCAL_FILENAME).write_text(
            "# my notes\nCBORG_API_KEY=llm-secret\n", encoding="utf-8"
        )

        cli_runner.invoke(users, ["seed"])

        text = (repo_root / ENV_LOCAL_FILENAME).read_text(encoding="utf-8")
        assert "# my notes" in text
        assert "CBORG_API_KEY=llm-secret" in text

    def test_a_roster_that_cannot_be_keyed_refuses_rather_than_guessing(
        self, cli_runner, tmp_path, monkeypatch, stub_engines
    ):
        """Two usernames on one env-var suffix would share a single secret — one
        user's key opening another's terminal."""
        colliding = yaml.safe_load(RENDERED_CONFIG)
        colliding["modules"]["web_terminals"]["users"] = [
            {"name": "alice-b", "index": 0},
            {"name": "alice_b", "index": 1},
        ]
        root = _make_repo(tmp_path, yaml.safe_dump(colliding))
        monkeypatch.chdir(root)

        result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code != 0
        assert not (root / ENV_LOCAL_FILENAME).exists()

    def test_a_repo_with_no_roster_writes_no_env_at_all(
        self, cli_runner, tmp_path, monkeypatch, stub_engines
    ):
        rosterless = yaml.safe_load(RENDERED_CONFIG)
        rosterless["modules"]["web_terminals"].pop("users")
        root = _make_repo(tmp_path, yaml.safe_dump(rosterless))
        monkeypatch.chdir(root)

        result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code == 0
        assert not (root / ENV_LOCAL_FILENAME).exists()

    def test_the_verb_writes_the_repo_root_env_not_one_under_build(
        self, cli_runner, repo_root, stub_engines, monkeypatch
    ):
        """The repo root's .env is the file compose interpolates from; the verbs
        chdir there, so a bare relative path would be right by accident."""
        subdir = repo_root / "deep" / "nested"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        result = cli_runner.invoke(users, ["seed"])

        assert result.exit_code == 0
        assert ALICE_SECRET in read_env(repo_root)
        assert not (subdir / ENV_LOCAL_FILENAME).exists()
        assert not (repo_root / "build" / ENV_LOCAL_FILENAME).exists()

    def test_the_working_directory_is_restored_afterwards(
        self, cli_runner, repo_root, stub_engines, monkeypatch, tmp_path
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        cli_runner.invoke(users, ["seed", "--repo", str(repo_root)])

        assert Path(os.getcwd()).resolve() == elsewhere.resolve()


#: The same deployment with the one field an absolute URL needs. `deploy.fqdn`
#: is what the render builds every external link from — the landing URL, the
#: OIDC redirect_uri, and the per-user login URL — so a repo without it can name
#: no origin at all.
RENDERED_CONFIG_WITH_FQDN = RENDERED_CONFIG.replace(
    "llm:",
    "deploy:\n  host: dls-deploy\n  fqdn: dls-deploy.example.org\nllm:",
)


#: The same deployment behind a login wall, with one entry deliberately left
#: outside it. `login: false` and `auth.method: password` together are the two
#: halves of the predicate the verb refuses on.
RENDERED_CONFIG_WALLED = RENDERED_CONFIG_WITH_FQDN.replace(
    "    users:\n",
    "    auth:\n      method: password\n      allow_insecure_http: true\n    users:\n",
).replace(
    "      - name: bob\n        index: 1\n",
    "      - name: bob\n        index: 1\n"
    "      - name: kiosk\n        index: 2\n        login: false\n",
)


#: The same deployment thrown OPEN: nginx vouches for every non-exempt
#: terminal itself, so there is no login page and no `?token=` URL either.
RENDERED_CONFIG_OPEN = RENDERED_CONFIG_WITH_FQDN.replace(
    "    users:\n",
    "    auth:\n      method: none\n    users:\n",
)


@pytest.fixture
def repo_with_origin(tmp_path, monkeypatch):
    root = _make_repo(tmp_path, RENDERED_CONFIG_WITH_FQDN)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def open_repo(tmp_path, monkeypatch):
    """An `auth.method: none` deployment: open, navigation-only."""
    root = _make_repo(tmp_path, RENDERED_CONFIG_OPEN)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def walled_repo(tmp_path, monkeypatch):
    """A password-authenticated deployment whose `kiosk` entry sits outside the wall."""
    root = _make_repo(tmp_path, RENDERED_CONFIG_WALLED)
    monkeypatch.chdir(root)
    return root


class TestLoginUrl:
    """``osprey users login-url`` — the only way into a token-gated terminal.

    With `auth.method: token` (the default) nginx injects no operator secret
    and runs no login flow, so the per-user app's own gate is the only one
    and a browser gets past it exactly once, by opening that user's `?token=`
    URL. Nothing else in the multi-user shape hands that URL out.
    """

    def test_prints_the_users_own_login_url(self, cli_runner, repo_with_origin, stub_engines):
        cli_runner.invoke(users, ["seed"])
        secret = read_env(repo_with_origin)[ALICE_SECRET]

        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code == 0
        assert (
            result.stdout.strip() == f"http://dls-deploy.example.org:8080/u/alice/?token={secret}"
        )

    def test_stdout_carries_the_url_and_nothing_else(
        self, cli_runner, repo_with_origin, stub_engines
    ):
        """So `osprey users login-url alice | pbcopy` copies a working URL.

        The warning about what the URL contains is real and must be printed —
        just not onto the stream a pipe is reading.
        """
        cli_runner.invoke(users, ["seed"])

        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code == 0
        assert len(result.stdout.strip().splitlines()) == 1
        assert result.stdout.startswith("http://")
        # The caution is not silently dropped; it goes to stderr.
        assert "secret" in result.stderr.lower()

    def test_a_long_url_is_printed_on_one_unwrapped_line(
        self, cli_runner, tmp_path, monkeypatch, stub_engines
    ):
        """The URL goes through the renderer, which wraps by default.

        A newline folded into the middle of a URL an operator is about to paste
        produces a link that silently 404s, and the fold moves with the
        terminal width — so it would not reproduce for whoever is asked about
        it. `output.report` prints with soft_wrap for exactly this reason;
        pinned here because it is a property of the renderer, not of this verb.
        """
        long_host = "a-really-quite-long-deployment-hostname.subdomain.example.org"
        root = _make_repo(
            tmp_path, RENDERED_CONFIG_WITH_FQDN.replace("dls-deploy.example.org", long_host)
        )
        monkeypatch.chdir(root)
        secret = "Zx9-thisIsAVeryLongOperatorSecretValue-abcdefghijklmnop_0123456789"
        (root / ENV_LOCAL_FILENAME).write_text(f"{ALICE_SECRET}={secret}\n", encoding="utf-8")

        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code == 0
        assert result.stdout == f"http://{long_host}:8080/u/alice/?token={secret}\n"

    def test_each_user_gets_their_own_secret(self, cli_runner, repo_with_origin, stub_engines):
        """One secret per user is the isolation the carrier exists to establish;
        a URL that handed out a shared value would erase it."""
        cli_runner.invoke(users, ["seed"])
        stored = read_env(repo_with_origin)

        alice = cli_runner.invoke(users, ["login-url", "alice"]).stdout.strip()
        bob = cli_runner.invoke(users, ["login-url", "bob"]).stdout.strip()

        assert stored[ALICE_SECRET] in alice
        assert stored[BOB_SECRET] in bob
        assert stored[BOB_SECRET] not in alice
        assert stored[ALICE_SECRET] not in bob

    def test_a_hand_pinned_secret_is_percent_encoded_into_the_query(
        self, cli_runner, repo_with_origin, stub_engines
    ):
        """A minted secret needs no escaping; one an operator pinned by hand may."""
        (repo_with_origin / ENV_LOCAL_FILENAME).write_text(
            f"{ALICE_SECRET}=a+b/c=d\n", encoding="utf-8"
        )

        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code == 0
        assert result.stdout.strip().endswith("?token=a%2Bb%2Fc%3Dd")

    def test_a_user_off_the_roster_is_refused_by_name(
        self, cli_runner, repo_with_origin, stub_engines
    ):
        """Named before the secret is looked up: a user who is not on the roster
        has no variable to be missing, and reporting the absent variable would
        send an operator to edit .env over what is really a roster edit."""
        cli_runner.invoke(users, ["seed"])

        result = cli_runner.invoke(users, ["login-url", "eve"])

        assert result.exit_code != 0
        combined = result.stdout + result.stderr
        assert "eve" in combined
        assert "alice" in combined and "bob" in combined
        assert "http://" not in result.stdout

    def test_an_unminted_secret_is_refused_with_the_verb_that_mints_it(
        self, cli_runner, repo_with_origin, stub_engines
    ):
        """No `.env` at all: the deployment has never been started. The remedy is
        the mint, which runs in every auth method, not a value to invent."""
        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code != 0
        combined = result.stdout + result.stderr
        assert ALICE_SECRET in combined
        assert "osprey up" in combined
        assert "http://" not in result.stdout

    def test_a_blank_secret_is_refused_rather_than_handed_out(
        self, cli_runner, repo_with_origin, stub_engines
    ):
        """A present-but-empty variable builds a URL with an empty token, which
        the app refuses — an operator would read that as their own mistake."""
        (repo_with_origin / ENV_LOCAL_FILENAME).write_text(f"{ALICE_SECRET}=\n", encoding="utf-8")

        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code != 0
        assert "token=" not in result.stdout

    def test_a_user_behind_the_login_wall_is_refused_and_told_where_they_sign_in(
        self, cli_runner, walled_repo, stub_engines
    ):
        """The URL is inert for a walled user, so printing it leaks a live
        credential for nothing.

        nginx's `auth_request` denies the request before the app ever sees the
        `?token=`, and even after a successful sidecar login the gated location
        forwards `Cookie ""` — so the token->cookie exchange the URL exists to
        trigger can never apply. The refusal names the page they DO sign in on.
        """
        cli_runner.invoke(users, ["seed"])

        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code != 0
        combined = result.stdout + result.stderr
        assert "http://dls-deploy.example.org:8080/auth/login" in combined
        # No URL and no secret anywhere: refusing is the whole point. (The
        # explanation names the `?token=` mechanism, so the URL is matched on
        # its path rather than on the query it would have carried.)
        assert "/u/alice/" not in combined
        assert read_env(walled_repo)[ALICE_SECRET] not in combined

    def test_a_user_on_an_open_deployment_is_refused_and_told_the_plain_address(
        self, cli_runner, open_repo, stub_engines
    ):
        """`auth.method: none` vouches for every non-exempt terminal itself, so
        there is neither a login page nor a `?token=` exchange to trade — the
        refusal must name the terminal's own address, not the login page a
        walled deployment would.
        """
        cli_runner.invoke(users, ["seed"])

        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code != 0
        combined = result.stdout + result.stderr
        assert "http://dls-deploy.example.org:8080/u/alice/" in combined
        assert "/auth/login" not in combined
        # No live secret anywhere: refusing is the whole point. (The
        # explanation names the `?token=` mechanism, so the URL is matched on
        # its path rather than on the query it would have carried.)
        assert read_env(open_repo)[ALICE_SECRET] not in combined

    def test_a_login_exempt_entry_still_gets_its_url_with_auth_on(
        self, cli_runner, walled_repo, stub_engines
    ):
        """`login: false` puts one entry outside the wall, and that terminal is
        reached exactly the way an auth-off one is — so the verb still answers
        for it while refusing its walled neighbours."""
        cli_runner.invoke(users, ["seed"])
        secret = read_env(walled_repo)["OSPREY_TERMINAL_SECRET_KIOSK"]

        result = cli_runner.invoke(users, ["login-url", "kiosk"])

        assert result.exit_code == 0
        assert (
            result.stdout.strip() == f"http://dls-deploy.example.org:8080/u/kiosk/?token={secret}"
        )

    def test_a_deployment_with_no_fqdn_says_which_key_is_missing(
        self, cli_runner, repo_root, stub_engines
    ):
        """`deploy.fqdn` is the only host this deployment records; without it
        there is no origin to put in front of the path, and a URL built from
        `localhost` would be one the containers' own Origin check refuses."""
        cli_runner.invoke(users, ["seed"])

        result = cli_runner.invoke(users, ["login-url", "alice"])

        assert result.exit_code != 0
        assert "fqdn" in (result.stdout + result.stderr)
