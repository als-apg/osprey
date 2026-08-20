"""The OpenObserve telemetry INGEST identity, and why only half of it is minted.

The agent ships telemetry to a co-deployed OpenObserve store. It used to do that
as ``root`` — the account that also owns the store's administration — so root's
password travelled into every rendered ``config.yml`` and every web-terminal
environment. The ingest identity exists to stop that: a second, separate account
the agent authenticates as, leaving root's credentials on the deploy host.

That identity is an OpenObserve **service account**, and the choice is what makes
this file different from every other mint suite here. A service account's secret
is issued by the STORE (``POST /api/{org}/service_accounts`` mints a 16-character
alphanumeric token, read back with ``GET /api/{org}/service_accounts/{email}``
once the container is live), not chosen by osprey. So the identity splits across
two registries with opposite contracts:

* ``ZO_INGEST_USER_EMAIL`` — an account NAME. Non-secret, constant, written into
  ``.env`` for discoverability by ``_SERVICE_DEFAULT_VARS``, exactly like the
  root email beside it.
* ``ZO_INGEST_SA_TOKEN`` — a SERVER-ISSUED secret. Registered in
  ``_STORE_ISSUED_VARS`` and nowhere else. Not minted, not generated, not
  validated: osprey records this value, it does not choose it.

Three ways of getting that wrong are pinned below, because each looks like the
obvious thing to do and each fails silently or fatally:

* a ``_VAR_GENERATORS`` recipe would fabricate a token the store never issued —
  a healthy-looking ``.env`` and a 401 on every telemetry request;
* ``_validate_openobserve_password`` (8–128 characters, four character classes)
  rejects a legitimate 16-character alphanumeric server token, so registering it
  would refuse the credential the store just handed over;
* a ``_VOLUME_INITIALIZED_VARS`` entry would make ``deploy_restart`` predict a
  mint that never happens and abort as "stale" over a token that is absent
  exactly as designed.

The API shapes quoted here are the ones verified live against the pinned
``v0.14.4`` image in ``research/d5-spike.md``.
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.container_lifecycle import (
    _SERVICE_DEFAULT_VARS,
    _SERVICE_TOKEN_VARS,
    _STORE_ISSUED_VARS,
    _VOLUME_INITIALIZED_VARS,
    _deploy_written_env_vars,
    _preflight_pinned_writers,
    _required_env_problems,
)

_TOKEN_VAR = "ZO_INGEST_SA_TOKEN"
_EMAIL_VAR = "ZO_INGEST_USER_EMAIL"
_EMAIL_DEFAULT = "ingest@example.com"

#: A fabricated token shaped like the ones the store issues:
#: 16 characters drawn from ``[A-Za-z0-9]``.
_SERVER_TOKEN = "Fak3T0kenFak3T0k"


def _parse_dotenv(path: Path) -> dict[str, str]:
    from osprey.utils.dotenv import parse_dotenv_file

    return parse_dotenv_file(path) if path.is_file() else {}


def _openobserve_template_text() -> str:
    import osprey.templates

    path = (
        Path(osprey.templates.__file__).parent
        / "services"
        / "openobserve"
        / "docker-compose.yml.j2"
    )
    return path.read_text(encoding="utf-8")


@pytest.fixture
def _clean_ingest_env(monkeypatch):
    """No ambient copy of any name this suite reasons about.

    A developer shell (or a previous test) exporting one of these would make
    ``_effective_value`` report it as already provided, and several assertions
    here are about what happens when nothing provides it.
    """
    for name in (
        _TOKEN_VAR,
        _EMAIL_VAR,
        "ZO_ROOT_USER_EMAIL",
        "ZO_ROOT_USER_PASSWORD",
        "EVENT_DISPATCHER_TOKEN",
        "DISPATCH_WORKER_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


def _openobserve() -> dict:
    return {"deployed_services": ["openobserve"]}


def _repo_with_pins(tmp_path: Path, *pinned: str) -> Path:
    """A deployment root whose ``profile.yml`` pins the given names."""
    root = tmp_path / "pinned-repo"
    root.mkdir(parents=True, exist_ok=True)
    entries = "".join(f"    - {name}\n" for name in pinned)
    (root / "profile.yml").write_text(f"name: proj\nenv:\n  pinned:\n{entries}", encoding="utf-8")
    return root


def _repo_requiring(tmp_path: Path, *required: str) -> Path:
    """A deployment root whose ``profile.yml`` declares ``env.required`` names."""
    root = tmp_path / "required-repo"
    root.mkdir(parents=True, exist_ok=True)
    entries = "".join(f"    - {name}\n" for name in required)
    (root / "profile.yml").write_text(f"name: proj\nenv:\n  required:\n{entries}", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# The account NAME — non-secret, written for discoverability, and handled by
# exactly the same block as the root email beside it. Half a login an operator
# cannot find is a login nobody can use.
# ---------------------------------------------------------------------------


def test_deploying_openobserve_writes_the_ingest_account_name(tmp_path, _clean_ingest_env):
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )

    assert _parse_dotenv(env_path)[_EMAIL_VAR] == _EMAIL_DEFAULT


def test_both_account_names_are_written_together(tmp_path, _clean_ingest_env):
    """Root's name and the ingest name are one answer to one question.

    An operator looking up "how do I reach this store" needs the administration
    login and the identity their telemetry authenticates as; finding one and not
    the other sends them back to grepping templates.
    """
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )

    env = _parse_dotenv(env_path)
    assert env["ZO_ROOT_USER_EMAIL"] == "root@example.com"
    assert env[_EMAIL_VAR] == _EMAIL_DEFAULT
    # One block, not two: the defaults writer appends once per deploy.
    assert env_path.read_text().count("# Service account names (osprey up)") == 1


def test_the_ingest_name_is_a_different_account_from_root(tmp_path, _clean_ingest_env):
    """The whole point of the identity — same default value would be a no-op."""
    written = dict(_SERVICE_DEFAULT_VARS["openobserve"])

    assert written[_EMAIL_VAR] == _EMAIL_DEFAULT
    assert written[_EMAIL_VAR] != written["ZO_ROOT_USER_EMAIL"]


def test_an_operator_ingest_email_is_never_overwritten(tmp_path, _clean_ingest_env):
    env_path = tmp_path / ".env"
    env_path.write_text(f"{_EMAIL_VAR}=telemetry@facility.example\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )

    assert _parse_dotenv(env_path)[_EMAIL_VAR] == "telemetry@facility.example"
    assert env_path.read_text().count(f"{_EMAIL_VAR}=") == 1


def test_a_stale_empty_export_does_not_outrank_the_written_ingest_email(
    tmp_path, monkeypatch, _clean_ingest_env
):
    """Writing the line is not enough while the process still says otherwise.

    The deploy loads the project ``.env`` over its own environment before
    reaching the writer, so a name the file spells empty leaves an empty copy in
    ``os.environ`` — and that copy outranks the ``--env-file`` line for compose's
    interpolation. Without the repair the container receives the blank while
    ``.env`` shows the name: written down and not delivered.
    """
    monkeypatch.setenv(_EMAIL_VAR, "")
    env_path = tmp_path / ".env"
    env_path.write_text(f"{_EMAIL_VAR}=\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )

    assert _parse_dotenv(env_path)[_EMAIL_VAR] == _EMAIL_DEFAULT
    assert os.environ[_EMAIL_VAR] == _EMAIL_DEFAULT


def test_the_ingest_name_is_written_even_when_the_password_already_exists(
    tmp_path, _clean_ingest_env
):
    """The upgrade path: every deployment that predates the ingest identity.

    Those projects carry a minted ``ZO_ROOT_USER_PASSWORD`` and nothing else, so
    the mint generates nothing on their next deploy. The defaults block sits
    outside the mint for exactly that reason — folded in, the deployments that
    need the line would be the only ones that never got it.
    """
    env_path = tmp_path / ".env"
    env_path.write_text("ZO_ROOT_USER_PASSWORD=MyStr0ng#Facility@Pass\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )

    assert _parse_dotenv(env_path)[_EMAIL_VAR] == _EMAIL_DEFAULT


def test_no_ingest_name_is_written_when_openobserve_is_not_deployed(
    tmp_path, monkeypatch, _clean_ingest_env
):
    """Keyed by deployed service, exactly like the token map."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "a" * 40)
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", "b" * 40)
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["event_dispatcher"]}, expose_network=False, env_path=env_path
    )

    assert _EMAIL_VAR not in _parse_dotenv(env_path)


def test_writing_the_ingest_name_is_idempotent(tmp_path, _clean_ingest_env):
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )
    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )

    assert env_path.read_text().count(f"{_EMAIL_VAR}=") == 1


def test_the_ingest_name_has_no_fallback_in_the_openobserve_template():
    """Deliberate, and the inverse of the root email's contract.

    ``ZO_ROOT_USER_EMAIL`` must equal the template's ``${...:-default}`` because
    the container reads it to initialize itself. The store reads nothing about
    the ingest account: that identity is created through the API after the
    container is live, so a fallback here would suggest a consumer that does not
    exist. The default that matters is the agent's telemetry config.
    """
    template = _openobserve_template_text()

    assert re.search(r"\$\{ZO_ROOT_USER_EMAIL:-", template), "root email fallback went missing"
    assert _EMAIL_VAR not in template
    assert _TOKEN_VAR not in template


# ---------------------------------------------------------------------------
# The TOKEN — issued by the store, never by osprey. Every assertion here is the
# negative of something the other mint suites assert, which is the point.
# ---------------------------------------------------------------------------


def test_no_service_declares_the_ingest_token_as_a_minted_var():
    """``_SERVICE_TOKEN_VARS`` membership is what makes a var get generated."""
    declared = {var for names in _SERVICE_TOKEN_VARS.values() for var in names}

    assert _TOKEN_VAR not in declared


def test_the_ingest_token_has_no_generator_recipe():
    """A recipe would fabricate a token the store never issued.

    The failure that produces is the worst shape available: ``.env`` carries a
    strong-looking secret, the deploy reports a successful mint, and every
    telemetry request 401s against a store that has never heard of the value.
    """
    assert _TOKEN_VAR not in container_lifecycle._VAR_GENERATORS


def test_deploying_openobserve_writes_no_ingest_token(tmp_path, _clean_ingest_env):
    """The pre-start writers leave the token alone — it does not exist yet.

    At this point in the run the container has not started, so the store has not
    issued anything. The value arrives later, harvested from the live store.
    """
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )

    env = _parse_dotenv(env_path)
    assert _TOKEN_VAR not in env
    # ...while the halves osprey *does* own were written, so this is not a
    # vacuous pass over a writer that did nothing at all.
    assert env["ZO_ROOT_USER_PASSWORD"]
    assert env[_EMAIL_VAR] == _EMAIL_DEFAULT


def test_an_exposed_deploy_does_not_refuse_over_the_absent_ingest_token(
    tmp_path, _clean_ingest_env
):
    """The empty-token refusal must not reach a store-issued name.

    A deploy reachable off-host refuses to start with an empty *required* token,
    because that means a service comes up unauthenticated. The ingest token is
    absent on every first deploy by construction, so a refusal here would make
    the exposed path unstartable — the concrete cost of registering this var in
    ``_SERVICE_TOKEN_VARS``.
    """
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=True, env_path=env_path
    )

    assert _TOKEN_VAR not in _parse_dotenv(env_path)


def test_the_root_password_policy_would_reject_a_server_issued_token():
    """Spike trap 8, pinned as an executable fact rather than a comment.

    OpenObserve issues 16 alphanumeric characters. That value carries no special
    character, so the four-class root-password policy rejects it — applying the
    validator to this var would refuse the very credential the store handed
    over. The policy is not wrong; it grades a value osprey mints.
    """
    assert len(_SERVER_TOKEN) == 16
    assert _SERVER_TOKEN.isalnum()

    assert not container_lifecycle._validate_openobserve_password(_SERVER_TOKEN)
    # ...and the minted root password, which the policy exists for, still passes.
    assert container_lifecycle._validate_openobserve_password(
        container_lifecycle._generate_openobserve_password()
    )


def test_the_ingest_token_has_no_registered_format_constraint():
    """Neither validated as a required var nor as a validate-only one.

    Absence from both maps is what keeps ``_validate_var`` fail-open for it. A
    format rule here would be osprey grading a value it did not choose, against
    an alphabet only the store's current version defines.
    """
    assert _TOKEN_VAR not in container_lifecycle._VAR_VALIDATORS
    assert _TOKEN_VAR not in container_lifecycle._VALIDATE_ONLY_VARS


@pytest.mark.parametrize(
    "token",
    [
        _SERVER_TOKEN,
        "Fixture0Fixture0",
        string.ascii_letters[:10] + "0123456",
        "0000000000000000",
    ],
)
def test_a_harvested_token_survives_a_dotenv_round_trip(tmp_path, token):
    """Whatever the harvest writes must read back byte-identical.

    The value is not osprey's to choose, but the *write* is osprey's to get
    right: the harvest appends through the same block writer every other
    provisioner uses, and a value that did not round-trip would leave the store
    holding one token and the agent presenting another.
    """
    env_path = tmp_path / ".env"

    container_lifecycle._append_env_block(env_path, "harvest under test", {_TOKEN_VAR: token})

    assert _parse_dotenv(env_path)[_TOKEN_VAR] == token
    # The block writer also owns the file mode, secrets-destined by construction.
    assert env_path.stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------------------
# Registry agreement — every map that names the token, and every map that
# deliberately does not.
# ---------------------------------------------------------------------------


class TestRegistryAgreement:
    """The registrations that make one server-issued value reach its readers."""

    def test_the_token_is_registered_as_store_issued(self):
        assert _STORE_ISSUED_VARS[_TOKEN_VAR].service == "openobserve"
        assert _STORE_ISSUED_VARS[_TOKEN_VAR].identity_var == _EMAIL_VAR

    def test_the_registry_keyset_is_pinned(self):
        """Blast-radius pin, mirroring the ``_VAR_GENERATORS`` one.

        A store-issued credential is a genuinely unusual thing to register, and
        the registry's whole value is that a reader can tell at a glance which
        secrets osprey does not choose. Widening it silently loses that.
        """
        assert set(_STORE_ISSUED_VARS) == {_TOKEN_VAR}

    def test_every_store_issued_var_names_a_service_that_can_be_deployed(self):
        """A service name nothing deploys makes the registration unreachable."""
        deployable = set(_SERVICE_TOKEN_VARS) | set(_SERVICE_DEFAULT_VARS)
        for var, store in _STORE_ISSUED_VARS.items():
            assert store.service in deployable, f"{var} names an undeployable service"

    def test_every_store_issued_var_pairs_with_an_identity_the_deploy_writes(self):
        """The token is useless without the account name it is presented with.

        Both halves go on the wire together (``Basic base64(email:token)``), so
        the identity var must be one this deploy actually writes down — and for
        the same service, or the pair names two different stores.
        """
        for var, store in _STORE_ISSUED_VARS.items():
            written = dict(_SERVICE_DEFAULT_VARS.get(store.service, ()))
            assert store.identity_var in written, f"{var}'s identity is never written to .env"

    def test_the_token_is_in_the_writer_census(self):
        """Written later in the run than the rest, but written by this command.

        That is the only question the census asks, and two other checks read the
        answer: the pin refusal below, and the required-env probe.
        """
        assert _TOKEN_VAR in _deploy_written_env_vars()
        assert _EMAIL_VAR in _deploy_written_env_vars()

    def test_a_pin_on_the_token_refuses(self, tmp_path):
        """``env.pinned`` says the chain owns a value outright; this deploy writes it."""
        repo = _repo_with_pins(tmp_path, _TOKEN_VAR)

        with pytest.raises(RuntimeError) as excinfo:
            _preflight_pinned_writers(repo)

        assert _TOKEN_VAR in str(excinfo.value) or "1 variable" in str(excinfo.value)

    def test_the_required_env_probe_does_not_ask_for_the_token(self, tmp_path):
        """Nobody can supply it — asking would send an operator hunting.

        The probe refuses a deploy whose profile declares a required name no
        source provides. This name is provided by the store, after the start it
        would otherwise be blocking.
        """
        repo = _repo_requiring(tmp_path, _TOKEN_VAR, _EMAIL_VAR)

        assert _required_env_problems(repo, {}) == []

    def test_the_token_is_not_registered_as_volume_initialized(self):
        """The harvest model there reads the CONTAINER's environment.

        A service-account token was never in any container's environment — it
        lives inside the volume's own database — so
        ``_harvest_original_credential`` would answer ``None`` forever and
        ``--reuse-stores`` would report an adoptable store as unrecoverable.
        """
        assert _TOKEN_VAR not in _VOLUME_INITIALIZED_VARS

    def test_restart_predicts_no_mint_for_the_token(self, tmp_path, _clean_ingest_env):
        """The other half of that decision, and the fatal one.

        ``deploy_restart`` asks which store credentials a mint *would* generate
        and feeds the answer to the stale-volume preflight as ``minted``. A
        ``_VOLUME_INITIALIZED_VARS`` entry would put this token in that answer on
        every run where it is absent — which is every run before the first
        harvest — and abort the restart as stale over a credential that is
        missing by design.
        """
        env_path = tmp_path / ".env"
        env_path.write_text("ZO_ROOT_USER_PASSWORD=MyStr0ng#Facility@Pass\n", encoding="utf-8")

        predicted = container_lifecycle._volume_initialized_vars_that_would_be_minted(
            _openobserve(), env_path
        )

        assert _TOKEN_VAR not in predicted
        # The root password is present, so nothing at all is predicted here —
        # the assertion above is not riding on an empty answer for a third reason.
        assert predicted == set()


# ---------------------------------------------------------------------------
# The unconditional-mint contract is untouched. Adding a var that is NOT minted
# to a module full of vars that are is exactly the change that could quietly
# widen or narrow what the mint does.
# ---------------------------------------------------------------------------


def test_an_openobserve_deploy_still_mints_exactly_the_root_password(tmp_path, _clean_ingest_env):
    env_path = tmp_path / ".env"

    minted = container_lifecycle._ensure_service_tokens(
        _openobserve(), expose_network=False, env_path=env_path
    )

    env = _parse_dotenv(env_path)
    secrets_written = set(env) - {"ZO_ROOT_USER_EMAIL", _EMAIL_VAR}
    assert secrets_written == {"ZO_ROOT_USER_PASSWORD"}
    # The stale-volume return value is the minted subset a data volume ignores.
    assert minted == {"ZO_ROOT_USER_PASSWORD"}


def test_the_other_services_mint_exactly_what_they_declared(tmp_path, _clean_ingest_env):
    """No new name leaks into an unrelated deploy's ``.env``."""
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["event_dispatcher", "bluesky"]},
        expose_network=False,
        env_path=env_path,
    )

    env = _parse_dotenv(env_path)
    assert set(env) == {
        "EVENT_DISPATCHER_TOKEN",
        "DISPATCH_WORKER_TOKEN",
        "BLUESKY_LAUNCH_TOKEN",
        "BLUESKY_TILED_API_KEY",
    }
