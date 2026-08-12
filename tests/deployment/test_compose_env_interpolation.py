"""Secrets bound for a compose ``env_file:`` must not contain ``$``.

Docker Compose interpolates the *values* it reads from an ``env_file:``, not
just the compose document. A ``$`` followed by an identifier character is
replaced with the named variable's value — the empty string when that name is
unset — so a ``$``-bearing secret arrives inside the container truncated. Two
variants carry no warning at all: ``$$`` collapses to a single ``$``, and a
``$`` followed by a name that *is* set on the deploy host splices the host's
value in. podman-compose mangles a different set — the braced ``${...}`` form
only, resolved against the same file's earlier entries before the host
environment — so the identical file can yield a different secret depending on
which implementation reads it.

The same substitution happens on the OTHER route out of a ``.env``: every
deploy runs compose with ``--env-file .env``, which makes it the variable source
for the compose *document*, so an ``environment: - X=${X}`` entry — how most
services here receive their secrets — resolves through it too. Measured on
Docker Compose v2.34, ``h0rse$battery`` becomes ``h0rse`` and ``secret$HOME``
becomes the deploy host's home path, the latter with no warning.

The failure is invisible from the host. The file on disk is byte-perfect, every
test that reads the file passes, and only the service that consumes the secret
refuses. These tests pin the guard at each boundary where a value OSPREY did not
mint can reach a container:

* ``.env.auth``   — the auth sidecar (includes the operator-appended OIDC secret)
* ``.env.production`` — every per-user web terminal
* ``.env``        — the compose document on every deploy, plus the dispatch
  worker, which receives the file wholesale

The rule is "no ``$`` at all", not "no interpolating ``$``". A trailing ``$``
survives Docker Compose today, but podman-compose is the documented default for
facility deploys and substitutes a different set, so no single escaping is
correct for both; the codebase already designs to a ``$``-free invariant
(``passwords.FIELD_SEP``, the minted alphabets in ``service_tokens``). A
rejected-but-harmless secret costs a rotation. A missed one is a silent outage.

Offending values are reported by variable *name* only — never echoed — matching
``_raise_invalid_var``'s existing discipline.
"""

from __future__ import annotations

import pytest

from osprey.deployment import container_lifecycle, service_tokens
from osprey.deployment.errors import ComposeInterpolationError
from osprey.deployment.web_terminals import auth_credentials, env_production, provision
from osprey.utils.dotenv import compose_unsafe_vars

# ---------------------------------------------------------------------------
# The detection helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "secret$abc",  # interpolated -> truncated (warned)
        "secret${abc}",  # braced form -> truncated (warned)
        "secret$_x",  # underscore is an identifier char
        "P@$$w0rd",  # $$ collapses to $ -- SILENT
        "secret$HOME",  # host value spliced in -- SILENT
        "trailing$",  # survives docker compose, not guaranteed elsewhere
        "digit$1abc",  # ditto
    ],
)
def test_compose_unsafe_vars_flags_every_dollar_form(value: str) -> None:
    assert compose_unsafe_vars({"SECRET": value}) == ["SECRET"]


def test_compose_unsafe_vars_passes_clean_values() -> None:
    clean = {
        "ANTHROPIC_API_KEY": "sk-ant-api03-AbC_def-123XyZ",  # base64url
        "ARIEL_DB_PASSWORD": "deadbeefdeadbeef",  # token_hex alphabet
        "OSPREY_AUTH_SESSION_SECRET": "xY-_09azAZ",  # token_urlsafe
        "TZ": "America/Los_Angeles",
    }
    assert compose_unsafe_vars(clean) == []


def test_compose_unsafe_vars_returns_names_never_values() -> None:
    """The helper's return type is the leak guard: it cannot carry a secret."""
    offenders = compose_unsafe_vars({"B_KEY": "b$ad", "A_KEY": "a$ad", "OK": "fine"})
    assert offenders == ["A_KEY", "B_KEY"]  # sorted names, no values


# ---------------------------------------------------------------------------
# service_tokens validators -- the three that reason about character safety
# and admitted `$` anyway
# ---------------------------------------------------------------------------


def test_ariel_dsn_with_dollar_in_password_is_rejected() -> None:
    """``_validate_ariel_dsn`` rejects ``@:/?#`` but ``$`` is a URI sub-delim.

    ``postgresql://ariel:se$cret@db/ariel`` parses cleanly and reaches the
    container as ``postgresql://ariel:se@db/ariel``.
    """
    assert not service_tokens._validate_var("ARIEL_DSN", "postgresql://ariel:se$cret@db:5432/ariel")


def test_ariel_db_password_with_dollar_is_rejected() -> None:
    assert not service_tokens._validate_var("ARIEL_DB_PASSWORD", "se$cret")


def test_openobserve_password_dollar_does_not_satisfy_the_special_class() -> None:
    """The policy check *requires* a non-alphanumeric character.

    ``$`` satisfies ``any(not c.isalnum())``, so it was not merely tolerated --
    it was an accepted way to pass. ``Passw0rd$x`` cleared validation and the
    container received ``Passw0rd``.
    """
    assert not service_tokens._validate_var("ZO_ROOT_USER_PASSWORD", "Passw0rd$x")


def test_dollar_guard_applies_to_vars_with_no_registered_validator() -> None:
    """``_validate_var`` is fail-open per-format, but never for ``$``."""
    assert not service_tokens._validate_var("EVENT_DISPATCHER_TOKEN", "tok$en")


@pytest.mark.parametrize(
    "var",
    [
        # The three with custom alphabets...
        "ZO_ROOT_USER_PASSWORD",
        "ARIEL_DB_PASSWORD",
        "BLUESKY_TILED_API_KEY",
        # ...and the three that fall back to _default_token.
        "EVENT_DISPATCHER_TOKEN",
        "DISPATCH_WORKER_TOKEN",
        "BLUESKY_LAUNCH_TOKEN",
    ],
)
def test_minted_secrets_still_satisfy_their_own_validators(var: str) -> None:
    """Regression: the guard must not reject what OSPREY itself mints.

    Also the premise the ``.env`` scan's placement rests on — it runs *before*
    the mint, which is only sound because minted values are ``$``-free.
    """
    minted = service_tokens._generate_token(var)
    assert "$" not in minted, f"{var} generator emits a value its own guard rejects"
    assert service_tokens._validate_var(var, minted)


# ---------------------------------------------------------------------------
# The project .env -- handed to the dispatch worker WHOLESALE, so the check
# has to cover the file, not just the vars that carry a registered validator
# ---------------------------------------------------------------------------


def test_ensure_service_tokens_refuses_a_dollar_outside_the_validated_vars(tmp_path) -> None:
    """The hole a per-var check cannot close.

    ``_validate_var`` only ever runs over the deployed services'
    ``_SERVICE_TOKEN_VARS`` plus ``_VALIDATE_ONLY_VARS``. ``OLOG_PASSWORD`` is
    in neither, yet ``env_file: ../../.env`` hands the dispatch worker the whole
    file — so the scan has to be over the file.
    """
    env_path = tmp_path / ".env"
    env_path.write_text("OLOG_PASSWORD=h0rse$battery\n", encoding="utf-8")

    with pytest.raises(ComposeInterpolationError) as excinfo:
        container_lifecycle._ensure_service_tokens({}, False, env_path)

    assert "OLOG_PASSWORD" in str(excinfo.value)
    assert "battery" not in str(excinfo.value)


def test_ensure_service_tokens_refuses_before_minting_anything(tmp_path) -> None:
    """A refused deploy must not leave freshly minted tokens behind.

    The operator fixes the offending value and re-runs; tokens appended before
    the refusal would still be sitting in ``.env``, unused by any container that
    ever started, and indistinguishable from the live ones. Checking ahead of
    the mint keeps the file exactly as the refusal found it.
    """
    env_path = tmp_path / ".env"
    env_path.write_text("OLOG_PASSWORD=h0rse$battery\n", encoding="utf-8")
    config = {"deployed_services": ["event_dispatcher"]}

    with pytest.raises(ComposeInterpolationError):
        container_lifecycle._ensure_service_tokens(config, False, env_path)

    assert env_path.read_text(encoding="utf-8") == "OLOG_PASSWORD=h0rse$battery\n", (
        "nothing may be appended to .env before the scan refuses"
    )


def test_ensure_service_tokens_passes_a_clean_env(tmp_path) -> None:
    """Regression: the ordinary path is undisturbed."""
    env_path = tmp_path / ".env"
    env_path.write_text("OLOG_PASSWORD=cleanpassword\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens({}, False, env_path)

    assert "cleanpassword" in env_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# .env.production -- every value is copied verbatim from the operator's .env
# ---------------------------------------------------------------------------


_CONFIG = {
    "facility": {"name": "T", "prefix": "t", "timezone": "UTC"},
    "llm": {"provider": "cborg", "api_key_env_var": "CBORG_API_KEY"},
    "modules": {
        "web_terminals": {"enabled": True, "image_source": "local"},
        "olog": {
            "enabled": True,
            "username_env_var": "OLOG_USERNAME",
            "password_env_var": "OLOG_PASSWORD",
        },
    },
}


def _write_env(tmp_path, values: dict) -> None:
    (tmp_path / ".env").write_text(
        "".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8"
    )


def test_env_production_refuses_a_dollar_bearing_secret(tmp_path) -> None:
    """An OLOG password with a ``$`` must abort the deploy, not ship truncated."""
    _write_env(
        tmp_path,
        {
            "CBORG_API_KEY": "sk-clean-key",
            "OLOG_USERNAME": "logbot",
            "OLOG_PASSWORD": "h0rse$battery",
        },
    )

    with pytest.raises(ComposeInterpolationError) as excinfo:
        env_production.ensure_env_production(_CONFIG, tmp_path)

    message = str(excinfo.value)
    assert "OLOG_PASSWORD" in message, "the offending variable must be named"
    assert "h0rse" not in message, "the secret value must never be echoed"
    assert not (tmp_path / ".env.production").exists(), (
        "a refused deploy must not leave a half-written secrets file behind"
    )


def test_env_production_writes_clean_secrets_unchanged(tmp_path) -> None:
    """Regression: the guard must not disturb the ordinary path."""
    _write_env(
        tmp_path,
        {
            "CBORG_API_KEY": "sk-clean-key",
            "OLOG_USERNAME": "logbot",
            "OLOG_PASSWORD": "cleanpassword",
        },
    )

    path = env_production.ensure_env_production(_CONFIG, tmp_path)

    assert "OLOG_PASSWORD=cleanpassword" in path.read_text(encoding="utf-8")


def test_env_production_scans_a_file_it_did_not_generate(tmp_path) -> None:
    """Registry mode -- the DEFAULT -- never generates this file.

    CI assembles it from masked variables and ships it beside the image; an
    operator-authored file takes the same path, since an existing one is never
    regenerated. Both hold values OSPREY never saw, so the generate-path check
    alone would leave the default deploy unguarded.
    """
    (tmp_path / ".env.production").write_text("OLOG_PASSWORD=h0rse$battery\n", encoding="utf-8")

    with pytest.raises(ComposeInterpolationError) as excinfo:
        env_production.ensure_env_production(_CONFIG, tmp_path)

    assert "OLOG_PASSWORD" in str(excinfo.value)
    assert "battery" not in str(excinfo.value)


def test_env_production_returns_an_existing_clean_file_untouched(tmp_path) -> None:
    """Regression: an existing clean file is still returned, never regenerated."""
    existing = tmp_path / ".env.production"
    existing.write_text("OLOG_PASSWORD=cleanpassword\n", encoding="utf-8")

    path = env_production.ensure_env_production(_CONFIG, tmp_path)

    assert path.read_text(encoding="utf-8") == "OLOG_PASSWORD=cleanpassword\n"


def test_env_production_refuses_a_dollar_in_the_ariel_dsn(tmp_path) -> None:
    """``ARIEL_DSN`` comes from facility config, not ``.env`` -- same guard."""
    config = {
        **_CONFIG,
        "modules": {
            **_CONFIG["modules"],
            "ariel": {"enabled": True, "dsn": "postgresql://ariel:se$cret@db:5432/ariel"},
        },
    }
    _write_env(tmp_path, {"CBORG_API_KEY": "sk-clean-key"})

    with pytest.raises(ComposeInterpolationError) as excinfo:
        env_production.ensure_env_production(config, tmp_path)

    assert "ARIEL_DSN" in str(excinfo.value)
    assert "se$cret" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# .env.auth -- the OIDC client secret, which no OSPREY code writes
# ---------------------------------------------------------------------------


def test_env_auth_scan_catches_an_operator_appended_oidc_secret(tmp_path) -> None:
    """The one value in ``.env.auth`` a human chooses, and no writer can guard.

    The compose template emits only the variable *name*; the operator appends
    the IdP-minted value by hand. The generator-side regression test cannot see
    it, so the deploy path must scan the file as it exists on disk.
    """
    (tmp_path / ".env.auth").write_text(
        "OSPREY_AUTH_OIDC_CLIENT_SECRET=idp$ecret-value\n", encoding="utf-8"
    )
    web_terminals = {
        "enabled": True,
        "auth": {"method": "password"},
        "users": [{"name": "alice"}],
    }

    with pytest.raises(ComposeInterpolationError) as excinfo:
        provision._provision_auth_secrets(web_terminals, str(tmp_path))

    message = str(excinfo.value)
    assert "OSPREY_AUTH_OIDC_CLIENT_SECRET" in message
    assert "ecret-value" not in message


def test_sidecar_recreate_itself_is_never_blocked_by_the_scan(tmp_path, monkeypatch) -> None:
    """The recreate must stay unconditional, and this is why.

    A recreate is always the step that puts an ALREADY-APPLIED change to
    ``.env.auth`` into force. ``decommission``/``prune`` remove a departed
    user's hash and then recreate; refusing in between would leave the file
    saying the user is gone while the running sidecar still accepts their
    password — the exact divergence those verbs exist to close, traded for an
    unrelated secret. So the scan goes ahead of each mutation, never here.
    """
    monkeypatch.chdir(tmp_path)
    # Where the recreate addresses it: the repo's render zone, not the root.
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "docker-compose.web.yml").write_text("services: {}\n", encoding="utf-8")
    (tmp_path / ".env.auth").write_text(
        "OSPREY_AUTH_OIDC_CLIENT_SECRET=idp$ecret\n", encoding="utf-8"
    )
    called: list[object] = []
    monkeypatch.setattr(provision, "_force_recreate_services", lambda *a, **k: called.append(a))
    monkeypatch.setattr(provision, "web_stack_compose_cmd", lambda *a, **k: ["compose"])
    monkeypatch.setattr(provision, "runtime_env", lambda *a, **k: {})
    # The pre-recreate re-render is out of scope here (pinned in
    # test_provision.py); the stub config is not renderable.
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda *a, **k: [])

    provision.force_recreate_auth_sidecar({})

    assert called, "a credential removal must still be put into force"


def test_password_rotation_is_refused_before_it_writes(tmp_path, monkeypatch) -> None:
    """The one removal-free verb, so refusing up front costs nothing.

    An operator who has just pasted an IdP client secret into ``.env.auth`` and
    reaches for ``passwd`` is the likeliest way a ``$``-bearing value meets a
    deploy verb. Refusing before ``set_auth_password`` writes means the operator
    is never left holding a password the sidecar was not given.
    """
    (tmp_path / ".env.auth").write_text(
        "OSPREY_AUTH_OIDC_CLIENT_SECRET=idp$ecret\n", encoding="utf-8"
    )
    before = (tmp_path / ".env.auth").read_text(encoding="utf-8")

    with pytest.raises(ComposeInterpolationError):
        auth_credentials.raise_if_env_auth_would_be_interpolated(tmp_path)

    assert (tmp_path / ".env.auth").read_text(encoding="utf-8") == before


def test_env_auth_scan_passes_a_clean_file(tmp_path) -> None:
    """Regression: a clean operator-supplied secret must deploy normally."""
    (tmp_path / ".env.auth").write_text(
        "OSPREY_AUTH_OIDC_CLIENT_SECRET=cleanidpsecret\n", encoding="utf-8"
    )
    web_terminals = {
        "enabled": True,
        "auth": {"method": "password"},
        "users": [{"name": "alice"}],
    }

    provision._provision_auth_secrets(web_terminals, str(tmp_path))

    assert "cleanidpsecret" in (tmp_path / ".env.auth").read_text(encoding="utf-8")
