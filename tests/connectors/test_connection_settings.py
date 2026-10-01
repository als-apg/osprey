"""The reader of an outbound connection block: shape, refusals, credential, trust and opener."""

import base64
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from osprey_connectors.connection import (
    ConnectionSettings,
    read_ca_bundle,
    read_connection_settings,
    read_credential_env_names,
    urllib_opener,
)
from tests.connectors._loopback_https import Reply, loopback_pair

WHERE = "archiver.settings"
SECRET = "s3cr3t-token-value"


def _read(block, **kwargs):
    return read_connection_settings(block, where=WHERE, **kwargs)


@pytest.fixture
def servers(tmp_path):
    with loopback_pair(tmp_path) as pair:
        yield pair


# -- shape ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", [None, {}])
def test_an_absent_block_reads_as_nothing_set(block):
    settings = _read(block)
    assert settings.url is None
    assert settings.timeout_s is None
    assert settings.login is None
    assert settings.ca_bundle is None
    assert settings.credential_env_names == ()


def test_a_token_login_names_its_variable():
    settings = _read({"auth": {"token_env": "OSPREY_ARCHIVER_TOKEN"}})
    assert settings.login is not None
    assert settings.login.kind == "token"
    assert settings.login.token_env == "OSPREY_ARCHIVER_TOKEN"
    assert settings.login.username is None
    assert settings.login.password_env is None
    assert settings.credential_env_names == ("OSPREY_ARCHIVER_TOKEN",)


def test_a_password_login_names_user_and_variable():
    settings = _read({"auth": {"username": "reader", "password_env": "ARCHIVER_PASSWORD"}})
    assert settings.login is not None
    assert settings.login.kind == "password"
    assert settings.login.username == "reader"
    assert settings.login.password_env == "ARCHIVER_PASSWORD"
    assert settings.login.token_env is None
    assert settings.credential_env_names == ("ARCHIVER_PASSWORD",)


def test_two_logins_are_refused():
    with pytest.raises(ValueError, match=r"`archiver\.settings\.auth` names two logins"):
        _read({"auth": {"token_env": "T", "username": "u", "password_env": "P"}})


@pytest.mark.parametrize(
    ("auth", "missing"),
    [({"username": "u"}, "password_env"), ({"password_env": "P"}, "username")],
)
def test_half_a_password_login_is_refused_naming_the_missing_key(auth, missing):
    with pytest.raises(ValueError, match=rf"`archiver\.settings\.auth\.{missing}` is missing"):
        _read({"auth": auth})


@pytest.mark.parametrize("key", ["token", "password", "secret", "api_key"])
def test_an_unknown_auth_key_is_refused_by_name(key):
    with pytest.raises(ValueError, match=rf"`archiver\.settings\.auth\.{key}`") as exc:
        _read({"auth": {key: "literal-value"}})
    assert "literal-value" not in str(exc.value)
    if key in ("token", "password", "secret"):
        assert "never written in config" in str(exc.value)


def test_a_block_may_name_its_own_extra_auth_keys():
    settings = _read(
        {"auth": {"source": "admin", "username": "u", "password_env": "P"}},
        extra_auth_keys=frozenset({"source"}),
    )
    assert settings.login is not None
    assert (settings.login.username, settings.login.password_env) == ("u", "P")


def test_an_extra_auth_key_is_refused_where_the_block_does_not_name_it():
    with pytest.raises(ValueError, match=r"`archiver\.settings\.auth\.source` is not a login key"):
        _read({"auth": {"source": "admin", "username": "u", "password_env": "P"}})


def test_an_extra_auth_key_is_not_a_login():
    with pytest.raises(ValueError, match="names no login"):
        _read({"auth": {"source": "admin"}}, extra_auth_keys=frozenset({"source"}))


@pytest.mark.parametrize("value", ["s3cr3t value", "${X}", "1ABC", ""])
def test_a_variable_name_that_is_not_one_is_refused_without_echoing_it(value):
    with pytest.raises(ValueError, match="must name an environment variable") as exc:
        _read({"auth": {"token_env": value}})
    if value:
        assert value not in str(exc.value)


@pytest.mark.parametrize(
    "auth",
    [{"token_env": "TOKEN\n"}, {"username": "u", "password_env": "PASSWORD\n"}],
    ids=["token_env", "password_env"],
)
def test_a_variable_name_with_a_trailing_newline_is_refused(auth):
    """No environment holds a name ending in a newline, so the login could never resolve."""
    with pytest.raises(ValueError, match="must name an environment variable"):
        _read({"auth": auth})


@pytest.mark.parametrize("value", ["", "  "])
def test_a_blank_url_reads_as_unset(value):
    assert _read({"url": value}).url is None


def test_a_url_that_is_not_a_string_is_refused():
    with pytest.raises(ValueError, match=r"`archiver\.settings\.url` must be a string"):
        _read({"url": 8443})


def test_a_url_carrying_a_password_is_refused_without_echoing_it():
    with pytest.raises(ValueError, match="may not carry a password") as exc:
        _read({"url": "https://reader:hunter2@archiver.example.org"})
    assert "hunter2" not in str(exc.value)


@pytest.mark.parametrize("key", ["verify", "insecure", "verify_ssl", "ciphers"])
def test_tls_takes_only_ca_bundle(key):
    with pytest.raises(ValueError, match=rf"`archiver\.settings\.tls\.{key}`") as exc:
        _read({"tls": {key: False}})
    if key != "ciphers":
        assert "no setting turns certificate verification off" in str(exc.value)


def test_a_relative_ca_bundle_is_refused():
    with pytest.raises(
        ValueError, match=r"`archiver\.settings\.tls\.ca_bundle` must be an absolute path"
    ):
        _read({"tls": {"ca_bundle": "certs/site-ca.pem"}})


_HOME_RELATIVE = (
    r"`archiver\.settings\.tls\.ca_bundle` must be an absolute path, got .*spell the full path"
)


@pytest.mark.parametrize("raw", ["~/site-ca.pem", "~operator/site-ca.pem"])
def test_a_home_relative_ca_bundle_is_refused(raw):
    with pytest.raises(ValueError, match=_HOME_RELATIVE):
        _read({"tls": {"ca_bundle": raw}})


@pytest.mark.parametrize("value", [0, -1, "60", True])
def test_timeout_s_must_be_a_positive_number(value):
    with pytest.raises(ValueError, match=r"`archiver\.settings\.timeout_s`"):
        _read({"timeout_s": value})


def test_a_blank_timeout_s_reads_as_unset():
    settings = _read({"timeout_s": None})
    assert settings.timeout_s is None
    assert settings.timeout_or(60) == 60
    assert _read({"timeout_s": 5}).timeout_or(60) == 5.0


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("timeout", "timeout_s"),
        ("token_env", "auth.token_env"),
        ("auth_token_env", "auth.token_env"),
        ("username", "auth.username"),
        ("password_env", "auth.password_env"),
        ("ca_bundle", "tls.ca_bundle"),
    ],
)
def test_a_flat_spelling_is_refused_naming_its_nested_key(old, new):
    with pytest.raises(ValueError) as exc:
        _read({old: "X"})
    assert f"`{WHERE}.{old}`" in str(exc.value)
    assert f"`{WHERE}.{new}`" in str(exc.value)


def test_a_headers_map_is_refused():
    with pytest.raises(ValueError, match="no free-form header map"):
        _read({"headers": {"Authorization": "Bearer x"}})


def test_a_connection_that_sends_no_login_refuses_auth_with_its_reason():
    with pytest.raises(ValueError, match=r"`archiver\.settings\.auth` is not accepted here: why"):
        _read({"auth": {"token_env": "T"}}, logins=frozenset(), unsupported_because="why")


def test_a_token_only_connection_refuses_a_password_login():
    with pytest.raises(ValueError, match="is not accepted here: tokens only"):
        _read(
            {"auth": {"username": "u", "password_env": "P"}},
            logins=frozenset({"token"}),
            unsupported_because="tokens only",
        )
    assert _read(
        {"auth": {"token_env": "T"}}, logins=frozenset({"token"}), unsupported_because="x"
    ).credential_env_names == ("T",)


def test_a_connection_that_sends_no_ca_refuses_tls_with_its_reason():
    with pytest.raises(ValueError, match=r"`archiver\.settings\.tls` is not accepted here: no CA"):
        _read({"tls": {"ca_bundle": "/etc/ca.pem"}}, tls=False, unsupported_because="no CA")


@pytest.mark.parametrize(
    "kwargs", [{"logins": frozenset()}, {"tls": False}, {"logins": frozenset({"token"})}]
)
def test_a_restriction_without_a_reason_is_a_programming_error(kwargs):
    with pytest.raises(TypeError):
        _read({}, **kwargs)


# -- credential variable names -------------------------------------------------------


def _names(block):
    return read_credential_env_names(block, where=WHERE)


def test_the_names_accessor_reports_a_token_variable_by_its_ruled_key():
    assert _names({"auth": {"token_env": "ARCHIVER_TOKEN"}}) == (
        ("auth.token_env", "ARCHIVER_TOKEN"),
    )


def test_the_names_accessor_reports_a_password_variable_but_never_the_username():
    block = {"auth": {"username": "reader", "password_env": "ARCHIVER_PW"}}
    assert _names(block) == (("auth.password_env", "ARCHIVER_PW"),)


@pytest.mark.parametrize("block", [None, {}, {"url": "https://a.example"}])
def test_the_names_accessor_reports_nothing_without_a_login(block):
    assert _names(block) == ()


def test_the_names_accessor_tolerates_auth_keys_outside_the_login_pair():
    block = {"auth": {"username": "u", "password_env": "PW", "source": "admin"}}
    assert _names(block) == (("auth.password_env", "PW"),)


def test_the_names_accessor_reads_no_flat_spelling():
    assert _names({"password_env": "PW", "token_env": "TOK"}) == ()


def test_the_names_accessor_reports_a_name_as_written():
    assert _names({"auth": {"token_env": "not a name"}}) == (("auth.token_env", "not a name"),)


def test_the_names_accessor_refuses_two_logins():
    block = {"auth": {"token_env": "T", "username": "u", "password_env": "P"}}
    with pytest.raises(ValueError, match=r"`archiver\.settings\.auth` names two logins"):
        _names(block)


def test_the_names_accessor_refuses_an_auth_that_is_not_a_mapping():
    with pytest.raises(ValueError, match=r"`archiver\.settings\.auth` must be a mapping"):
        _names({"auth": "admin"})


def test_the_names_accessor_reads_no_environment(monkeypatch):
    block = {"auth": {"token_env": "ARCHIVER_TOKEN"}}
    monkeypatch.delenv("ARCHIVER_TOKEN", raising=False)
    unset = _names(block)
    monkeypatch.setenv("ARCHIVER_TOKEN", SECRET)
    assert _names(block) == unset == (("auth.token_env", "ARCHIVER_TOKEN"),)


# -- CA file -------------------------------------------------------------------------


def _ca(block):
    return read_ca_bundle(block, where=WHERE)


def test_read_ca_bundle_returns_the_named_path():
    block = {"tls": {"ca_bundle": "/etc/ssl/certs/site-ca.pem"}}
    assert _ca(block) == Path("/etc/ssl/certs/site-ca.pem")


@pytest.mark.parametrize("block", [None, {}, {"tls": {}}])
def test_read_ca_bundle_is_none_without_tls(block):
    assert _ca(block) is None


def test_read_ca_bundle_refuses_a_home_relative_path():
    with pytest.raises(ValueError, match=_HOME_RELATIVE):
        _ca({"tls": {"ca_bundle": "~/site-ca.pem"}})


def test_read_ca_bundle_refuses_a_relative_path():
    with pytest.raises(
        ValueError, match=r"`archiver\.settings\.tls\.ca_bundle` must be an absolute path"
    ):
        _ca({"tls": {"ca_bundle": "certs/site-ca.pem"}})


def test_read_ca_bundle_ignores_a_flat_ca_bundle():
    assert _ca({"ca_bundle": "/x.pem"}) is None


def test_read_ca_bundle_checks_nothing_but_tls():
    block = {"auth": {"token": "inline"}, "tls": {"ca_bundle": "/etc/ssl/certs/site-ca.pem"}}
    with pytest.raises(ValueError):
        _read(block)
    assert _ca(block) == Path("/etc/ssl/certs/site-ca.pem")


def test_read_ca_bundle_agrees_with_the_full_reader():
    block = {
        "url": "https://archiver.example.org",
        "auth": {"token_env": "ARCHIVER_TOKEN"},
        "tls": {"ca_bundle": "/etc/ssl/certs/site-ca.pem"},
    }
    assert _ca(block) == _read(block).ca_bundle


def test_read_ca_bundle_reads_no_file(tmp_path):
    missing = tmp_path / "absent" / "site-ca.pem"
    assert _ca({"tls": {"ca_bundle": str(missing)}}) == missing


# -- credential ----------------------------------------------------------------------


@pytest.mark.parametrize("environ", [{}, {"ARCHIVER_TOKEN": "  "}])
def test_an_unset_or_blank_credential_variable_is_refused_naming_key_and_variable(environ):
    settings = _read({"auth": {"token_env": "ARCHIVER_TOKEN"}})
    with pytest.raises(ConnectionError) as exc:
        settings.resolve_credential(environ)
    assert "`archiver.settings.auth.token_env`" in str(exc.value)
    assert "'ARCHIVER_TOKEN'" in str(exc.value)


def test_a_token_becomes_a_bearer_header():
    settings = _read({"auth": {"token_env": "ARCHIVER_TOKEN"}})
    credential = settings.resolve_credential({"ARCHIVER_TOKEN": SECRET})
    assert credential is not None
    assert credential.authorization_header() == f"Bearer {SECRET}"


def test_a_password_becomes_a_basic_header():
    settings = _read({"auth": {"username": "reader", "password_env": "ARCHIVER_PASSWORD"}})
    credential = settings.resolve_credential({"ARCHIVER_PASSWORD": "pä:ss"})
    assert credential is not None
    scheme, encoded = credential.authorization_header().split(" ", 1)
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode("utf-8") == "reader:pä:ss"


def test_no_login_resolves_to_no_credential():
    assert _read({}).resolve_credential({}) is None


def test_no_repr_carries_the_secret():
    settings = _read(
        {"url": "https://archiver.example.org", "auth": {"token_env": "ARCHIVER_TOKEN"}}
    )
    environ = {"ARCHIVER_TOKEN": SECRET}
    credential = settings.resolve_credential(environ)
    opener = urllib_opener(settings, environ=environ)
    for thing in [settings, settings.login, credential, *opener.handlers]:
        assert SECRET not in repr(thing)


# -- trust ---------------------------------------------------------------------------


def test_without_a_ca_bundle_the_default_trust_store_applies():
    context = _read({}).ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.cert_store_stats() == ssl.create_default_context().cert_store_stats()


def test_a_missing_ca_bundle_is_refused_naming_the_key(tmp_path):
    settings = _read({"tls": {"ca_bundle": str(tmp_path / "absent.pem")}})
    with pytest.raises(ValueError, match=r"`archiver\.settings\.tls\.ca_bundle`"):
        settings.ssl_context()


def test_a_file_that_holds_no_certificate_is_refused_naming_the_key(tmp_path):
    empty = tmp_path / "not-a-cert.pem"
    empty.write_text("this is not a certificate\n")
    settings = _read({"tls": {"ca_bundle": str(empty)}})
    with pytest.raises(ValueError, match=r"`archiver\.settings\.tls\.ca_bundle`"):
        settings.ssl_context()


def test_the_opener_trusts_the_named_ca(servers):
    servers.https.routes["/ok"] = Reply(body={"ok": True})
    settings = _read({"url": servers.https.url, "tls": {"ca_bundle": str(servers.ca_pem)}})
    with urllib_opener(settings).open(f"{servers.https.url}/ok", timeout=5) as resp:
        assert resp.status == 200


def test_without_the_named_ca_the_certificate_is_refused(servers):
    servers.https.routes["/ok"] = Reply(body={"ok": True})
    settings = _read({"url": servers.https.url})
    with pytest.raises(urllib.error.URLError) as exc:
        urllib_opener(settings).open(f"{servers.https.url}/ok", timeout=5)
    assert isinstance(exc.value.reason, ssl.SSLCertVerificationError)
    assert servers.https.seen == []


# -- opener --------------------------------------------------------------------------


def test_the_opener_sends_the_login_to_its_own_origin(servers):
    servers.https.routes["/ok"] = Reply(body={"ok": True})
    settings = _read(
        {
            "url": servers.https.url,
            "auth": {"token_env": "ARCHIVER_TOKEN"},
            "tls": {"ca_bundle": str(servers.ca_pem)},
        }
    )
    opener = urllib_opener(settings, environ={"ARCHIVER_TOKEN": SECRET})
    with opener.open(f"{servers.https.url}/ok", timeout=5):
        pass
    assert servers.https.seen == [("/ok", f"Bearer {SECRET}")]


def test_a_redirect_to_another_origin_carries_no_login(servers):
    servers.http.routes["/elsewhere"] = Reply(body={"ok": True})
    servers.https.routes["/moved"] = Reply(status=302, location=f"{servers.http.url}/elsewhere")
    settings = _read(
        {
            "url": servers.https.url,
            "auth": {"username": "reader", "password_env": "ARCHIVER_PASSWORD"},
            "tls": {"ca_bundle": str(servers.ca_pem)},
        }
    )
    opener = urllib_opener(settings, environ={"ARCHIVER_PASSWORD": SECRET})
    with opener.open(f"{servers.https.url}/moved", timeout=5) as resp:
        assert resp.status == 200
    assert servers.https.authorizations()[0].startswith("Basic ")
    assert servers.http.seen == [("/elsewhere", None)]


@pytest.mark.parametrize(
    "settings",
    [
        ConnectionSettings(where=WHERE, url="ftp://archiver.example.org/"),
        ConnectionSettings(where=WHERE),
    ],
)
def test_the_opener_refuses_a_url_that_is_not_http(settings):
    with pytest.raises(ValueError, match=r"`archiver\.settings\.url`"):
        urllib_opener(settings)
