"""The one reader of an outbound connection block's address, bound, login and CA.

Every block that reaches a service by address spells the same shape:

.. code-block:: yaml

    url: https://service.example.org:8443
    timeout_s: 60
    auth:
      token_env: SERVICE_TOKEN          # a bearer token, OR
      username: service-user            # a user and the variable
      password_env: SERVICE_PASSWORD    # that holds the password
    tls:
      ca_bundle: /etc/ssl/certs/site-ca.pem

Three invariants hold for every consumer:

- A secret is named by environment variable, never written in config. The
  settings object holds variable names only; the secret is read from the
  environment when the connection is made.
- There is no free-form header map. A login is declared under ``auth:``.
- No setting turns certificate verification off. ``tls.ca_bundle`` only changes
  which CA is trusted, for this endpoint.

:func:`read_connection_settings`, :func:`read_credential_env_names` and
:func:`read_ca_bundle` are pure: they read neither the environment nor a file,
so a build-time check can call them. The environment and the CA file are
touched only by :meth:`ConnectionSettings.resolve_credential` and
:meth:`ConnectionSettings.ssl_context`, when a connection is made.
"""

from __future__ import annotations

import base64
import os
import re
import ssl
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, Field
from typing_extensions import override

from osprey_connectors.errors import ConfigurationError
from osprey_connectors.model import FrozenModel
from osprey_connectors.model import Login as LoginName

__all__ = [
    "ConnectionSettings",
    "Credential",
    "ENV_NAME_RE",
    "LoginKind",
    "PasswordLogin",
    "ServiceLogin",
    "TokenLogin",
    "read_ca_bundle",
    "read_connection_settings",
    "read_credential_env_names",
    "urllib_opener",
]

LoginKind = Literal["token", "password"]

#: Every login form the shape defines.
_ALL_LOGINS: frozenset[LoginKind] = frozenset({"token", "password"})

#: What an environment variable name looks like, wherever a config names one. ``\A`` and
#: ``\Z`` anchor the whole string, so ``match`` and ``fullmatch`` agree; a ``$`` anchor would
#: also match just before a trailing newline.
ENV_NAME_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")

_AUTH_KEYS = ("token_env", "username", "password_env")
#: The ``auth:`` shape, named in every refusal of it.
_AUTH_SHAPE = "`token_env`, or `username` and `password_env`"
_SECRET_LIKE_KEYS = frozenset({"token", "password", "secret"})
_TLS_KEYS = ("ca_bundle",)
_VERIFY_OFF_KEYS = frozenset({"verify", "insecure", "verify_ssl"})

#: Flat spellings the shape refuses, mapped to the nested key that holds each.
_MOVED_KEYS = {
    "timeout": "timeout_s",
    "token_env": "auth.token_env",
    "auth_token_env": "auth.token_env",
    "username": "auth.username",
    "password_env": "auth.password_env",
    "ca_bundle": "tls.ca_bundle",
}

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _env_name(value: str) -> str:
    """*value* when it names an environment variable (:data:`ENV_NAME_RE`)."""
    if not ENV_NAME_RE.match(value):
        raise ValueError(f"{value!r} does not name an environment variable")
    return value


#: The name of the environment variable a secret is read from.
_EnvName = Annotated[str, AfterValidator(_env_name)]


class TokenLogin(FrozenModel):
    """A bearer-token login: the variable that holds the token. Never the token itself."""

    kind: Literal["token"] = "token"
    token_env: _EnvName

    @property
    def secret_key(self) -> Literal["token_env"]:
        """The ``auth:`` key that names this login's secret variable."""
        return "token_env"

    @property
    def secret_env(self) -> str:
        """The environment variable that holds this login's secret."""
        return self.token_env


class PasswordLogin(FrozenModel):
    """A user and the variable that holds the password. Never the password itself.

    ``username`` is validated, never normalised: a name with space around it is
    refused rather than taken for another name.
    """

    kind: Literal["password"] = "password"
    username: LoginName
    password_env: _EnvName

    @property
    def secret_key(self) -> Literal["password_env"]:
        """The ``auth:`` key that names this login's secret variable."""
        return "password_env"

    @property
    def secret_env(self) -> str:
        """The environment variable that holds this login's secret."""
        return self.password_env


#: A login a connection block names: a bearer token or a user and password,
#: told apart by ``kind``.
ServiceLogin = Annotated[TokenLogin | PasswordLogin, Field(discriminator="kind")]


class Credential(FrozenModel):
    """A login resolved from the environment. Its repr never shows the secret."""

    login: ServiceLogin
    secret: str = Field(repr=False)

    def authorization_header(self) -> str:
        """The ``Authorization`` value: a bearer token, or HTTP Basic sent up front."""
        if isinstance(self.login, TokenLogin):
            return f"Bearer {self.secret}"
        pair = f"{self.login.username}:{self.secret}".encode()
        return "Basic " + base64.b64encode(pair).decode("ascii")


class ConnectionSettings(FrozenModel):
    """An outbound connection block, read and checked. It never holds a secret."""

    where: str
    url: str | None = None
    timeout_s: float | None = None
    login: ServiceLogin | None = None
    ca_bundle: Path | None = None

    @property
    def credential_env_names(self) -> tuple[str, ...]:
        """The environment variables the login's secret is read from; ``()`` without one."""
        return (self.login.secret_env,) if self.login else ()

    def timeout_or(self, default: float) -> float:
        """The configured bound, or ``default`` when the block sets none."""
        return self.timeout_s if self.timeout_s is not None else default

    def resolve_credential(self, environ: Mapping[str, str] | None = None) -> Credential | None:
        """Read the login's secret from the environment.

        Args:
            environ: The environment to read; ``os.environ`` when None.

        Returns:
            The credential, or None when the block names no login.

        Raises:
            ConnectionError: The named variable is unset or blank. That is a
                deployment state, not a config error.
        """
        if self.login is None:
            return None
        env = os.environ if environ is None else environ
        name = self.login.secret_env
        secret = env.get(name, "")
        if not secret.strip():
            raise ConnectionError(
                f"Environment variable '{name}' named by `{self.where}.auth.{self.login.secret_key}` "
                "is not set, so there is no credential to send. Export it where the connector runs."
            )
        return Credential(login=self.login, secret=secret)

    def ssl_context(self) -> ssl.SSLContext:
        """The TLS context for this endpoint.

        Without ``tls.ca_bundle`` it is the default context, so the process
        trust store applies. With one, exactly that file is trusted for this
        endpoint, in place of the default roots. Verification stays on either way.

        Raises:
            ConfigurationError: The CA file is missing, unreadable or holds no certificate.
        """
        if self.ca_bundle is None:
            return ssl.create_default_context()
        key = f"`{self.where}.tls.ca_bundle`"
        try:
            return ssl.create_default_context(cafile=str(self.ca_bundle))
        except FileNotFoundError as e:
            raise ConfigurationError(
                f"{key} names {self.ca_bundle}, which does not exist.",
                key=f"{self.where}.tls.ca_bundle",
            ) from e
        except (IsADirectoryError, PermissionError) as e:
            raise ConfigurationError(
                f"{key} names {self.ca_bundle}, which cannot be read: {e}",
                key=f"{self.where}.tls.ca_bundle",
            ) from e
        except ssl.SSLError as e:
            raise ConfigurationError(
                f"{key} names {self.ca_bundle}, which holds no PEM certificate: {e}",
                key=f"{self.where}.tls.ca_bundle",
            ) from e


def read_connection_settings(
    block: Mapping[str, object] | None,
    *,
    where: str,
    logins: frozenset[LoginKind] = _ALL_LOGINS,
    tls: bool = True,
    unsupported_because: str = "",
    extra_auth_keys: frozenset[str] = frozenset(),
) -> ConnectionSettings:
    """Read the shared keys of an outbound connection block.

    Only ``url``, ``timeout_s``, ``auth`` and ``tls`` are read; every other key
    is left to the consumer. The call reads no environment and no file.

    Args:
        block: The settings mapping; None or ``{}`` reads as nothing set.
        where: The block's dotted key, named in every message.
        logins: The login forms this consumer can send.
        tls: Whether this consumer can apply a per-connection CA.
        unsupported_because: Why a refused login or ``tls:`` cannot be sent.
            Required whenever ``logins`` or ``tls`` is restricted.
        extra_auth_keys: Keys under ``auth:`` this consumer reads itself beside
            the login. They are accepted and left to the consumer; they never
            count as a login.

    Raises:
        TypeError: ``logins`` or ``tls`` is restricted without a reason.
        ConfigurationError: The block breaks the shape. The message starts with the key
            and never repeats a value from under ``auth:`` or a url's userinfo.
    """
    if (logins != _ALL_LOGINS or not tls) and not unsupported_because:
        raise TypeError("a restricted login or tls reader needs unsupported_because")
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise ConfigurationError(
            f"`{where}` must be a mapping, got {type(block).__name__}", key=where
        )

    for old, new in _MOVED_KEYS.items():
        if old in block:
            raise ConfigurationError(
                f"`{where}.{old}` is spelled `{where}.{new}`", key=f"{where}.{old}"
            )
    if "headers" in block:
        raise ConfigurationError(
            f"`{where}.headers`: no free-form header map: name a login under `auth:`",
            key=f"{where}.headers",
        )

    return ConnectionSettings(
        where=where,
        url=_read_url(block.get("url"), where),
        timeout_s=_read_timeout(block.get("timeout_s"), where),
        login=_read_auth(block.get("auth"), where, logins, unsupported_because, extra_auth_keys),
        ca_bundle=_read_tls(block.get("tls"), where, tls, unsupported_because),
    )


def read_credential_env_names(
    block: Mapping[str, object] | None, *, where: str
) -> tuple[tuple[str, object], ...]:
    """The credential variables a block names under ``auth:``, with the key naming each.

    Returns ``(dotted key, name as written)`` pairs, ``auth.token_env`` before
    ``auth.password_env``. A username is not a secret and is never reported. The
    names are not checked as variable names and other ``auth:`` keys are not
    checked at all: :func:`read_connection_settings` refuses those when the
    connector reads the block. Only a flat spelling outside ``auth:`` counts
    for nothing here. The call reads no environment and no file.

    Args:
        block: The settings mapping; None or ``{}`` names no variable.
        where: The block's dotted key, named in every message.

    Raises:
        ConfigurationError: The block or its ``auth:`` is not a mapping, or ``auth:``
            names two logins.
    """
    if block is None:
        return ()
    if not isinstance(block, Mapping):
        raise ConfigurationError(
            f"`{where}` must be a mapping, got {type(block).__name__}", key=where
        )
    auth = block.get("auth")
    if auth is None:
        return ()
    if not isinstance(auth, Mapping):
        raise ConfigurationError(
            f"`{where}.auth` must be a mapping of {_AUTH_SHAPE}", key=f"{where}.auth"
        )
    if "token_env" in auth and ("username" in auth or "password_env" in auth):
        raise ConfigurationError(
            f"`{where}.auth` names two logins: use {_AUTH_SHAPE}, not both", key=f"{where}.auth"
        )
    return tuple((f"auth.{key}", auth[key]) for key in ("token_env", "password_env") if key in auth)


def read_ca_bundle(block: Mapping[str, object] | None, *, where: str) -> Path | None:
    """The CA file a block names under ``tls.ca_bundle``, as the connector will open it.

    The path follows the rules :func:`read_connection_settings` applies: it must
    be absolute, and a ``~`` path is refused. Nothing else in the block is
    checked, so a flat ``ca_bundle`` outside ``tls:`` names nothing here. The
    call reads no environment and no file: whether the file exists is decided
    when the connection is made.

    Args:
        block: The settings mapping; None or ``{}`` names no CA.
        where: The block's dotted key, named in every message.

    Raises:
        ConfigurationError: The block or its ``tls:`` is not a mapping, ``tls:`` holds a
            key other than ``ca_bundle``, or the path is blank, ``~``-spelled or
            relative.
    """
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ConfigurationError(
            f"`{where}` must be a mapping, got {type(block).__name__}", key=where
        )
    return _read_tls(block.get("tls"), where, True, "")


def _read_url(value: object, where: str) -> str | None:
    # A blank url reads as unset, so the consumer's own "url is required" applies.
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"`{where}.url` must be a string", key=f"{where}.url")
    try:
        password = urllib.parse.urlsplit(value).password
    except ValueError:
        raise ConfigurationError(f"`{where}.url` is not a valid url", key=f"{where}.url") from None
    if password is not None:
        raise ConfigurationError(
            f"`{where}.url`: a url may not carry a password; name it under `auth:`",
            key=f"{where}.url",
        )
    return value


def _read_timeout(value: object, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ConfigurationError(
            f"`{where}.timeout_s` must be a number of seconds greater than 0",
            key=f"{where}.timeout_s",
        )
    return float(value)


def _read_auth(
    value: object,
    where: str,
    logins: frozenset[LoginKind],
    unsupported_because: str,
    extra_auth_keys: frozenset[str],
) -> ServiceLogin | None:
    if value is None:
        return None
    key = f"`{where}.auth`"
    if not logins:
        raise ConfigurationError(
            f"{key} is not accepted here: {unsupported_because}", key=f"{where}.auth"
        )
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping of {_AUTH_SHAPE}", key=f"{where}.auth")
    for k in value:
        if k not in _AUTH_KEYS and k not in extra_auth_keys:
            hint = (
                "; secrets are named by environment variable, never written in config"
                if k in _SECRET_LIKE_KEYS
                else ""
            )
            raise ConfigurationError(
                f"`{where}.auth.{k}` is not a login key (use {_AUTH_SHAPE}){hint}",
                key=f"{where}.auth.{k}",
            )
    has_token = "token_env" in value
    has_user = "username" in value
    has_password = "password_env" in value
    if has_token and (has_user or has_password):
        raise ConfigurationError(
            f"{key} names two logins: use {_AUTH_SHAPE}, not both", key=f"{where}.auth"
        )
    if not (has_token or has_user or has_password):
        raise ConfigurationError(f"{key} names no login: set {_AUTH_SHAPE}", key=f"{where}.auth")
    if has_user != has_password:
        missing = "password_env" if has_user else "username"
        raise ConfigurationError(
            f"`{where}.auth.{missing}` is missing: a password login needs both keys",
            key=f"{where}.auth.{missing}",
        )

    for env_key in ("token_env", "password_env"):
        if env_key in value:
            name = value[env_key]
            if not isinstance(name, str) or not ENV_NAME_RE.match(name):
                raise ConfigurationError(
                    f"`{where}.auth.{env_key}` must name an environment variable (letters, "
                    "digits and underscores, not starting with a digit)",
                    key=f"{where}.auth.{env_key}",
                )

    kind: LoginKind = "token" if has_token else "password"
    if kind not in logins:
        raise ConfigurationError(
            f"{key} is not accepted here: {unsupported_because}", key=f"{where}.auth"
        )
    if kind == "token":
        return TokenLogin(token_env=value["token_env"])
    username = value["username"]
    if not isinstance(username, str) or not username.strip():
        raise ConfigurationError(
            f"`{where}.auth.username` must be a non-empty string", key=f"{where}.auth.username"
        )
    if username != username.strip():
        # A login is validated, never normalised: " bob" is not taken for "bob".
        raise ConfigurationError(
            f"`{where}.auth.username` is {username!r}: a login has no space around it",
            key=f"{where}.auth.username",
        )
    return PasswordLogin(username=username, password_env=value["password_env"])


def _read_tls(value: object, where: str, tls: bool, unsupported_because: str) -> Path | None:
    if value is None:
        return None
    key = f"`{where}.tls`"
    if not tls:
        raise ConfigurationError(
            f"{key} is not accepted here: {unsupported_because}", key=f"{where}.tls"
        )
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"{key} must be a mapping whose only key is `ca_bundle`", key=f"{where}.tls"
        )
    for k in value:
        if k not in _TLS_KEYS:
            hint = (
                "; no setting turns certificate verification off" if k in _VERIFY_OFF_KEYS else ""
            )
            raise ConfigurationError(
                f"`{where}.tls.{k}` is not a tls key (use `ca_bundle`){hint}",
                key=f"{where}.tls.{k}",
            )
    raw = value.get("ca_bundle")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(
            f"`{where}.tls.ca_bundle` must be a non-empty path", key=f"{where}.tls.ca_bundle"
        )
    if raw.startswith("~"):
        raise ConfigurationError(
            f"`{where}.tls.ca_bundle` must be an absolute path, got {raw!r}: `~` expands "
            "against each process's own home, so spell the full path",
            key=f"{where}.tls.ca_bundle",
        )
    path = Path(raw)
    if not path.is_absolute():
        raise ConfigurationError(
            f"`{where}.tls.ca_bundle` must be an absolute path, got {raw!r}: a connector "
            "has no project root to anchor a relative one",
            key=f"{where}.tls.ca_bundle",
        )
    return path


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), parts.port or _DEFAULT_PORTS.get(scheme)


class _OriginAuthorization(urllib.request.BaseHandler):
    """Adds one ``Authorization`` value to requests for one origin only.

    The header is unredirected and the origin is checked on every request, so a
    redirect to another scheme, host or port carries no credential.
    """

    def __init__(self, url: str, value: str) -> None:
        self._origin = _origin(url)
        self._value = value

    @override
    def __repr__(self) -> str:
        scheme, host, port = self._origin
        return f"<_OriginAuthorization {scheme}://{host}:{port}>"

    def _add(self, req: urllib.request.Request) -> urllib.request.Request:
        if _origin(req.full_url) == self._origin and not req.has_header("Authorization"):
            req.add_unredirected_header("Authorization", self._value)
        return req

    http_request = _add
    https_request = _add


def urllib_opener(
    settings: ConnectionSettings, *, environ: Mapping[str, str] | None = None
) -> urllib.request.OpenerDirector:
    """Build the one ``urllib`` opener an HTTP client sends every request through.

    It trusts the block's CA (or the default store) and adds the login to
    requests for the configured origin only. Building it opens no socket.

    Raises:
        ConfigurationError: The block has no http(s) url, or its CA file cannot be loaded.
        ConnectionError: The login's variable is unset.
    """
    url = settings.url or ""
    if urllib.parse.urlsplit(url).scheme.lower() not in _DEFAULT_PORTS:
        raise ConfigurationError(
            f"`{settings.where}.url` must be an http or https url", key=f"{settings.where}.url"
        )
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPSHandler(context=settings.ssl_context())
    ]
    credential = settings.resolve_credential(environ)
    if credential is not None:
        handlers.append(_OriginAuthorization(url, credential.authorization_header()))
    return urllib.request.build_opener(*handlers)
