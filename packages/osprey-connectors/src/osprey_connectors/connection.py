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

:func:`read_connection_settings` is pure: it reads neither the environment nor
a file, so a build-time check can call it. The environment and the CA file are
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LoginKind = Literal["token", "password"]

#: Every login form the shape defines.
ALL_LOGINS: frozenset[LoginKind] = frozenset({"token", "password"})

#: What a ``*_env`` value must look like: an environment variable name.
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_AUTH_KEYS = ("token_env", "username", "password_env")
_SECRET_LIKE_KEYS = frozenset({"token", "password", "secret"})
_TLS_KEYS = ("ca_bundle",)
_VERIFY_OFF_KEYS = frozenset({"verify", "insecure", "verify_ssl"})

#: Flat spellings the nested shape replaced, mapped to the key that holds them now.
_MOVED_KEYS = {
    "timeout": "timeout_s",
    "token_env": "auth.token_env",
    "auth_token_env": "auth.token_env",
    "username": "auth.username",
    "password_env": "auth.password_env",
    "ca_bundle": "tls.ca_bundle",
}

_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True)
class Login:
    """A login named by a connection block. It holds variable names, never a secret."""

    kind: LoginKind
    token_env: str | None = None
    username: str | None = None
    password_env: str | None = None

    @property
    def env_names(self) -> tuple[str, ...]:
        """The environment variable that holds this login's secret."""
        name = self.token_env if self.kind == "token" else self.password_env
        return (name,) if name else ()


@dataclass(frozen=True)
class Credential:
    """A login resolved from the environment. Its repr never shows the secret."""

    kind: LoginKind
    username: str | None
    secret: str = field(repr=False)

    def authorization_header(self) -> str:
        """The ``Authorization`` value: a bearer token, or HTTP Basic sent up front."""
        if self.kind == "token":
            return f"Bearer {self.secret}"
        pair = f"{self.username}:{self.secret}".encode()
        return "Basic " + base64.b64encode(pair).decode("ascii")


@dataclass(frozen=True)
class ConnectionSettings:
    """An outbound connection block, read and checked. It never holds a secret."""

    where: str
    url: str | None = None
    timeout_s: float | None = None
    login: Login | None = None
    ca_bundle: Path | None = None

    @property
    def credential_env_names(self) -> tuple[str, ...]:
        """The environment variables the login's secret is read from; ``()`` without one."""
        return self.login.env_names if self.login else ()

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
        key = "token_env" if self.login.kind == "token" else "password_env"
        (name,) = self.login.env_names
        secret = env.get(name, "")
        if not secret.strip():
            raise ConnectionError(
                f"Environment variable '{name}' named by `{self.where}.auth.{key}` is not "
                "set, so there is no credential to send. Export it where the connector runs."
            )
        return Credential(kind=self.login.kind, username=self.login.username, secret=secret)

    def ssl_context(self) -> ssl.SSLContext:
        """The TLS context for this endpoint.

        Without ``tls.ca_bundle`` it is the default context, so the process
        trust store applies. With one, exactly that file is trusted for this
        endpoint, in place of the default roots. Verification stays on either way.

        Raises:
            ValueError: The CA file is missing, unreadable or holds no certificate.
        """
        if self.ca_bundle is None:
            return ssl.create_default_context()
        key = f"`{self.where}.tls.ca_bundle`"
        try:
            return ssl.create_default_context(cafile=str(self.ca_bundle))
        except FileNotFoundError as e:
            raise ValueError(f"{key} names {self.ca_bundle}, which does not exist.") from e
        except (IsADirectoryError, PermissionError) as e:
            raise ValueError(f"{key} names {self.ca_bundle}, which cannot be read: {e}") from e
        except ssl.SSLError as e:
            raise ValueError(
                f"{key} names {self.ca_bundle}, which holds no PEM certificate: {e}"
            ) from e


def read_connection_settings(
    block: Mapping[str, Any] | None,
    *,
    where: str,
    logins: frozenset[LoginKind] = ALL_LOGINS,
    tls: bool = True,
    unsupported_because: str = "",
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

    Raises:
        TypeError: ``logins`` or ``tls`` is restricted without a reason.
        ValueError: The block breaks the shape. The message starts with the key
            and never repeats a value from under ``auth:`` or a url's userinfo.
    """
    if (logins != ALL_LOGINS or not tls) and not unsupported_because:
        raise TypeError("a restricted login or tls reader needs unsupported_because")
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise ValueError(f"`{where}` must be a mapping, got {type(block).__name__}")

    for old, new in _MOVED_KEYS.items():
        if old in block:
            raise ValueError(f"`{where}.{old}` is spelled `{where}.{new}`")
    if "headers" in block:
        raise ValueError(f"`{where}.headers`: no free-form header map: name a login under `auth:`")

    return ConnectionSettings(
        where=where,
        url=_read_url(block.get("url"), where),
        timeout_s=_read_timeout(block.get("timeout_s"), where),
        login=_read_auth(block.get("auth"), where, logins, unsupported_because),
        ca_bundle=_read_tls(block.get("tls"), where, tls, unsupported_because),
    )


def _read_url(value: Any, where: str) -> str | None:
    # A blank url reads as unset, so the consumer's own "url is required" applies.
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError(f"`{where}.url` must be a string")
    try:
        password = urllib.parse.urlsplit(value).password
    except ValueError:
        raise ValueError(f"`{where}.url` is not a valid url") from None
    if password is not None:
        raise ValueError(f"`{where}.url`: a url may not carry a password; name it under `auth:`")
    return value


def _read_timeout(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"`{where}.timeout_s` must be a number of seconds greater than 0")
    return float(value)


def _read_auth(
    value: Any, where: str, logins: frozenset[LoginKind], unsupported_because: str
) -> Login | None:
    if value is None:
        return None
    key = f"`{where}.auth`"
    if not logins:
        raise ValueError(f"{key} is not accepted here: {unsupported_because}")
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{key} must be a mapping of `token_env`, or `username` and `password_env`"
        )
    for k in value:
        if k not in _AUTH_KEYS:
            hint = (
                "; secrets are named by environment variable, never written in config"
                if k in _SECRET_LIKE_KEYS
                else ""
            )
            raise ValueError(
                f"`{where}.auth.{k}` is not a login key (use `token_env`, or `username` "
                f"and `password_env`){hint}"
            )
    has_token = "token_env" in value
    has_user = "username" in value
    has_password = "password_env" in value
    if has_token and (has_user or has_password):
        raise ValueError(
            f"{key} names two logins: use `token_env`, or `username` and `password_env`, not both"
        )
    if not (has_token or has_user or has_password):
        raise ValueError(f"{key} names no login: set `token_env`, or `username` and `password_env`")
    if has_user != has_password:
        missing = "password_env" if has_user else "username"
        raise ValueError(f"`{where}.auth.{missing}` is missing: a password login needs both keys")

    for env_key in ("token_env", "password_env"):
        if env_key in value:
            name = value[env_key]
            if not isinstance(name, str) or not ENV_NAME_RE.match(name):
                raise ValueError(
                    f"`{where}.auth.{env_key}` must name an environment variable (letters, "
                    "digits and underscores, not starting with a digit)"
                )

    kind: LoginKind = "token" if has_token else "password"
    if kind not in logins:
        raise ValueError(f"{key} is not accepted here: {unsupported_because}")
    if kind == "token":
        return Login(kind="token", token_env=value["token_env"])
    username = value["username"]
    if not isinstance(username, str) or not username.strip():
        raise ValueError(f"`{where}.auth.username` must be a non-empty string")
    return Login(kind="password", username=username, password_env=value["password_env"])


def _read_tls(value: Any, where: str, tls: bool, unsupported_because: str) -> Path | None:
    if value is None:
        return None
    key = f"`{where}.tls`"
    if not tls:
        raise ValueError(f"{key} is not accepted here: {unsupported_because}")
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping whose only key is `ca_bundle`")
    for k in value:
        if k not in _TLS_KEYS:
            hint = (
                "; no setting turns certificate verification off" if k in _VERIFY_OFF_KEYS else ""
            )
            raise ValueError(f"`{where}.tls.{k}` is not a tls key (use `ca_bundle`){hint}")
    raw = value.get("ca_bundle")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"`{where}.tls.ca_bundle` must be a non-empty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"`{where}.tls.ca_bundle` must be an absolute path, got {raw!r}: a connector "
            "has no project root to anchor a relative one"
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
        ValueError: The block has no http(s) url, or its CA file cannot be loaded.
        ConnectionError: The login's variable is unset.
    """
    url = settings.url or ""
    if urllib.parse.urlsplit(url).scheme.lower() not in _DEFAULT_PORTS:
        raise ValueError(f"`{settings.where}.url` must be an http or https url")
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPSHandler(context=settings.ssl_context())
    ]
    credential = settings.resolve_credential(environ)
    if credential is not None:
        handlers.append(_OriginAuthorization(url, credential.authorization_header()))
    return urllib.request.build_opener(*handlers)
