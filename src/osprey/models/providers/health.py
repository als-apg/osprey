"""The one health verdict a provider returns, and the one table that classifies its failures.

Every provider health check answers with a :class:`HealthResult`. Its ``reason``
is set whenever ``reachable`` is false, so a caller decides what an unhealthy
verdict means from the reason alone and never from the message, which is prose
for a person to read.

:func:`failure_reason` is the single place a library exception becomes a reason.
It matches exception *classes* and HTTP status codes; it never reads message
text. A library's classes are looked up only once something has imported it:
an exception can only be an instance of a class whose module is loaded, so a
library that is not installed, or not yet imported, simply never matches, and
classifying a failure never pays for importing one.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from .base import BaseProvider

FailureReason = Literal["unreachable", "auth", "model"]

_AUTH_STATUSES = frozenset({401, 403})
_MODEL_STATUSES = frozenset({404})


class HealthResult(NamedTuple):
    """A provider health verdict.

    Attributes:
        reachable: True when the endpoint answered as expected, False when it
            did not, None when it was not checked at all.
        message: Human-readable detail. Never parsed.
        reason: Why the verdict is unhealthy — ``unreachable``, ``auth``,
            ``model`` or ``config``. Always set when ``reachable`` is False;
            None otherwise.
    """

    reachable: bool | None
    message: str
    reason: str | None


def library_exception_classes(module_name: str, *names: str) -> tuple[type[BaseException], ...]:
    """The named exception classes of *module_name*, or none while it is not imported.

    Never imports *module_name*: an exception can only be an instance of one of
    its classes once the module is loaded, and a cold import of a library such
    as LiteLLM takes seconds on whatever thread classifies the failure, which
    for a search lane is the event loop.
    """
    module = sys.modules.get(module_name)
    if module is None:
        return ()
    found = []
    for name in names:
        cls = getattr(module, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            found.append(cls)
    return tuple(found)


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status carried by a ``requests`` or ``httpx`` status error, if any."""
    status_errors = library_exception_classes("requests", "HTTPError") + library_exception_classes(
        "httpx", "HTTPStatusError"
    )
    if not (status_errors and isinstance(exc, status_errors)):
        return None
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def failure_reason(exc: BaseException) -> FailureReason | None:
    """Classify a provider failure by its exception class and HTTP status.

    ============================================================  ===============
    Exception                                                     Reason
    ============================================================  ===============
    builtin ``ConnectionError``, ``requests.ConnectionError``,    ``unreachable``
    ``httpx.ConnectError``, LiteLLM ``APIConnectionError``
    HTTP 401/403 (``requests``/``httpx`` status error), LiteLLM   ``auth``
    ``AuthenticationError`` / ``PermissionDeniedError``
    HTTP 404 (``requests``/``httpx`` status error), LiteLLM       ``model``
    ``NotFoundError``
    anything else                                                 None
    ============================================================  ===============

    Args:
        exc: The exception a provider call raised.

    Returns:
        The reason, or None when the table does not know the exception.
    """
    unreachable = (
        (ConnectionError,)
        + library_exception_classes("requests", "ConnectionError")
        + library_exception_classes("httpx", "ConnectError")
        + library_exception_classes("litellm", "APIConnectionError")
    )
    if isinstance(exc, unreachable):
        return "unreachable"

    if isinstance(
        exc, library_exception_classes("litellm", "AuthenticationError", "PermissionDeniedError")
    ):
        return "auth"
    if isinstance(exc, library_exception_classes("litellm", "NotFoundError")):
        return "model"

    status = _http_status(exc)
    if status in _AUTH_STATUSES:
        return "auth"
    if status in _MODEL_STATUSES:
        return "model"
    return None


_ANTHROPIC_VERSION = "2023-06-01"
_NOT_PROBED = HealthResult(None, "not probed", None)


def _models_url(base_url: str) -> str:
    """The listing URL under *base_url*: ``…/v1/models`` whether or not it ends in ``/v1``."""
    base = base_url.rstrip("/")
    return base + "/models" if base.endswith("/v1") else base + "/v1/models"


def probe_models_endpoint(
    provider_cls: type[BaseProvider],
    base_url: str | None,
    api_key: str | None,
    model_id: str | None = None,
    timeout: float = 5,
) -> HealthResult:
    """Ask a chat route's model listing whether it answers, and serves *model_id*.

    Only a provider class declaring ``models_probe`` is probed; any other, or
    one with no resolvable probe base, answers ``HealthResult(None, "not
    probed", None)``. The probe base is the provider's effective base URL, else
    its ``models_probe_base_url``, else its ``default_base_url``. No model is
    ever called.

    ===========================================  ===============
    Outcome                                      Reason
    ===========================================  ===============
    HTTP 401/403                                 ``auth``
    connection error, timeout, 404, 429, 5xx     ``unreachable``
    200 whose ``data[].id`` lacks *model_id*     ``model``
    200 (serving *model_id*, when given)         None (healthy)
    ===========================================  ===============

    A 404 here means the listing itself is missing, so the base URL is wrong:
    that is ``unreachable``, unlike a 404 from a model call.

    Args:
        provider_cls: The provider adapter class.
        base_url: The caller's base URL, usually from deployment config.
        api_key: The caller's key; the provider's keyless placeholder is sent
            for a route that needs none, and no auth header when it is absent.
        model_id: Model the listing must serve; unchecked when None.
        timeout: Request timeout in seconds.

    Returns:
        The verdict. Never raises.
    """
    kind = getattr(provider_cls, "models_probe", None)
    if kind is None:
        return _NOT_PROBED
    try:
        base = (
            provider_cls.effective_base_url(base_url)
            or getattr(provider_cls, "models_probe_base_url", None)
            or getattr(provider_cls, "default_base_url", None)
        )
        if not base:
            return _NOT_PROBED
        url = _models_url(base)

        headers: dict[str, str] = {}
        key = provider_cls.effective_api_key(api_key)
        if key:
            headers["Authorization"] = f"Bearer {key}"
            if kind == "anthropic":
                headers["x-api-key"] = key
        if kind == "anthropic":
            headers["anthropic-version"] = _ANTHROPIC_VERSION

        import httpx

        try:
            response = httpx.get(url, headers=headers, timeout=timeout)
        except httpx.TimeoutException as e:
            return HealthResult(False, f"model listing at {url} timed out: {e}", "unreachable")
        except Exception as e:
            reason = failure_reason(e) or "unreachable"
            return HealthResult(False, f"model listing at {url} failed: {e}", reason)

        status = response.status_code
        if status in _AUTH_STATUSES:
            return HealthResult(
                False, f"model listing at {url} refused the key (HTTP {status})", "auth"
            )
        if status != 200:
            return HealthResult(
                False, f"model listing at {url} answered HTTP {status}", "unreachable"
            )
        if model_id is None:
            return HealthResult(True, f"model listing at {url} answered", None)

        served = {
            item.get("id") for item in response.json().get("data", []) if isinstance(item, dict)
        }
        if model_id not in served:
            return HealthResult(False, f"model {model_id!r} is not served at {url}", "model")
        return HealthResult(True, f"model {model_id!r} is served at {url}", None)
    except Exception as e:
        return HealthResult(False, f"model listing probe failed: {e}", "unreachable")
