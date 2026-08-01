"""Drain-gate health check: is it safe to drain queued work back into dispatch?

A bridge that has queued messages while the dispatch pipeline was unhealthy must
not replay them until the pipeline is demonstrably back. :func:`gate_open` is
that go/no-go check — it returns ``True`` only when all of the following hold
*right now*:

  (a) the dispatcher's ``GET /health`` is ``200`` with ``status == "ok"``;
  (b) the worker's ``GET /health`` likewise;
  (c) **only where an alert host is configured** — the GitLab instance reports
      **no** open ``osprey-alert`` issue for the project (an open issue with that
      label means the watchdog currently considers the deployment broken).

Check (c) is **opt-in**: it runs only when both ``cfg.gitlab_url`` and
``cfg.gitlab_issues_token`` are non-empty. A site that files no alert issues has
no alert host to ask, and asking a host that is not there would fail-close the
gate forever — parked messages would never replay and every outage would end in
give-up notices. Unconfigured, the gate is (a) AND (b) alone.

Where the checks *do* run they are **fail-closed**: any non-200, timeout,
malformed body, or exception on ANY of them yields ``False``. A GitLab *API*
failure (unreachable, non-200, bad JSON) is therefore treated identically to a
still-open alert — both close the gate — but they are distinct causes: (c) is
only satisfied by a positive ``200`` + empty-list answer, never by an inability
to ask.

Connection behavior is the deployment's to decide: the client is built with
``trust_env=cfg.trust_env`` (default ``False``), so by default it ignores
``HTTP(S)_PROXY`` entirely and reaches the dispatcher, the worker, and the alert
host directly. A site whose bridge sits behind an egress proxy sets
``trust_env`` true once, in config. Tests inject their own
:class:`httpx.MockTransport` client.

Like every ``osprey.bridges.core`` module this carries ZERO osprey dispatch
imports and ZERO channel (chat/email) imports — the gate is pure stdlib + httpx.
"""

from __future__ import annotations

import logging
import urllib.parse

import httpx

logger = logging.getLogger(__name__)

# The watchdog's alert label. An open issue carrying this label means the
# deployment is currently flagged broken, so the gate must stay closed until it
# is resolved.
ALERT_LABEL = "osprey-alert"

# Short per-request timeout: a hung dispatcher/worker/GitLab must fail the gate
# closed quickly rather than stall the drain loop that calls it.
_TIMEOUT = 10.0


def _health_ok(http: httpx.Client, base_url: str) -> bool:
    """``True`` iff ``{base_url}/health`` returns 200 with ``status == "ok"``.

    Fail-closed: any non-200, non-JSON body, or transport error -> ``False``.
    """
    try:
        resp = http.get(f"{base_url}/health")
        if resp.status_code != 200:
            logger.warning("health check %s/health returned %d", base_url, resp.status_code)
            return False
        return bool(resp.json().get("status") == "ok")
    except Exception as exc:
        logger.warning("health check %s/health failed: %s", base_url, exc)
        return False


def alert_check_configured(cfg) -> bool:
    """``True`` iff this deployment has an alert host the gate can ask.

    Both the host and the token are required: a URL with no token cannot query
    the issues API, and a token with no URL has nothing to query. Either one
    missing means the site simply does not run the alert-issue check.
    """
    return bool(cfg.gitlab_url) and bool(cfg.gitlab_issues_token)


def _no_open_alert(http: httpx.Client, cfg) -> bool:
    """``True`` iff GitLab returns 200 with an EMPTY open-``osprey-alert`` list.

    A non-200, non-list body, or any exception (unreachable, timeout, bad JSON)
    is an API *failure* and returns ``False`` — deliberately indistinguishable
    at the gate from a genuinely open alert, but arrived at only by failing to
    get a clean empty answer, never by observing one.
    """
    project = urllib.parse.quote(cfg.gitlab_project, safe="")
    url = f"{cfg.gitlab_url}/api/v4/projects/{project}/issues"
    try:
        resp = http.get(
            url,
            params={"labels": ALERT_LABEL, "state": "opened"},
            headers={"PRIVATE-TOKEN": cfg.gitlab_issues_token},
        )
        if resp.status_code != 200:
            logger.warning("gitlab alert query returned %d", resp.status_code)
            return False
        issues = resp.json()
        if not isinstance(issues, list):
            logger.warning("gitlab alert query returned non-list body")
            return False
        if issues:
            logger.info("gate closed: %d open %s issue(s)", len(issues), ALERT_LABEL)
            return False
        return True
    except Exception as exc:
        logger.warning("gitlab alert query failed: %s", exc)
        return False


def gate_open(cfg, client: httpx.Client | None = None) -> bool:
    """Return ``True`` only if it is safe to drain queued work into dispatch.

    Runs the fail-closed dispatcher-health and worker-health checks against
    ``cfg``'s ``dispatcher_url`` / ``worker_url``, then — only where
    :func:`alert_check_configured` — the no-open-``osprey-alert`` check against
    ``gitlab_url`` / ``gitlab_project`` / ``gitlab_issues_token`` (the real
    :class:`~osprey.bridges.core.config.CoreConfig` field names). Any single
    check that runs and fails closes the gate. ``client`` is injectable for
    tests; otherwise one honoring ``cfg.trust_env`` is built and closed here.
    """
    own_client = client is None
    http = client or httpx.Client(timeout=_TIMEOUT, trust_env=cfg.trust_env)
    try:
        if not (_health_ok(http, cfg.dispatcher_url) and _health_ok(http, cfg.worker_url)):
            return False
        if not alert_check_configured(cfg):
            logger.debug("gate: no alert host configured; dispatcher+worker health only")
            return True
        return _no_open_alert(http, cfg)
    finally:
        if own_client:
            http.close()
