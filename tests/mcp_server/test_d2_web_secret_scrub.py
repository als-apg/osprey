"""D2 invariant pin: the web server's own credentials never reach a child it spawns.

The web server holds several credentials that authenticate *it*: the operator
secret that unlocks a browser session, the panel token that unlocks the
panel-coordination routes, the event-dispatcher and dispatch-worker tokens
(the two legs of the dispatch chain), and the ``*_LAUNCH_TOKEN`` family that
arms a bridge's plan-launch endpoint. Every one of them is a value
that lets its holder act as the server, from outside the tool layer whose
in-tool ``writes_enabled`` re-checks are the actual write-safety authority.

The server also spawns children an agent can influence: the PTY child that runs
an interactive agent session, the SDK operator subprocess, and three sandboxes
that execute agent-authored Python (``python_executor.executor``,
``workspace.execution.sandbox_executor``,
``services.bluesky_bridge.plan_validation``). D2 is the invariant that no
sensitive name crosses from the first list into the second.

**These tests run the real machinery.** No scrub function is mocked and no
child environment is synthesized here: three of them spawn an actual
subprocess and read back what it could actually see, and the credential-pop
test drives a real ASGI request through the real auth middleware. The probe
set is derived from :data:`SENSITIVE_ENV_EXACT` and
:data:`SENSITIVE_ENV_SUFFIXES` rather than typed out, so a credential added to
that module is probed by these tests on the day it is added rather than the
day someone remembers to widen them.

``PATH`` and ``OSPREY_WEB_PORT`` are asserted present in every child
environment as negative controls. Without them a bug that handed a child an
empty environment would make every absence assertion below pass while proving
nothing at all.

**The honesty split.** D2 is not "every sensitive name disappears everywhere".
Two names are genuinely closed in the serving process — the operator secret and
the panel token are taken out of ``os.environ`` and kept only in
``web_auth``'s memory, first by ``_populate`` and then, on every launch shape,
by ``close_env_carriers`` at app construction (which is what covers the
direct-serve ``osprey web``, where a launcher deliberately re-published the
secret for a child that turned out to be this same process). The
event-dispatcher token, the dispatch-worker token and the ``*_LAUNCH_TOKEN``
family deliberately stay in the web server's ``os.environ``, because the server
must inject them upstream into the services it fronts; for those, the strip on
the spawn paths is defence-in-depth for subprocess children and nothing more. That split is encoded below as explicit expected behavior
(:data:`_POPPED_IN_WEB_SERVER` / :data:`_RETAINED_FOR_PROXY_UPSTREAM`), so an
edit that changes which side a name falls on trips a test instead of silently
widening or narrowing what the server keeps.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
import yaml

from osprey.utils.sensitive_env import SENSITIVE_ENV_EXACT, SENSITIVE_ENV_SUFFIXES

# --------------------------------------------------------------------------- #
# Probe set and negative controls
# --------------------------------------------------------------------------- #

#: Operational variables that MUST survive every strip. A child handed an empty
#: environment would satisfy every absence assertion in this module, so each
#: test asserts these are still there before believing its own result.
_NEGATIVE_CONTROLS = ("PATH", "OSPREY_WEB_PORT")

#: Value given to the negative-control port variable; any non-empty string works.
_WEB_PORT_VALUE = "8080"

#: Names the web server's own ``os.environ`` must NOT still hold once credentials
#: have been populated — ``web_auth._populate`` pops both.
_POPPED_IN_WEB_SERVER = frozenset({"OSPREY_TERMINAL_SECRET", "OSPREY_PANEL_TOKEN"})

#: Names the web server's own ``os.environ`` deliberately KEEPS. The server
#: injects these upstream into the services it proxies, so popping them would
#: break the proxy; the spawn-path strip is defence-in-depth for subprocess
#: children, not a boundary in the serving process itself.
#:
#: ``DISPATCH_WORKER_TOKEN`` is on this side for the same reason its peer
#: ``EVENT_DISPATCHER_TOKEN`` is: it authenticates the dispatcher to a worker's
#: ``/dispatch`` API, the dispatch worker verifies incoming calls against its own
#: ``os.environ`` copy, and every compose template that sets one sets the other.
#: Nothing pops it anywhere.
_RETAINED_FOR_PROXY_UPSTREAM = frozenset({"EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"})


def _probe_env() -> dict[str, str]:
    """Build one distinctly-valued parent-env entry for every sensitive name.

    Derived from the canonical lists rather than hand-written: every name in
    :data:`SENSITIVE_ENV_EXACT` gets an entry, and every suffix in
    :data:`SENSITIVE_ENV_SUFFIXES` gets one concrete ``BLUESKY``-prefixed
    instance. A credential added to :mod:`osprey.utils.sensitive_env` is
    therefore probed by every test in this module without any edit here.

    Returns:
        Mapping of variable name to a unique, recognisable secret value.
    """
    env = {name: f"d2-secret-value-for-{name}" for name in SENSITIVE_ENV_EXACT}
    for suffix in SENSITIVE_ENV_SUFFIXES:
        name = f"BLUESKY{suffix}"
        env[name] = f"d2-secret-value-for-{name}"
    return env


@pytest.fixture
def sensitive_parent_env(monkeypatch) -> dict[str, str]:
    """Set every sensitive credential plus the negative controls in the parent env.

    ``monkeypatch.setenv`` records the prior value of each name, so the process
    environment is restored on teardown even for the names the code under test
    pops out of ``os.environ`` itself.

    Returns:
        The probe mapping that was installed (name to secret value).
    """
    probe = _probe_env()
    for name, value in probe.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OSPREY_WEB_PORT", _WEB_PORT_VALUE)
    return probe


def _assert_no_sensitive_names(env_keys, probe: dict[str, str], where: str) -> None:
    """Assert no probed credential name appears in ``env_keys``.

    Args:
        env_keys: The child environment's variable names (any container).
        probe: The mapping returned by :func:`_probe_env`.
        where: Human-readable name of the spawn path, used in failure messages.
    """
    keys = set(env_keys)
    leaked = sorted(name for name in probe if name in keys)
    assert not leaked, f"{where}: sensitive credential(s) reached the child: {leaked}"


def _assert_negative_controls(env_keys, where: str) -> None:
    """Assert the operational variables survived the strip in ``env_keys``.

    Args:
        env_keys: The child environment's variable names (any container).
        where: Human-readable name of the spawn path, used in failure messages.
    """
    keys = set(env_keys)
    missing = [name for name in _NEGATIVE_CONTROLS if name not in keys]
    assert not missing, (
        f"{where}: negative control(s) {missing} were stripped too — this child got a "
        "gutted environment, so the absence assertions above prove nothing"
    )


# --------------------------------------------------------------------------- #
# Spawn path: the PTY child (complete replacement environment)
# --------------------------------------------------------------------------- #


def test_pty_child_env_holds_no_sensitive_credential(sensitive_parent_env):
    """``build_pty_env`` result — the PTY child's complete ``env=`` — is clean.

    This is the strongest of the launch paths: the returned dict becomes the
    child's entire environment, so a name absent here is absent from the agent
    session and from every MCP server it spawns.
    """
    from osprey.interfaces.web_terminal.pty_manager import build_pty_env

    env = build_pty_env()

    _assert_no_sensitive_names(env, sensitive_parent_env, "build_pty_env")
    _assert_negative_controls(env, "build_pty_env")


def test_pty_child_env_secret_values_are_absent_too(sensitive_parent_env):
    """No probed secret *value* survives under a renamed key either.

    Absence of the names is the invariant, but a future helper that copied a
    credential into a differently-named variable would slip past a name-only
    check. Comparing values catches that shape.
    """
    from osprey.interfaces.web_terminal.pty_manager import build_pty_env

    env = build_pty_env()

    leaked = sorted(
        f"{key}={name}"
        for key, value in env.items()
        for name, secret in sensitive_parent_env.items()
        if value == secret
    )
    assert not leaked, f"build_pty_env: secret value(s) survived under another key: {leaked}"


def test_sdk_child_env_holds_no_sensitive_credential(sensitive_parent_env):
    """``build_clean_env`` — the SDK path's ``ClaudeAgentOptions.env`` — is clean.

    Honest scope: the SDK overlays this dict onto ``os.environ`` rather than
    substituting for it, so omitting a name here does not by itself keep it
    from the child. What closes the operator secret and the panel token on this
    path is the ``os.environ`` pop asserted further down; this test pins the
    defence-in-depth layer, not a boundary.
    """
    from osprey.agent_runner.clean_env import build_clean_env

    env = build_clean_env()

    _assert_no_sensitive_names(env, sensitive_parent_env, "build_clean_env")
    _assert_negative_controls(env, "build_clean_env")


# --------------------------------------------------------------------------- #
# Spawn path: the python-executor sandbox (real subprocess)
# --------------------------------------------------------------------------- #

_DUMP_ENV_CODE = (
    "import json, os\n"
    "print('ENV_KEYS_JSON_START')\n"
    "print(json.dumps(sorted(os.environ.keys())))\n"
    "print('ENV_KEYS_JSON_END')\n"
)


def _extract_env_keys(stdout: str) -> list[str]:
    """Pull the JSON-encoded sorted env key list out of a sandbox's stdout.

    Mirrors the helper in ``test_executor_token_regression`` so both pins read
    the same probe output the same way.

    Args:
        stdout: Captured standard output of a sandbox that ran
            :data:`_DUMP_ENV_CODE`.

    Returns:
        The sorted environment variable names the sandboxed code could see.
    """
    start = stdout.index("ENV_KEYS_JSON_START") + len("ENV_KEYS_JSON_START")
    end = stdout.index("ENV_KEYS_JSON_END")
    return json.loads(stdout[start:end])


@pytest.fixture
def reset_config_caches(monkeypatch):
    """Reset every config cache around a test that writes its own ``config.yml``.

    The python executor resolves its execution method through the shared config
    caches; a test that installs a temporary config must not inherit — or leave
    behind — another test's cached configuration.
    """
    from osprey.utils.workspace import reset_config_cache

    reset_config_cache()

    import osprey.utils.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", None)
    monkeypatch.setattr(_cfg, "_default_configurable", None)
    saved_cache = _cfg._config_cache.copy()
    _cfg._config_cache.clear()

    yield

    reset_config_cache()
    _cfg._config_cache.clear()
    _cfg._config_cache.update(saved_cache)


async def test_executor_sandbox_subprocess_sees_no_sensitive_credential(
    tmp_path, monkeypatch, sensitive_parent_env, reset_config_caches
):
    """Real python-executor subprocess: agent code cannot read any credential.

    The sandboxed code introspects its own ``os.environ`` and reports back, so
    this catches a scrub that only fails at runtime — a wrong dict identity, an
    encoding fault — where inspecting a mocked ``env=`` kwarg would not.
    """
    from osprey.mcp_server.python_executor.executor import execute_code

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "control_system": {"type": "mock", "limits_checking": {"enabled": False}},
                "execution": {"execution_method": "subprocess"},
                "python_executor": {"execution_timeout_seconds": 60},
            }
        )
    )

    result = await execute_code(_DUMP_ENV_CODE, "readonly", "D2 env probe")

    assert result.success, f"sandbox execution failed: {result.error_message}\n{result.stderr}"
    env_keys = _extract_env_keys(result.stdout)
    _assert_no_sensitive_names(env_keys, sensitive_parent_env, "python_executor sandbox")
    _assert_negative_controls(env_keys, "python_executor sandbox")
    for secret in sensitive_parent_env.values():
        assert secret not in result.stdout
        assert secret not in result.stderr


# --------------------------------------------------------------------------- #
# Spawn path: the workspace visualization sandbox (real subprocess)
# --------------------------------------------------------------------------- #


async def test_workspace_sandbox_subprocess_sees_no_sensitive_credential(
    tmp_path, sensitive_parent_env
):
    """Real workspace-sandbox subprocess: agent code cannot read any credential.

    The lighter visualization-only sandbox is a separate spawn seam from the
    python executor and could drift from it; this drives it end to end for the
    same reason.
    """
    from osprey.mcp_server.workspace.execution.sandbox_executor import execute_sandbox_code

    execution_folder = tmp_path / "d2_execution"
    execution_folder.mkdir()

    result = await execute_sandbox_code(_DUMP_ENV_CODE, execution_folder, timeout=120)

    assert result.success, f"sandbox execution failed: {result.error_message}\n{result.stderr}"
    env_keys = _extract_env_keys(result.stdout)
    _assert_no_sensitive_names(env_keys, sensitive_parent_env, "workspace sandbox")
    _assert_negative_controls(env_keys, "workspace sandbox")
    for secret in sensitive_parent_env.values():
        assert secret not in result.stdout
        assert secret not in result.stderr


# --------------------------------------------------------------------------- #
# Spawn path: the Bluesky plan-validation dry run (real subprocess)
# --------------------------------------------------------------------------- #

#: A body that clears stages 1 and 2 (no imports, no CA constructs) so stage 3
#: actually spawns its subprocess. Whether the dry run then *succeeds* is
#: irrelevant here — the environment is built and handed over before the plan
#: runs at all.
_TRIVIAL_PLAN_BODY = "def build_plan():\n    yield\n"


async def test_plan_validation_dry_run_subprocess_gets_no_sensitive_credential(
    monkeypatch, sensitive_parent_env
):
    """Real plan-validation dry run: the spawned subprocess's env holds no credential.

    A plan body cannot import ``os`` — the authoring allowlist forbids it — so
    unlike the two sandboxes above this path cannot report its own environment
    back. Instead the real ``asyncio.create_subprocess_exec`` is wrapped in a
    passthrough recorder: the scrub runs unmocked, the subprocess really is
    spawned, and the assertion is made against the exact ``env=`` mapping that
    real spawn received. ``monkeypatch`` restores the stdlib function on
    teardown.

    A future change that rejects this body in stage 1 or 2 would spawn nothing
    and fail this test on the empty recording, which is the correct failure —
    a silently-skipped probe is worse than a red test.
    """
    from osprey.services.bluesky_bridge import plan_validation

    captured: list[dict[str, str]] = []
    real_spawn = asyncio.create_subprocess_exec

    async def recording_spawn(*args, **kwargs):
        captured.append(dict(kwargs.get("env") or {}))
        return await real_spawn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording_spawn)

    await plan_validation.validate_plan(
        _TRIVIAL_PLAN_BODY, plan_name="d2_env_probe", dry_run_timeout=20.0
    )

    assert captured, (
        "plan validation spawned no dry-run subprocess, so its environment was never "
        "observed — stages 1/2 must have rejected the probe body"
    )
    for env in captured:
        _assert_no_sensitive_names(env, sensitive_parent_env, "plan-validation dry run")
        _assert_negative_controls(env, "plan-validation dry run")


# --------------------------------------------------------------------------- #
# The serving process: which names it pops, and which it deliberately keeps
# --------------------------------------------------------------------------- #


@pytest.fixture
def unpopulated_web_credentials():
    """Start from an unpopulated process-wide credential holder, and restore it.

    ``_populate`` runs at most once per process. A test that exercises the
    environment-driven population path has to reach it, and must not leave the
    holder it settled behind for the next test.
    """
    from osprey.interfaces.web_auth import reset_web_credentials

    reset_web_credentials()
    yield
    reset_web_credentials()


async def _drive_one_request_through_auth_middleware() -> int:
    """Send one unauthenticated ASGI request through the real auth middleware.

    Uses a raw scope rather than a test client: the middleware answers before
    any application exists, so a scope is the honest input, and nothing can
    inject default credentials behind the test's back. Reaching the middleware
    is what makes it call ``get_web_credentials`` — the real trigger for
    ``_populate`` and its ``os.environ.pop``.

    Returns:
        The HTTP status the middleware answered with. A refusal (401) is the
        expected outcome and is itself evidence the credential path ran.
    """
    from osprey.interfaces.common_middleware import WebAuthMiddleware

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"downstream"})

    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = WebAuthMiddleware(downstream)
    await middleware(
        {"type": "http", "path": "/api/d2-probe", "method": "GET", "headers": []},
        receive,
        send,
    )
    return sent[0]["status"]


async def test_serving_process_pops_operator_secret_and_panel_token(
    sensitive_parent_env, unpopulated_web_credentials
):
    """Once credentials are populated, the popped names are gone from ``os.environ``.

    This is what actually closes those two names on the SDK path, where the
    child inherits the parent's ``os.environ`` regardless of what the launch
    helper chose to copy.
    """
    status = await _drive_one_request_through_auth_middleware()
    assert status == 401, (
        "the probe request was not refused, so the auth middleware may not have "
        "consulted the credentials at all"
    )

    still_present = sorted(name for name in _POPPED_IN_WEB_SERVER if name in os.environ)
    assert not still_present, (
        f"{still_present} survived credential population in the serving process's own "
        "os.environ — every SDK-spawned child would inherit it"
    )


async def test_serving_process_keeps_proxy_upstream_credentials(
    sensitive_parent_env, unpopulated_web_credentials
):
    """The proxy-upstream credentials deliberately REMAIN in ``os.environ``.

    Stating the other half of the split explicitly is the point: the web server
    injects the event-dispatcher token and the ``*_LAUNCH_TOKEN`` family
    upstream into the services it fronts, so popping them would break the proxy.
    For those names the spawn-path strip is defence-in-depth for subprocess
    children and nothing more — the safety authority stays the target
    endpoint's own in-tool re-checks. A change that starts popping them should
    fail here and be argued for, not land unnoticed.
    """
    await _drive_one_request_through_auth_middleware()

    expected_retained = sorted(
        name
        for name in sensitive_parent_env
        if name in _RETAINED_FOR_PROXY_UPSTREAM or name.endswith(SENSITIVE_ENV_SUFFIXES)
    )
    assert expected_retained, "the probe set covers no proxy-upstream credential"
    missing = [name for name in expected_retained if name not in os.environ]
    assert not missing, (
        f"{missing} was removed from the serving process's os.environ; the reverse proxy "
        "reads it from there to inject upstream"
    )


def test_every_sensitive_name_is_classified_popped_or_retained():
    """Honesty regression: each exact credential is declared popped OR retained.

    Adding a name to :data:`SENSITIVE_ENV_EXACT` without deciding which side of
    the split it falls on fails here, which forces the decision to be made and
    written down rather than left implicit. The two sides must also stay
    disjoint — a name cannot be both popped from the serving process and kept
    there for the proxy.
    """
    declared = _POPPED_IN_WEB_SERVER | _RETAINED_FOR_PROXY_UPSTREAM
    unclassified = sorted(set(SENSITIVE_ENV_EXACT) - declared)
    assert not unclassified, (
        f"{unclassified} is in SENSITIVE_ENV_EXACT but is neither declared popped by "
        "web_auth._populate nor declared retained for proxy upstream injection; decide "
        "which it is and record it in this module"
    )
    stale = sorted(declared - set(SENSITIVE_ENV_EXACT))
    assert not stale, f"{stale} is classified here but no longer in SENSITIVE_ENV_EXACT"
    assert not (_POPPED_IN_WEB_SERVER & _RETAINED_FOR_PROXY_UPSTREAM)


def test_populate_pops_exactly_the_names_declared_popped(monkeypatch, unpopulated_web_credentials):
    """``_populate`` really pops every name :data:`_POPPED_IN_WEB_SERVER` claims.

    The companion to the classification test above: that one checks the split
    is *complete*, this one checks it is *true* of the code. A name moved onto
    the popped list without a matching ``os.environ.pop`` in ``_populate``
    fails here.
    """
    from osprey.interfaces import web_auth

    for name in _POPPED_IN_WEB_SERVER:
        monkeypatch.setenv(name, f"d2-secret-value-for-{name}")

    web_auth._populate()

    still_present = sorted(name for name in _POPPED_IN_WEB_SERVER if name in os.environ)
    assert not still_present, f"_populate did not pop {still_present} out of os.environ"


def test_close_env_carriers_covers_the_same_names(monkeypatch, unpopulated_web_credentials):
    """The construction-time close removes every name declared popped, too.

    ``_populate`` is not the only mechanism, and on the default ``osprey web``
    it is not the operative one: a launcher re-publishes the operator secret
    after population so a spawned worker can inherit it, and on the
    direct-serve path the "spawned worker" is this same process.
    ``close_env_carriers`` runs at every app construction and is what closes the
    carrier there, so a name on :data:`_POPPED_IN_WEB_SERVER` that only
    ``_populate`` handles would be closed on some launch shapes and not others.
    """
    from osprey.interfaces import web_auth

    web_auth.get_web_credentials()  # settle the holder first, as a launcher does
    for name in _POPPED_IN_WEB_SERVER:
        monkeypatch.setenv(name, f"d2-republished-value-for-{name}")

    web_auth.close_env_carriers()

    still_present = sorted(name for name in _POPPED_IN_WEB_SERVER if name in os.environ)
    assert not still_present, (
        f"close_env_carriers left {still_present} in os.environ; a launcher that "
        "re-published it for a child would leave it there for the serving process's life"
    )


def test_direct_serve_app_construction_closes_a_republished_secret(
    sensitive_parent_env, unpopulated_web_credentials, tmp_path
):
    """The real seam, in the real order: publish, construct, then check a child env.

    ``mint_and_announce`` is what every OSPREY launcher calls just before it
    serves or spawns, and it deliberately puts the operator secret back into
    ``os.environ``. On the foreground path the app is then built in that same
    process. Anything still in ``os.environ`` at that point is inherited by
    every SDK-spawned agent, because the SDK's ``env`` is an overlay — so the
    assertion is made against ``{**os.environ, **build_clean_env()}``, not
    against ``build_clean_env()`` alone.
    """
    from starlette.applications import Starlette

    from osprey.agent_runner.clean_env import build_clean_env
    from osprey.interfaces._app_setup import configure_interface_app
    from osprey.interfaces.web_auth import mint_and_announce

    mint_and_announce("127.0.0.1", 8087)
    assert "OSPREY_TERMINAL_SECRET" in os.environ, (
        "the launcher no longer publishes the carrier at all, so the --reload "
        "and --detach handoffs have nothing to inherit"
    )

    configure_interface_app(Starlette(), static_dir=tmp_path)

    sdk_child_env = {**os.environ, **build_clean_env()}
    still_present = sorted(name for name in _POPPED_IN_WEB_SERVER if name in sdk_child_env)
    assert not still_present, f"{still_present} reached an SDK-spawned agent"
    _assert_negative_controls(sdk_child_env, "direct-serve SDK child")


# --------------------------------------------------------------------------- #
# The panel token's legitimate carrier: the MCP server's bearer header
# --------------------------------------------------------------------------- #


def test_panel_bearer_header_is_sent_when_the_token_is_present(monkeypatch):
    """The MCP server turns ``OSPREY_PANEL_TOKEN`` into an ``Authorization`` header.

    D2 removes the panel token from environments that must not hold it. This
    pins the other side: the process that legitimately carries it does send it,
    so the panel routes stay reachable rather than being closed by accident.
    """
    from osprey.mcp_server.http import _panel_auth_headers

    monkeypatch.setenv("OSPREY_PANEL_TOKEN", "d2-panel-token-value")

    assert _panel_auth_headers() == {"Authorization": "Bearer d2-panel-token-value"}


@pytest.mark.parametrize("value", ["", "   "])
def test_panel_bearer_header_is_omitted_when_the_token_is_blank(monkeypatch, value):
    """A blank carrier yields no header, never a bare ``Bearer``.

    An uninterpolated compose variable arrives as an empty string; sending
    ``"Bearer "`` would be a credential-shaped lie the route has to reject.
    """
    from osprey.mcp_server.http import _panel_auth_headers

    monkeypatch.setenv("OSPREY_PANEL_TOKEN", value)

    assert _panel_auth_headers() == {}


def test_panel_bearer_header_is_omitted_when_the_token_is_unset(monkeypatch):
    """With no carrier at all there is no header to send."""
    from osprey.mcp_server.http import _panel_auth_headers

    monkeypatch.delenv("OSPREY_PANEL_TOKEN", raising=False)

    assert _panel_auth_headers() == {}
