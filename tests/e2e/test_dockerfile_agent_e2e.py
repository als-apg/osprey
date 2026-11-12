"""E2E: the image the generated reference Dockerfile builds serves an agent over HTTP.

Boots the image's actual ``CMD`` (``osprey web``), waits for ``/health``, and
drives one real agent turn through ``POST /api/chat`` with a live LLM call via
the als-apg provider. This is the only test that proves the *shipped*
entrypoint actually serves an agent — ``claude --version`` proves the binary
launches, not that the assembled image answers a prompt.

It spends model tokens, so it lives apart from the model-free image tests in
``test_dockerfile_e2e.py``: CI runs it in the ``dockerfile-agent-e2e`` lane,
gated like every model-spending lane, and fails that lane if it skips. The
image is built exactly as that module builds it (``_dockerfile_image``).

Needs ``ALS_APG_API_KEY`` (``requires_als_apg``); it skips without it, and
without docker.
"""

import http.cookiejar
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from tests.e2e import _dockerfile_image as image
from tests.e2e._dockerfile_image import PLATFORM_ARGS, WEB_SLOT
from tests.e2e._image_record import record_credential_mounts

pytestmark = image.MODULE_MARKS

HEALTH_TIMEOUT = 120  # server process up + first port bind
CHAT_TIMEOUT = 240  # one real LLM round-trip (haiku, single turn)

# A distinctive token the model is unlikely to emit by chance — proves the
# prompt reached a live model and came back, not that *some* text returned.
CHAT_MARKER = "OSPREY-E2E-OK"
CHAT_PROMPT = f"Respond with exactly this token and nothing else: {CHAT_MARKER}"


@pytest.fixture(scope="module")
def built_image(tmp_path_factory):
    """Build the repo + its reference image once for the whole module."""
    yield from image.built_image(tmp_path_factory)


# ── Functional: the shipped CMD actually serves an agent ─────────────────────


def _host_port(cid: str) -> int:
    """Resolve the ephemeral host port docker mapped to the image's web slot."""
    out = subprocess.run(
        ["docker", "port", cid, f"{WEB_SLOT}/tcp"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert out.returncode == 0, f"docker port failed: {out.stderr}"
    # Output like "127.0.0.1:54321" (possibly multiple lines for v4/v6).
    return int(out.stdout.strip().splitlines()[0].rsplit(":", 1)[1])


def _container_logs(cid: str) -> str:
    logs = subprocess.run(
        ["docker", "logs", "--tail", "60", cid], capture_output=True, text=True, timeout=60
    )
    return (logs.stdout + logs.stderr)[-4000:]


def _is_running(cid: str) -> bool:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", cid],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return out.stdout.strip() == "true"


def _wait_for_health(base_url: str, cid: str, timeout: float) -> None:
    """Poll ``GET /health`` until healthy, failing fast if the container dies."""
    deadline = time.monotonic() + timeout
    last_err = "no response"
    while time.monotonic() < deadline:
        if not _is_running(cid):
            pytest.fail(f"container exited before becoming healthy:\n{_container_logs(cid)}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200 and json.loads(resp.read()).get("status") == "healthy":
                    return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    pytest.fail(f"server never became healthy ({last_err}):\n{_container_logs(cid)}")


# The launcher prints its one-time login URL as `Open: http://<host>:<port>/?token=<secret>`
# (mint_and_announce). The container mints its OWN operator secret, so the token
# can only be recovered from its logs — the whole reason this parses the printed
# URL rather than assuming a bare, credential-less base URL.
_OPEN_URL_RE = re.compile(r"Open:\s*(http://\S+\?token=\S+)")


def _login_opener(base_url: str, cid: str) -> urllib.request.OpenerDirector:
    """Exchange the container's printed ``?token=`` login URL for a session cookie.

    Every ``/api/*`` route on the deployed image is now behind the web-auth gate,
    so an unauthenticated ``POST /api/chat`` is refused ``401``. A browser gets in
    by following the ``Open: …?token=…`` URL the launcher prints, which
    ``GET``s to a ``303`` that sets an ``HttpOnly`` session cookie; this does the
    same, returning a cookie-jar-backed opener that carries that session on every
    subsequent request.

    The token is read from the container's FULL logs (not the tailed view), since
    the ``Open:`` line is printed once at startup and would scroll out of a short
    tail by the time the agent turn runs.
    """
    logs = subprocess.run(["docker", "logs", cid], capture_output=True, text=True, timeout=15)
    match = _OPEN_URL_RE.search(logs.stdout + logs.stderr)
    assert match, f"container never printed an 'Open: …?token=…' login URL:\n{_container_logs(cid)}"
    login_url = match.group(1)
    # The token identifies the container's secret; the exchange itself must be
    # driven at the host-mapped base_url, so swap the authority the container
    # announced (its internal web slot) for the reachable one.
    token_query = urllib.parse.urlparse(login_url).query
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    with opener.open(f"{base_url}/?{token_query}", timeout=15) as resp:
        assert resp.status == 200, f"token exchange did not resolve to 200 (got {resp.status})"
    return opener


def _post_chat(
    base_url: str, opener: urllib.request.OpenerDirector, prompt: str, timeout: float
) -> dict:
    """POST a prompt to the buffered chat endpoint, return the JSON body.

    Sent through the cookie-bearing *opener* from :func:`_login_opener` with a
    same-origin ``Origin`` header: ``/api/chat`` is a mutating cookie-authenticated
    route, so the web-auth gate refuses it ``403`` without an ``Origin`` matching
    the app's external origin (derived from the request ``Host`` = *base_url*).
    """
    # chat_id is a required field — it keys the server-side ChatSessionPool.
    # A single fixed id is enough for this one-turn smoke test.
    req = urllib.request.Request(
        f"{base_url}/api/chat?stream=false",
        data=json.dumps({"prompt": prompt, "chat_id": "e2e"}).encode(),
        headers={"Content-Type": "application/json", "Origin": base_url},
        method="POST",
    )
    with opener.open(req, timeout=timeout) as resp:
        assert resp.status == 200, f"chat returned HTTP {resp.status}"
        return json.loads(resp.read())


@pytest.mark.requires_als_apg
def test_generated_image_serves_agent_over_http(built_image):
    """Boot the image's CMD and drive one real LLM turn through osprey web."""
    tag, repo, project_name, out_dir = built_image

    # Mirror the production run contract (`docker run --env-file .env`): pass
    # the raw provider secret, never a pre-resolved token. The full osprey web stack resolves
    # ${ALS_APG_API_KEY} from config at runtime and stands up an internal proxy
    # that authenticates upstream with it — injecting a pre-resolved
    # ANTHROPIC_AUTH_TOKEN would bypass that and leave config resolution (and
    # the proxy) without the key.
    api_key = os.environ["ALS_APG_API_KEY"]  # guaranteed present by requires_als_apg
    env_args = ["-e", f"ALS_APG_API_KEY={api_key}"]

    # The endpoint override rides the same contract. It is deliberately absent
    # from the image (build-time rendering never bakes the builder's
    # environment), so a run pointed at a non-default gateway must pass it in
    # here exactly as a deployment passes it via env_file — otherwise the
    # container dials the provider's built-in default instead.
    base_url = os.environ.get("ALS_APG_BASE_URL")
    if base_url:
        env_args += ["-e", f"ALS_APG_BASE_URL={base_url}"]

    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            *PLATFORM_ARGS,
            "-p",
            f"127.0.0.1:0:{WEB_SLOT}",
            *env_args,
            # The terminal holds the agent record: its login and service file
            # come in at run time, as the provider key does.
            *record_credential_mounts(repo, project_name, out_dir),
            tag,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, f"docker run failed: {run.stderr}"
    cid = run.stdout.strip()

    try:
        base_url = f"http://127.0.0.1:{_host_port(cid)}"
        _wait_for_health(base_url, cid, HEALTH_TIMEOUT)

        try:
            opener = _login_opener(base_url, cid)
            data = _post_chat(base_url, opener, CHAT_PROMPT, CHAT_TIMEOUT)
        except urllib.error.HTTPError as exc:
            pytest.fail(f"chat HTTP {exc.code}: {exc.read()[:2000]!r}\n{_container_logs(cid)}")

        assert data.get("is_error") is False, (
            f"agent returned an error: {data.get('error')}\n{_container_logs(cid)}"
        )
        text = (data.get("text") or "").strip()
        assert text, f"agent returned empty text\n{_container_logs(cid)}"
        assert CHAT_MARKER in text, (
            f"expected marker {CHAT_MARKER!r} in agent reply, got: {text[:500]!r}"
        )
        # The buffered chat payload is reduced to {text, events, is_error} —
        # turn counts are deliberately withheld from the chat client — so a
        # completed turn is evidenced by its terminal result event.
        events = data.get("events") or []
        assert any(e.get("type") == "result" for e in events), (
            f"expected a completed agent turn, got events: {events}"
        )
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
