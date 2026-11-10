#!/usr/bin/env python3
"""One small call to a cell's model and to its judge, before the cell does anything else.

A refused call stops the cell: it writes the cell's summary with a ``setup_error``
naming the provider, the model and the reason, and exits 1, so no further tokens are
spent and the board shows why the cell did not run. ``matrix.py`` runs such a cell
again on resume. What counts as a refusal is the rule the mid-session path applies
(``tests/e2e/refusals.py``), with no model turn yet: a rate limit or a 5xx is retried
first and is a refusal once it outlasts the retries, and an ordinary 400 is not one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

RETRY_DELAYS_S = (5, 15)
TIMEOUT_S = 120
ASK = [{"role": "user", "content": "Reply with the word ok."}]


def _post(url: str, headers: dict[str, str], body: dict) -> tuple[int | None, str]:
    """(status, body text); a status of None means no response came back."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.status, ""
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")
    except (urllib.error.URLError, OSError) as error:
        return None, str(error)


def _request(protocol: str, base_url: str, key: str, model: str) -> tuple[str, dict, dict]:
    """The smallest call the protocol allows: (url, headers, body)."""
    base = base_url.rstrip("/")
    body = {"model": model, "max_tokens": 16, "messages": ASK}
    if protocol == "openai":
        return f"{base}/chat/completions", {"authorization": f"Bearer {key}"} if key else {}, body
    # A gateway's base URL may or may not carry the version the Anthropic SDK appends.
    headers = {
        "x-api-key": key,
        "authorization": f"Bearer {key}",
        "anthropic-version": "2023-06-01",
    }
    return f"{base.removesuffix('/v1')}/v1/messages", headers, body


def _refusals():
    """The one refusal rule and scrubber the mid-session path uses too (tests/e2e/refusals.py)."""
    path = Path(__file__).resolve().parents[2] / "tests" / "e2e" / "refusals.py"
    spec = importlib.util.spec_from_file_location("benchmark_refusals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _readable(text: str, key: str) -> str:
    """The provider's own words, short, with nothing a published board should not carry."""
    try:
        error = json.loads(text).get("error", {})
        text = str(error.get("message", text)) if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        pass
    return _refusals().scrub(text, key=key)[:300].strip()


def check(
    protocol: str,
    base_url: str,
    key: str,
    model: str,
    post: Callable[[str, dict, dict], tuple[int | None, str]] = _post,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Why the endpoint refused *model*, or None when it answered or failed in a way
    that is not a refusal (an ordinary 400 stays the cell's own outcome)."""
    url, headers, body = _request(protocol, base_url, key, model)
    status, text = None, ""
    for delay in (*RETRY_DELAYS_S, None):
        status, text = post(url, headers, body)
        if status is not None and status < 300:
            return None
        if status is not None and status != 429 and status < 500:
            break
        if delay is not None:
            sleep(delay)
    text = _readable(text, key)
    if status is None:
        # No response at all: name it the way the agent CLI does, so the rule reads it.
        text = f"connection error: {text}"
    return _refusals().refusal_reason(status=status, text=text, model_turns=False)


def _parse_cell(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--model", "--seed", "--provider", "--route", "--judge", "--out"):
        ap.add_argument(name, default="")
    ap.add_argument(
        "--setup-error",
        default="",
        help="record this setup error for the cell instead of calling anything (a "
        "prerequisite the worker found missing), and exit 1",
    )
    return ap.parse_args(argv)


def _write_setup_error(args: argparse.Namespace, message: str) -> None:
    Path(args.out).write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": int(args.seed),
                "provider": args.provider,
                "route": args.route,
                "judge": args.judge,
                "pytest_rc": None,
                "total_duration_s": 0,
                "passed": 0,
                "failed": 0,
                "timeout": 0,
                "skipped": 0,
                "errors": 0,
                "total": 0,
                "tests": [],
                "setup_error": message,
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_cell(argv)
    if args.setup_error:
        print(f"PREFLIGHT: {args.setup_error}", file=sys.stderr)
        _write_setup_error(args, args.setup_error)
        return 1
    env = os.environ
    protocol = "openai" if args.route == "proxy" else "anthropic"
    calls = [
        (
            args.provider,
            args.model,
            protocol,
            env.get("OSPREY_BENCH_SUT_BASE_URL", ""),
            env.get("OSPREY_BENCH_SUT_KEY", ""),
        )
    ]
    judge_url = env.get("ALS_APG_BASE_URL", "")
    if args.judge and judge_url:
        calls.append(("judge", args.judge, "anthropic", judge_url, env.get("ALS_APG_API_KEY", "")))
    for who, model, proto, base_url, key in calls:
        reason = check(proto, base_url, key, model)
        if reason is None:
            continue
        message = f"{who} refused {model}: {reason}"
        print(f"PREFLIGHT: {message}", file=sys.stderr)
        _write_setup_error(args, message)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
