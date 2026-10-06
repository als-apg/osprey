"""The scope that runs litellm's success handlers on a pool the test owns.

A completed synchronous litellm call hands its success handler to a pool. Inside
the scope that pool is the test's own, and when the scope closes the library's
pool is back in place and every worker of the test's pool has been joined.
"""

from __future__ import annotations

import threading

import litellm
import litellm.utils

from tests._litellm_callbacks import THREAD_NAME_PREFIX, litellm_callback_pool


def _scoped_workers() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith(THREAD_NAME_PREFIX)]


def test_a_completed_call_hands_its_success_handler_to_the_scoped_pool():
    library_pool = litellm.utils.executor

    with litellm_callback_pool() as pool:
        assert litellm.utils.executor is pool
        reply = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            mock_response="pong",
            api_key="unused",
        )
        assert reply.choices[0].message.content == "pong"
        assert _scoped_workers()

    assert litellm.utils.executor is library_pool
    assert _scoped_workers() == []
