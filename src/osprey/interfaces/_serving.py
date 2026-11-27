"""Supported helpers for serving a FastAPI interface on a throwaway port.

Every OSPREY server binds its listening socket through :func:`listening_socket`
before it reports a port, and serves on that socket: port 0 lets the system
choose, and the port a server names is the one it holds.

These utilities run any OSPREY interface ``create_app()`` on a localhost port
the system chooses, in a background thread — the shared surface used by the real-browser
Playwright suites under ``tests/interfaces/`` *and* by the documentation
screenshot runner (``docs/screenshots/``). Both consumers import a real,
supported API here instead of reaching into test internals.

The module depends only on the standard library plus ``uvicorn``/``fastapi``
(both already core runtime dependencies); it pulls in no test-only packages, so
importing it from shipped code drags nothing extra along.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

import uvicorn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI

logger = logging.getLogger(__name__)

#: Seconds uvicorn may spend draining open connections once it is told to
#: exit, before it cancels whatever is still running and moves on to the
#: application's lifespan shutdown. Uvicorn's default is *no* bound: it waits
#: for every connection and every in-flight handler to finish, and only then
#: runs the lifespan shutdown. A proxied WebSocket still closing upstream, or a
#: handler stuck behind a slow peer, therefore holds the lifespan shutdown open
#: for as long as it likes — and everything that shutdown owns (the notebook
#: sidecar and its kernels, the file watcher, PTY children) keeps running with
#: it. The callers here have already closed their pages by the time they stop
#: the server, so a connection that has not gone in this long is not going to.
_DRAIN_TIMEOUT = 5

#: Seconds to wait for the serving thread to finish after ``should_exit``.
#: Covers the drain above plus a lifespan shutdown that terminates
#: subprocesses with a grace period of its own (the notebook sidecar gives its
#: server five seconds on ``SIGTERM`` before ``SIGKILL``, then reaps kernels).
#: The earlier five-second join returned before such a shutdown could possibly
#: have finished, so a test that checked for leaked processes afterwards was
#: racing the shutdown it was meant to observe.
_SHUTDOWN_JOIN = 30.0

#: Seconds a server may stay alive and still starting before
#: :func:`run_app_server` gives up on it and reports where its lifespan startup
#: is waiting. A hang ceiling, not an expected startup time: real interface
#: apps are serving in about a tenth of a second, and a server that died is
#: reported at once through its thread, never after this wait. Read at call
#: time, so a test lowers it on the module.
_STARTUP_CEILING_S = 30.0

#: Innermost frames of the serving thread's stack a startup report keeps.
_STACK_FRAMES_KEPT = 8

#: Longest await chain a startup report follows from one task.
_AWAIT_CHAIN_LIMIT = 64


#: The address ``osprey web`` bound its listening socket to, published to its
#: children beside ``OSPREY_WEB_PORT``. The terminal reads it back to tell
#: whether its sessions, which dial ``127.0.0.1``, can reach it at all.
WEB_HOST_ENV = "OSPREY_WEB_HOST"


#: Pending-connection queue of a socket :func:`listening_socket` returns;
#: uvicorn's own default, so a server handed the socket listens as it would
#: have on a socket it bound itself.
_BACKLOG = 2048


def listening_socket(host: str, port: int) -> socket.socket:
    """Bind and listen on *host*:*port*, and return the socket.

    The one way an OSPREY server takes its port: the caller reads the number
    from ``sock.getsockname()[1]`` only after this returns, reports it, and
    hands the same socket to uvicorn (``server.run(sockets=[sock])``), so the
    port it names is the one it holds for its whole life. Port 0 lets the
    system choose. ``SO_REUSEADDR`` matches uvicorn's own bind, so a port held
    only by ``TIME_WAIT`` connections of a server that just stopped is free.

    Args:
        host: The address to bind; an IPv6 literal binds an IPv6 socket.
        port: The port to bind, or 0 for one the system chooses.

    Raises:
        OSError: If the address cannot be bound, a port in use among them.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(_BACKLOG)
    except BaseException:
        sock.close()
        raise
    return sock


def unix_listening_socket(path: str) -> socket.socket:
    """Bind and listen on the AF_UNIX socket *path*, and return the socket.

    :func:`listening_socket` for a local socket a parent holds for a child it
    starts. The caller owns *path*'s directory and its mode.

    Raises:
        OSError: If *path* cannot be bound, an existing file among them.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        sock.listen(_BACKLOG)
    except BaseException:
        sock.close()
        raise
    return sock


def wait_for_port(port: int, timeout: float = 10.0) -> None:
    """Block until the server accepts TCP connections, or raise RuntimeError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not become ready on port {port} within {timeout}s")


def _frame_of(awaitable: Any) -> Any:
    """The frame *awaitable* is suspended in, or ``None`` if it has none."""
    for attr in ("cr_frame", "ag_frame", "gi_frame"):
        frame = getattr(awaitable, attr, None)
        if frame is not None:
            return frame
    return None


def _awaited_by(awaitable: Any, frame: Any) -> Any:
    """What *awaitable*, suspended in *frame*, is itself waiting on."""
    for attr in ("cr_await", "ag_await", "gi_yieldfrom"):
        inner = getattr(awaitable, attr, None)
        if inner is not None:
            break
    # An async context manager's ``__aenter__`` awaits ``anext(gen)``, an
    # awaitable with no frame of its own; the wait continues in the generator.
    if (
        _frame_of(inner) is None
        and frame.f_code.co_name == "__aenter__"
        and frame.f_code.co_filename.endswith("contextlib.py")
    ):
        return getattr(frame.f_locals.get("self"), "gen", None)
    return inner


def _await_chain(task: asyncio.Task[Any]) -> str:
    """*task*'s await chain, outermost first, as ``name (file:line)`` steps."""
    steps: list[str] = []
    awaitable: Any = task.get_coro()
    while awaitable is not None and len(steps) < _AWAIT_CHAIN_LIMIT:
        frame = _frame_of(awaitable)
        if frame is None:
            steps.append(type(awaitable).__name__)
            break
        code = frame.f_code
        steps.append(f"{code.co_name} ({code.co_filename}:{frame.f_lineno})")
        awaitable = _awaited_by(awaitable, frame)
    return " -> ".join(steps)


def _startup_whereabouts(thread: threading.Thread, loop: list[asyncio.AbstractEventLoop]) -> str:
    """Where a server still starting is waiting, for its startup report.

    Two sections: the await chain of every pending task on the server's loop,
    which names a step the lifespan waits on by ``await``, and the serving
    thread's stack, which names a step that blocks the loop itself. Both are
    read from another thread while the server runs, so each is best effort and
    a section that cannot be read says so instead of raising.
    """
    lines = ["lifespan awaiting:"]
    try:
        tasks = list(asyncio.all_tasks(loop[0])) if loop else []
        if not tasks:
            lines.append("  no pending task")
        for task in tasks:
            try:
                lines.append(f"  {task.get_name()}: {_await_chain(task)}")
            except Exception as exc:
                lines.append(f"  {task.get_name()}: unreadable ({exc!r})")
    except Exception as exc:
        lines.append(f"  unreadable ({exc!r})")
    lines.append("serving thread at:")
    try:
        frame = sys._current_frames().get(thread.ident or -1)
        if frame is None:
            lines.append("  no running frame")
        else:
            stack = traceback.format_stack(frame)[-_STACK_FRAMES_KEPT:]
            lines.extend(entry.rstrip("\n") for entry in stack)
    except Exception as exc:
        lines.append(f"  unreadable ({exc!r})")
    return "\n".join(lines)


def _wait_until_serving(
    server: uvicorn.Server,
    thread: threading.Thread,
    thread_error: list[BaseException],
    port: int,
    app_name: str,
    loop: list[asyncio.AbstractEventLoop],
) -> None:
    """Block until *server* is serving, or explain why it never will be.

    Waits on uvicorn's own ``started`` flag rather than probing the port, and
    treats the serving thread exiting as a terminal answer. A server that dies
    during startup — a lifespan that raises, a failed bind — otherwise looks
    exactly like one that is merely slow, so the caller pays the whole timeout
    and is then told the server "did not become ready", which points at the
    timeout rather than at the error that actually occurred.

    Because a dead server is detected immediately through the thread check,
    :data:`_STARTUP_CEILING_S` only ever bounds a *slow but alive* startup —
    app lifespans do real work (service imports, degraded-mode probes) that
    stretches under parallel test load, so the ceiling is generous rather than
    tight. A server still starting when it expires is reported with where its
    lifespan startup is waiting, read from *loop* (the server's event loop,
    once it has one) and the serving *thread*.
    """
    ceiling = _STARTUP_CEILING_S
    deadline = time.monotonic() + ceiling
    while time.monotonic() < deadline:
        if server.started:
            return
        if not thread.is_alive():
            raise RuntimeError(
                f"Server exited during startup without listening on port {port}; "
                "the application's startup failed (see the server log above)"
            ) from (thread_error[0] if thread_error else None)
        time.sleep(0.02)
    whereabouts = _startup_whereabouts(thread, loop)
    raise RuntimeError(
        f"{app_name} on port {port} was still starting after {ceiling:.0f} s "
        f"(hang ceiling); its lifespan startup is\n{whereabouts}"
    )


@contextmanager
def run_app_server(app: FastAPI) -> Iterator[str]:
    """Run any FastAPI app on a free port in a background thread.

    Yields:
        The server's base URL, e.g. ``"http://127.0.0.1:54321"``.
    """
    # The server owns its socket for its whole life, so the yielded URL is
    # provably this app's: nothing else on the host can take the port between
    # the bind and the readiness check.
    sock = listening_socket("127.0.0.1", 0)
    port = int(sock.getsockname()[1])

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=_DRAIN_TIMEOUT,
    )
    server = uvicorn.Server(config)
    thread_error: list[BaseException] = []
    loop: list[asyncio.AbstractEventLoop] = []

    def _serve() -> None:
        try:
            # What ``Server.run`` does, with the loop kept for a startup report.
            with asyncio.Runner(loop_factory=config.get_loop_factory()) as runner:
                loop.append(runner.get_loop())
                runner.run(server.serve(sockets=[sock]))
        except BaseException as exc:  # surfaced to the caller as the cause
            # Includes the SystemExit uvicorn raises when startup fails.
            thread_error.append(exc)

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    def _shutdown() -> None:
        server.should_exit = True
        t.join(timeout=_SHUTDOWN_JOIN)
        if t.is_alive():
            # Said out loud rather than raised: this runs in the caller's
            # ``finally``, where an exception would replace the test's own.
            # What it names is a lifespan shutdown that has not finished, which
            # is what a process-leak check further down will report in detail.
            logger.warning(
                "Server on port %d still shutting down after %.0f s; "
                "its lifespan shutdown has not completed",
                port,
                _SHUTDOWN_JOIN,
            )
        with suppress(OSError):
            sock.close()

    try:
        app_name = getattr(app, "title", type(app).__name__)
        _wait_until_serving(server, t, thread_error, port, app_name, loop)
    except BaseException:
        _shutdown()
        raise

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        _shutdown()


def authorize_browser_context(context: Any, *, host: str = "127.0.0.1") -> str:
    """Give a Playwright browser context a valid operator session cookie.

    Every OSPREY interface app is gated by
    :class:`osprey.interfaces.common_middleware.WebAuthMiddleware`, so a browser
    that has not exchanged a login token is refused with ``401`` on its very
    first navigation and never renders. Both real-browser callers — the
    Playwright suites under ``tests/interfaces/`` and the docs screenshot runner
    (``docs/screenshots/capture.py``) — drive an app this same process serves via
    :func:`run_app_server`, so they can be authorized directly: this mints a
    fresh browser session in *this process's* credential holder
    (:func:`osprey.interfaces.web_auth.get_web_credentials`) and installs it as
    the session cookie the gate reads, without going through the ``?token=``
    login exchange.

    The cookie is **host-scoped, not port-scoped**. Cookies ignore ports, and
    :func:`run_app_server` binds a fresh random port on every call, so a cookie
    pinned to one ``http://127.0.0.1:<port>`` origin would fail against the next
    server. Setting it for the bare host ``127.0.0.1`` (no port) means every
    ``run_app_server`` port on that host receives it. The cookie *name* comes
    from :func:`osprey.interfaces.common_middleware.session_cookie_name`, derived
    exactly as the gate derives the name it looks for, so the two always agree —
    including the port-suffixed spelling when ``OSPREY_WEB_PORT`` is set.

    This authorizes only servers running in *this* process, whose in-memory
    credential holder the session is minted in. A server in a *separate* process
    — a detached ``osprey web`` — issues its own sessions from its own holder and
    will not recognise this cookie; a browser reaching such a server authorizes
    through the ``?token=`` login URL instead. The session is minted ephemeral
    (unpersisted) for the same reason: it is same-process by construction and
    must never reach a persisted store, since the docs screenshot runner and the
    Playwright suites drive this path.

    Args:
        context: A Playwright ``BrowserContext`` (the sync API). Only its
            ``add_cookies`` method is used, so any object exposing that method
            works — including a ``Page.context``.
        host: The loopback host the servers bind, used verbatim as the cookie's
            domain. Defaults to ``127.0.0.1`` to match :func:`run_app_server`.

    Returns:
        The session id that was installed, for a caller that wants to assert on
        it or revoke it.
    """
    # Imported lazily so this module keeps depending only on the standard
    # library plus uvicorn/fastapi at import time; the auth modules are always
    # importable, but the deferral keeps the serving helpers' import surface
    # narrow and matches how the rest of this module stays test-package-free.
    from osprey.interfaces.common_middleware import session_cookie_name
    from osprey.interfaces.web_auth import get_web_credentials

    session_id = get_web_credentials().create_session(persist=False)
    context.add_cookies(
        [
            {
                "name": session_cookie_name(),
                "value": session_id,
                "domain": host,
                "path": "/",
            }
        ]
    )
    return session_id
