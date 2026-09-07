"""The shell stream re-arm: a request that lands while a reply is going out is still served.

``ipykernel`` 7 answers shell requests with a direct send on the ROUTER socket
that a ``ZMQStream`` is reading, from the shell channel thread. That send makes
libzmq process the socket's commands, consuming the wake-up the stream's loop
is waiting on; a request that arrived in the window is then stuck until some
later command happens to wake the stream. The re-arm in
:func:`osprey.jupyter_kernel.install_shell_stream_rearm` re-reads the stream's
events after each such send.

The plumbing under test is ``ipykernel``'s own ``SubshellManager`` — the inproc
pair the main thread's replies travel through, and its ``_send_on_shell_channel``
callback — driven without a kernel. The interleaving is forced: peer A's reply
is held until peer B's request has reached the ROUTER inside the same loop
iteration, and peer A has closed before its reply goes out, the way
``jupyter_server``'s connection nudge closes its transient channel before the
kernel answers. Without the re-arm B's request is never read (it is freed only
by a third peer's connection); with it, B is answered.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

zmq = pytest.importorskip("zmq")
subshell_manager = pytest.importorskip("ipykernel.subshell_manager")

from tornado.ioloop import IOLoop  # noqa: E402
from zmq.eventloop.zmqstream import ZMQStream  # noqa: E402

from osprey import jupyter_kernel  # noqa: E402

#: Longer than libzmq's send-side command throttle on any host (about 1 ms on an
#: x86 TSC, longer on counters that tick slower), so the direct send that follows
#: is one that processes the socket's pending commands.
SETTLE = 0.3

#: How long a peer waits for its reply before the test calls it stuck.
REPLY_DEADLINE_MS = 3000

pytestmark = pytest.mark.timeout(60)


class _ShellChannel:
    """A shell channel thread as ``ipykernel`` builds it, minus the kernel.

    One IOLoop on its own thread; a ``ZMQStream`` reading the ROUTER and
    forwarding every request to the "main thread" over the manager's inproc
    pair; a main-thread stand-in that echoes each request back as its reply;
    and the manager's own ``_send_on_shell_channel`` writing the reply to the
    ROUTER directly.
    """

    def __init__(self, hold_reply: threading.Event | None) -> None:
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.linger = 0
        self.port = self.router.bind_to_random_port("tcp://127.0.0.1")
        self.loop = IOLoop(make_current=False)
        self.hold_reply = hold_reply
        self.manager = subshell_manager.SubshellManager(self.context, self.loop, self.router)
        self.shell_stream = ZMQStream(self.router, self.loop)
        self.shell_stream.on_recv(self._forward_to_main, copy=False)
        self.to_main = self.manager.get_shell_channel_to_subshell_socket(None)
        self.main_in = self.manager.get_shell_channel_to_subshell_pair(None).to_socket
        self.reply_out = self.manager.get_subshell_to_shell_channel_socket(None)
        self.loop_thread = threading.Thread(
            target=self.loop.start, daemon=True, name=subshell_manager.SHELL_CHANNEL_THREAD_NAME
        )
        self.main_thread = threading.Thread(target=self._main, daemon=True, name="main")
        self.loop_thread.start()
        self.main_thread.start()

    def _forward_to_main(self, frames: list[zmq.Frame]) -> None:
        self.to_main.send_multipart(frames, copy=False)

    def _main(self) -> None:
        while True:
            frames = self.main_in.recv_multipart()
            if frames[-1] == b"stop":
                return
            self.reply_out.send_multipart(frames)

    def peer(self) -> zmq.Socket:
        socket = self.context.socket(zmq.DEALER)
        socket.linger = 0
        socket.connect(f"tcp://127.0.0.1:{self.port}")
        return socket

    def close(self) -> None:
        self.to_main.send_multipart([b"", b"stop"])
        self.main_thread.join(timeout=2)

        def shut_down() -> None:  # the manager insists on its own thread
            self.manager.close()
            self.shell_stream.close()
            self.loop.stop()

        self.loop.add_callback(shut_down)
        self.loop_thread.join(timeout=2)
        self.context.destroy(linger=0)


@pytest.fixture
def gated_send(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, threading.Event]]:
    """``_send_on_shell_channel`` that waits for peer B's request before sending A's reply.

    Installed under the re-arm so the re-arm wraps it, the way it wraps the
    real send. The gate opens once per test: replies after the first pass
    straight through.
    """
    gate = {"a_reply_pending": threading.Event(), "b_sent": threading.Event()}
    original = subshell_manager.SubshellManager._send_on_shell_channel
    gated_once = threading.Event()

    def gated(self, msg):  # type: ignore[no-untyped-def]
        if not gated_once.is_set():
            gated_once.set()
            gate["a_reply_pending"].set()
            gate["b_sent"].wait(timeout=5)
            time.sleep(SETTLE)
        original(self, msg)

    monkeypatch.setattr(subshell_manager.SubshellManager, "_send_on_shell_channel", gated)
    monkeypatch.setattr(
        subshell_manager.SubshellManager, jupyter_kernel._SHELL_REARM_MARK, False, raising=False
    )
    yield gate


def _run_nudge_shaped_exchange(channel: _ShellChannel, gate: dict[str, threading.Event]) -> bool:
    """Peer A asks and leaves; peer B asks while A's reply is held. Was B answered?"""
    a, b = channel.peer(), channel.peer()
    time.sleep(0.05)
    a.send_multipart([b"", b"from-a"])
    assert gate["a_reply_pending"].wait(timeout=5), "A's request never reached the reply path"
    a.close()  # the transient channel is gone before its reply, as under the nudge
    b.send_multipart([b"", b"from-b"])
    time.sleep(0.05)  # let B's request reach the ROUTER inside the held callback
    gate["b_sent"].set()
    answered = bool(b.poll(REPLY_DEADLINE_MS))
    b.close()
    return answered


def test_the_rearmed_stream_serves_a_request_that_landed_during_a_reply(gated_send):
    channel = _ShellChannel(None)
    try:
        assert not getattr(channel.manager, jupyter_kernel._SHELL_REARM_MARK, False)
    finally:
        channel.close()

    jupyter_kernel.install_shell_stream_rearm(lambda: channel.shell_stream)
    channel = _ShellChannel(None)
    try:
        assert _run_nudge_shaped_exchange(channel, gated_send)
    finally:
        channel.close()


def test_without_the_rearm_that_request_is_stuck_until_another_peer_connects(gated_send):
    """The failure the re-arm exists for, pinned so the test above is known to bite.

    Stuck means: B is not answered within the deadline, and a third peer's
    connection then frees it. If this test ever fails because B is answered
    promptly, ``ipykernel`` has fixed its send and the re-arm can go.
    """
    channel = _ShellChannel(None)
    try:
        answered = _run_nudge_shaped_exchange(channel, gated_send)
        assert not answered
        c = channel.peer()
        c.send_multipart([b"", b"from-c"])
        assert c.poll(REPLY_DEADLINE_MS)
        c.close()
    finally:
        channel.close()


def test_the_rearm_installs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subshell_manager.SubshellManager, jupyter_kernel._SHELL_REARM_MARK, False, raising=False
    )
    original = subshell_manager.SubshellManager._send_on_shell_channel
    monkeypatch.setattr(subshell_manager.SubshellManager, "_send_on_shell_channel", original)

    jupyter_kernel.install_shell_stream_rearm(lambda: None)
    first = subshell_manager.SubshellManager._send_on_shell_channel
    jupyter_kernel.install_shell_stream_rearm(lambda: None)

    assert subshell_manager.SubshellManager._send_on_shell_channel is first
    assert first is not original
