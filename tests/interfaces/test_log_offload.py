"""Tests for rendering log records off the thread that logs them."""

from __future__ import annotations

import logging
import threading

import pytest
from rich.logging import RichHandler

from osprey.interfaces._log_offload import rendering_logs_off_thread

LOGGER_NAME = "osprey.tests.log_offload"


class _Blocking(RichHandler):
    """A renderer that holds every record until the test lets it go."""

    def __init__(self) -> None:
        super().__init__()
        self.unblock = threading.Event()
        self.rendered: list[str] = []
        self.threads: set[int] = set()

    def emit(self, record: logging.LogRecord) -> None:
        self.unblock.wait(timeout=10)
        self.rendered.append(record.getMessage())
        self.threads.add(threading.get_ident())


@pytest.fixture
def renderer():
    """A blocking renderer on the root logger, as the Osprey handler sits there."""
    root = logging.getLogger()
    saved = list(root.handlers)
    handler = _Blocking()
    root.handlers[:] = [h for h in saved if not isinstance(h, RichHandler)] + [handler]
    yield handler
    handler.unblock.set()
    root.handlers[:] = saved


def test_a_slow_renderer_does_not_block_the_thread_that_logs(renderer):
    """What wedged the web terminal: a record whose rendering takes seconds held
    the event loop. Behind the queue the logging call returns at once."""
    logger = logging.getLogger(LOGGER_NAME)

    with rendering_logs_off_thread():
        caller = threading.Thread(target=logger.error, args=("slow to render",))
        caller.start()
        caller.join(timeout=2)
        returned = not caller.is_alive()
        renderer.unblock.set()

    assert returned
    assert renderer.rendered == ["slow to render"]
    assert threading.get_ident() not in renderer.threads


def test_records_are_rendered_in_order_and_none_is_lost_at_shutdown(renderer):
    logger = logging.getLogger(LOGGER_NAME)
    messages = [f"record {n}" for n in range(200)]

    with rendering_logs_off_thread():
        for message in messages:
            logger.error(message)
        assert renderer.rendered == []
        # Released only once the block is closing: leaving it waits for the queue.
        threading.Timer(0.05, renderer.unblock.set).start()

    assert renderer.rendered == messages


def test_the_handler_is_back_on_the_root_logger_afterwards(renderer):
    root = logging.getLogger()
    before = list(root.handlers)

    with rendering_logs_off_thread():
        assert renderer not in root.handlers
    renderer.unblock.set()

    assert root.handlers == before
    logging.getLogger(LOGGER_NAME).error("direct again")
    assert renderer.rendered == ["direct again"]
    assert renderer.threads == {threading.get_ident()}


def test_a_failing_block_still_flushes_and_restores(renderer):
    root = logging.getLogger()
    before = list(root.handlers)
    renderer.unblock.set()

    with pytest.raises(RuntimeError), rendering_logs_off_thread():
        logging.getLogger(LOGGER_NAME).error("last words")
        raise RuntimeError("server died")

    assert renderer.rendered == ["last words"]
    assert root.handlers == before


def test_a_traceback_reaches_the_renderer_intact(renderer):
    """The record is handed over as it is, so the renderer still decides how an
    exception is drawn."""
    seen: list[logging.LogRecord] = []
    renderer.emit = seen.append  # type: ignore[method-assign]

    with rendering_logs_off_thread():
        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger(LOGGER_NAME).error("failed: %s", "x", exc_info=True)

    (record,) = seen
    assert record.getMessage() == "failed: x"
    assert record.exc_info is not None and record.exc_info[0] is ValueError


def test_the_renderers_own_level_and_filters_still_apply(renderer):
    renderer.unblock.set()
    renderer.setLevel(logging.ERROR)
    renderer.addFilter(lambda record: "secret" not in record.getMessage())
    logger = logging.getLogger(LOGGER_NAME)

    with rendering_logs_off_thread():
        logger.warning("below the level")
        logger.error("a secret")
        logger.error("shown")

    assert renderer.rendered == ["shown"]


def test_capture_handlers_stay_synchronous(renderer, caplog):
    """Only the renderer moves: a test's capture sees a record the moment it is logged."""
    with rendering_logs_off_thread(), caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        logging.getLogger(LOGGER_NAME).info("captured at once")
        assert "captured at once" in caplog.messages
        renderer.unblock.set()


def test_without_a_renderer_nothing_changes():
    root = logging.getLogger()
    saved = list(root.handlers)
    root.handlers[:] = [h for h in saved if not isinstance(h, RichHandler)]
    try:
        before = list(root.handlers)
        threads = threading.active_count()
        with rendering_logs_off_thread():
            assert root.handlers == before
            assert threading.active_count() == threads
    finally:
        root.handlers[:] = saved


def test_the_web_terminal_serves_behind_the_queue(renderer, monkeypatch):
    """``run_web`` is where the web terminal's server runs; the renderer is off
    the root logger for exactly as long as it serves."""
    import uvicorn

    from osprey.interfaces.web_terminal import app as web_app

    root = logging.getLogger()
    while_serving: list[bool] = []

    class _Server:
        started = True

        def __init__(self, config):
            pass

        def run(self, **_kw):
            while_serving.append(renderer in root.handlers)

    monkeypatch.setattr(uvicorn, "Server", _Server)
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **kw: None)
    monkeypatch.setattr(web_app, "create_app", lambda **kw: object())
    monkeypatch.setattr(web_app, "_open_browser_when_ready", lambda url: None)
    renderer.unblock.set()

    class _Sock:
        def close(self):
            pass

    web_app.run_web(sock=_Sock())

    assert while_serving == [False]
    assert renderer in root.handlers
