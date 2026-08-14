"""Tests for ``configure_logging()`` — the explicit entry-point logging setup.

The contract under test: idempotent, strictly additive (never removes handlers
it did not install), routes records to stderr, and quiets chatty third-party
loggers.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest
from rich.console import Console
from rich.logging import RichHandler

import osprey
import osprey.utils.logger
from osprey.utils.logger import QUIET_THIRD_PARTY_LOGGERS, configure_logging, get_logger


@pytest.fixture(autouse=True)
def restore_root_logging():
    """Isolate every test from the process-wide root logger.

    ``configure_logging()`` mutates global state that no test owns: the root
    logger's level and handler list, plus the levels of the third-party loggers
    it quiets. Left unrestored, an installed ``RichHandler`` would swallow the
    next test's ``caplog`` expectations and the raised third-party levels would
    silence log assertions in unrelated files.

    Whatever ran earlier in the process may itself have left a ``RichHandler``
    on the root logger, so the pre-``yield`` reset strips those too — otherwise
    the additive skip-if-present branch would hide the installation under test.
    """
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_lib_levels = {name: logging.getLogger(name).level for name in QUIET_THIRD_PARTY_LOGGERS}

    def restore(*, strip_rich: bool = False) -> None:
        root.setLevel(saved_level)
        root.handlers[:] = [
            h for h in saved_handlers if not (strip_rich and isinstance(h, RichHandler))
        ]
        for name, level in saved_lib_levels.items():
            logging.getLogger(name).setLevel(level)

    restore(strip_rich=True)
    yield
    restore()


def _rich_handlers() -> list[RichHandler]:
    return [h for h in logging.getLogger().handlers if isinstance(h, RichHandler)]


class TestHandlerInstallation:
    def test_installs_a_stderr_rich_handler(self):
        configure_logging()

        handlers = _rich_handlers()
        assert len(handlers) == 1
        assert handlers[0].console.stderr is True

    def test_is_idempotent(self):
        configure_logging()
        configure_logging()
        configure_logging()

        assert len(_rich_handlers()) == 1

    def test_does_not_replace_a_pre_existing_rich_handler(self):
        preexisting = RichHandler(console=Console(stderr=True))
        logging.getLogger().addHandler(preexisting)

        configure_logging()

        assert _rich_handlers() == [preexisting]

    def test_sets_the_root_level(self):
        configure_logging(logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG

        configure_logging(logging.WARNING)
        assert logging.getLogger().level == logging.WARNING

    def test_survives_an_unavailable_config_system(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("config unavailable")

        monkeypatch.setattr("osprey.utils.logger.get_config_value", boom)

        configure_logging()

        assert len(_rich_handlers()) == 1


class TestAdditiveSemantics:
    def test_foreign_handlers_survive(self):
        root = logging.getLogger()
        foreign = logging.NullHandler()
        root.addHandler(foreign)

        configure_logging()

        assert foreign in root.handlers

    def test_caplog_still_captures_after_configuration(self, caplog):
        """The regression this redesign exists to prevent."""
        with caplog.at_level(logging.INFO):
            configure_logging()
            logging.getLogger("osprey.test.caplog_survives").info("still captured")

        assert "still captured" in caplog.text


class TestGetLoggerPurity:
    """``get_logger()`` is a lookup, not a configuration call."""

    def test_installs_no_handler(self):
        root = logging.getLogger()
        before = list(root.handlers)

        get_logger("purity_check")

        assert root.handlers == before
        assert _rich_handlers() == []

    def test_does_not_change_the_root_level(self):
        root = logging.getLogger()
        root.setLevel(logging.CRITICAL)

        get_logger("purity_check")
        get_logger(name="purity_check_explicit", color="blue")

        assert root.level == logging.CRITICAL

    def test_leaves_caplog_capturing(self, caplog):
        with caplog.at_level(logging.INFO):
            logger = get_logger("purity_check")
            logger.info("captured without configuration")

        assert "captured without configuration" in caplog.text


class TestStreamRouting:
    def test_records_go_to_stderr_not_stdout(self, capsys):
        configure_logging()

        logging.getLogger("osprey.test.routing").info("stderr-routed-marker")

        captured = capsys.readouterr()
        assert "stderr-routed-marker" in captured.err
        assert captured.out == ""


class TestThirdPartyQuieting:
    def test_quiets_the_documented_loggers(self):
        for name in QUIET_THIRD_PARTY_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)

        configure_logging()

        for name in QUIET_THIRD_PARTY_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING

    def test_get_logger_does_not_quiet_them(self):
        for name in QUIET_THIRD_PARTY_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)

        get_logger("purity_check")

        for name in QUIET_THIRD_PARTY_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG

    def test_quiet_list_matches_the_documented_set(self):
        assert set(QUIET_THIRD_PARTY_LOGGERS) == {
            "httpx",
            "httpcore",
            "requests",
            "urllib3",
            "LiteLLM",
            "claude_agent_sdk",
        }


class TestPackageRootExport:
    """``osprey.configure_logging`` is exported, and resolving it stays lazy."""

    def test_export_is_the_same_function(self):
        assert osprey.configure_logging is osprey.utils.logger.configure_logging

    def test_dir_advertises_the_export(self):
        # PEP 562 lazy attributes are invisible to the default dir(); the
        # module's __dir__ is what makes tab-completion find this one.
        assert "configure_logging" in dir(osprey)

    def test_bare_import_pulls_in_neither_rich_nor_the_logger_module(self):
        """``import osprey`` must stay cheap.

        The package resolves this export through ``__getattr__`` precisely so
        that importing the framework does not drag in rich and the config
        system. An eager ``from osprey.utils.logger import configure_logging``
        in ``__init__.py`` would satisfy every other test in this file while
        silently undoing that.
        """
        probe = (
            "import sys; import osprey; "
            "print('rich' in sys.modules, 'osprey.utils.logger' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )

        assert result.stdout.strip() == "False False", (
            f"bare `import osprey` should import neither rich nor "
            f"osprey.utils.logger; got {result.stdout.strip()!r}"
        )


class TestAcceptedStdoutLimitation:
    """A foreign stdout ``RichHandler`` defeats stderr routing — accepted.

    ``configure_logging()`` installs nothing when the root logger already has a
    ``RichHandler``, whatever stream that handler writes to. So a host process
    that installed its own stdout ``RichHandler`` before calling us keeps it,
    and Osprey's records follow it onto stdout.

    This is the deliberate trade, not an oversight: the same
    add-only-if-absent rule is what stops Osprey from evicting handlers it does
    not own — ``caplog``'s among them — which is the leak this guard exists to
    prevent. Nothing in the codebase installs a stdout ``RichHandler``
    (there is no ``redirect_logging_to_stderr()``), so the case is
    hypothetical here. It is pinned rather than described so that changing the
    guard's semantics in EITHER direction fails loudly instead of quietly
    altering where log bytes land.
    """

    def test_a_foreign_stdout_handler_is_kept_and_stdout_is_not_protected(self, capsys):
        foreign = RichHandler(console=Console(stderr=False, force_terminal=True, width=120))
        logging.getLogger().addHandler(foreign)

        configure_logging()

        assert _rich_handlers() == [foreign], "configure_logging must not add a second handler"

        logging.getLogger("osprey.test.accepted_limitation").warning("lands-on-stdout")

        captured = capsys.readouterr()
        assert "lands-on-stdout" in captured.out
        assert "lands-on-stdout" not in captured.err
