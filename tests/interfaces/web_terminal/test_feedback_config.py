"""Startup resolution of the docs/feedback config keys.

Covers the ``_create_lifespan`` block that reads ``web.docs_url`` and the
``web.feedback.*`` keys onto ``app.state``, the derivation of the feedback
store location (deliberately independent of ``web_terminal.watch_dir``), and
the three UI-facing keys echoed by ``GET /api/panels``.

Two client shapes are used, matching the conventions of this directory:
a *full lifespan* client (``_lifespan_client``) for the config plumbing, and a
*router-only* app for the ``/api/panels`` getattr defaults.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    DEFAULT_DOCS_URL,
    DEFAULT_FEEDBACK_EMAIL,
    DEFAULT_FEEDBACK_GITHUB_REPO,
    DEFAULT_FEEDBACK_MAX_STORE_BYTES,
    coerce_config_str,
    coerce_store_ceiling,
    create_app,
)
from osprey.interfaces.web_terminal.routes.panels import router as panels_router


def _config_reader(overrides: dict | None = None):
    """A ``get_config_value`` stand-in that maps dotted keys to *overrides*.

    Every unmapped key falls back to the default the caller passed, so the
    other lifespan blocks reading through the same function (``web.theme``,
    ``web.chat_*``) keep resolving to their own defaults.
    """
    mapping = overrides or {}

    def _get(path: str, default=None, config_path=None):
        return mapping.get(path, default)

    return _get


@contextmanager
def _lifespan_client(
    project_dir: Path,
    shared_root: Path,
    *,
    overrides: dict | None = None,
    watch_dir: Path | None = None,
    config_reader=None,
):
    """Run a full ``create_app`` lifespan with config injected at three seams.

    ``shared_root`` becomes ``agent_data.base_dir`` (absolute, so it survives
    anchoring untouched), which is what ``resolve_shared_data_root`` derives the
    feedback store from. ``watch_dir`` — when given — is the ``web_terminal``
    section's key, which moves ``app.state.workspace_dir`` and nothing else.
    """
    web_terminal_section: dict = {}
    if watch_dir is not None:
        web_terminal_section["watch_dir"] = str(watch_dir)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value=web_terminal_section,
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {}, "agent_data": {"base_dir": str(shared_root)}},
        ),
        patch(
            "osprey.utils.config.get_config_value",
            side_effect=config_reader or _config_reader(overrides),
        ),
    ):
        app = create_app(shell_command="echo", project_dir=project_dir)
        with TestClient(app) as client:
            yield client, app


@pytest.fixture
def shared_root(tmp_path):
    root = tmp_path / "shared_data"
    root.mkdir()
    return root


@pytest.fixture
def project_dir(tmp_path):
    proj = tmp_path / "project"
    proj.mkdir()
    return proj


class TestFeedbackConfigDefaults:
    def test_absent_keys_resolve_to_the_shipped_defaults(self, project_dir, shared_root):
        """A config naming none of the keys still leaves every attribute set."""
        with _lifespan_client(project_dir, shared_root) as (_client, app):
            assert app.state.docs_url == DEFAULT_DOCS_URL
            assert app.state.feedback_github_repo == DEFAULT_FEEDBACK_GITHUB_REPO
            assert app.state.feedback_email == DEFAULT_FEEDBACK_EMAIL
            assert app.state.feedback_max_store_bytes == DEFAULT_FEEDBACK_MAX_STORE_BYTES

    def test_default_ceiling_is_256_mb(self):
        """The documented default, spelled out so a silent change is caught."""
        assert DEFAULT_FEEDBACK_MAX_STORE_BYTES == 268435456

    def test_configured_values_win(self, project_dir, shared_root):
        """Each key is read from its own dotted path."""
        overrides = {
            "web.docs_url": "https://docs.example.org/osprey",
            "web.feedback.github_repo": "facility/osprey-fork",
            "web.feedback.email": "controls@example.org",
            "web.feedback.max_store_bytes": 1048576,
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (_client, app):
            assert app.state.docs_url == "https://docs.example.org/osprey"
            assert app.state.feedback_github_repo == "facility/osprey-fork"
            assert app.state.feedback_email == "controls@example.org"
            assert app.state.feedback_max_store_bytes == 1048576

    def test_ceiling_is_coerced_to_int(self, project_dir, shared_root):
        """A YAML-quoted ceiling still reaches the store as an int."""
        with _lifespan_client(
            project_dir, shared_root, overrides={"web.feedback.max_store_bytes": "4096"}
        ) as (_client, app):
            assert app.state.feedback_max_store_bytes == 4096
            assert isinstance(app.state.feedback_max_store_bytes, int)

    def test_config_read_failure_falls_open_to_defaults(self, project_dir, shared_root):
        """A broken config must never keep the server from starting."""

        def _explode(path: str, default=None, config_path=None):
            raise RuntimeError("config.yml is unreadable")

        with _lifespan_client(project_dir, shared_root, config_reader=_explode) as (_client, app):
            assert app.state.docs_url == DEFAULT_DOCS_URL
            assert app.state.feedback_github_repo == DEFAULT_FEEDBACK_GITHUB_REPO
            assert app.state.feedback_email == DEFAULT_FEEDBACK_EMAIL
            assert app.state.feedback_max_store_bytes == DEFAULT_FEEDBACK_MAX_STORE_BYTES


class TestBadValuesAreContained:
    """One unusable key must never take the other three down with it."""

    def test_garbage_ceiling_leaves_the_string_keys_alone(self, project_dir, shared_root):
        """The failure that would redirect a facility's feedback to upstream."""
        overrides = {
            "web.docs_url": "https://docs.example.org/osprey",
            "web.feedback.github_repo": "facility/osprey-fork",
            "web.feedback.email": "controls@example.org",
            "web.feedback.max_store_bytes": "256MB",
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (_client, app):
            assert app.state.docs_url == "https://docs.example.org/osprey"
            assert app.state.feedback_github_repo == "facility/osprey-fork"
            assert app.state.feedback_email == "controls@example.org"
            assert app.state.feedback_max_store_bytes == DEFAULT_FEEDBACK_MAX_STORE_BYTES

    def test_garbage_url_leaves_the_ceiling_alone(self, project_dir, shared_root):
        """The same containment in the other direction."""
        overrides = {
            "web.docs_url": {"nested": "by a mis-indented config.yml"},
            "web.feedback.max_store_bytes": 4096,
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (_client, app):
            assert app.state.docs_url == DEFAULT_DOCS_URL
            assert app.state.feedback_max_store_bytes == 4096

    @pytest.mark.parametrize(
        "bad", [0, -1, True, "256MB", "", [], None, 1.5e-3, float("inf"), float("-inf")]
    )
    def test_unusable_ceilings_fall_back_to_the_default(self, bad):
        """A non-positive ceiling would make the pruner empty the whole store.

        ``True`` is called out because ``int(True)`` is ``1``: a ceiling of one
        byte deletes every stored context on the next submission while looking
        like ordinary pruning. The infinities are the YAML spellings ``.inf``
        and ``1.0e+400``, on which ``int()`` raises ``OverflowError`` — this
        helper is called outside the lifespan's try, so an escape would abort
        startup rather than fail open.
        """
        assert coerce_store_ceiling(bad) == DEFAULT_FEEDBACK_MAX_STORE_BYTES

    @pytest.mark.parametrize(
        ("value", "expected"), [(4096, 4096), ("4096", 4096), (4096.9, 4096), (1, 1)]
    )
    def test_usable_ceilings_survive(self, value, expected):
        """Any positive byte count, however spelled in YAML, is honored."""
        assert coerce_store_ceiling(value) == expected

    @pytest.mark.parametrize("bad", [None, {"a": 1}, [], 42, False])
    def test_unusable_values_fall_back_to_the_default(self, bad):
        """A non-string would otherwise be repr'd into an href or a mailto:.

        ``None`` is in here rather than among the blanks on purpose: a key
        written with no value reads as "not decided", so it takes the default.
        """
        assert coerce_config_str("web.docs_url", bad, DEFAULT_DOCS_URL) == DEFAULT_DOCS_URL

    def test_configured_strings_are_stripped(self):
        """Trailing YAML whitespace never reaches a URL."""
        assert coerce_config_str("web.docs_url", "  https://x/  ", DEFAULT_DOCS_URL) == "https://x/"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_an_explicitly_blank_value_is_kept_blank(self, blank):
        """Blank means "no such target here", which the default cannot express."""
        assert coerce_config_str("web.docs_url", blank, DEFAULT_DOCS_URL) == ""

    def test_blank_config_reaches_app_state_for_all_three_keys(self, project_dir, shared_root):
        """The posture has to survive the whole lifespan, not just the helper.

        This is the air-gapped deployment: no documentation site to link to and
        no outbound channel to offer. Folding these back to the defaults would
        aim a control room's Feedback dialog at the upstream maintainers'
        tracker and put a dead docs link in its rail.
        """
        overrides = {
            "web.docs_url": "",
            "web.feedback.github_repo": "",
            "web.feedback.email": "",
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (_client, app):
            assert app.state.docs_url == ""
            assert app.state.feedback_github_repo == ""
            assert app.state.feedback_email == ""

    def test_absent_config_still_takes_the_defaults(self, project_dir, shared_root):
        """The counterpart: nothing configured is not the same as blanked."""
        with _lifespan_client(project_dir, shared_root) as (_client, app):
            assert app.state.docs_url == DEFAULT_DOCS_URL
            assert app.state.feedback_github_repo == DEFAULT_FEEDBACK_GITHUB_REPO
            assert app.state.feedback_email == DEFAULT_FEEDBACK_EMAIL


class TestFeedbackStoreLocation:
    def test_store_sits_under_the_shared_data_root(self, project_dir, shared_root):
        """``feedback_dir`` is ``<agent-data root>/feedback``."""
        with _lifespan_client(project_dir, shared_root) as (_client, app):
            assert app.state.feedback_dir == (shared_root / "feedback").resolve()

    def test_watch_dir_does_not_move_the_store(self, project_dir, shared_root, tmp_path):
        """``web_terminal.watch_dir`` moves the watcher, never the records.

        With ``watch_dir`` set, ``workspace_dir`` points outside the agent-data
        volume; records written there would be invisible to ``osprey feedback``.
        """
        watched = tmp_path / "watched_tree"
        watched.mkdir()
        with _lifespan_client(project_dir, shared_root, watch_dir=watched) as (_client, app):
            assert app.state.workspace_dir == watched.resolve()
            assert app.state.feedback_dir == (shared_root / "feedback").resolve()

    def test_relative_path_is_none_when_the_store_is_outside_the_watched_tree(
        self, project_dir, shared_root, tmp_path
    ):
        """A bare ``relative_to`` would raise here — the guard must return None."""
        watched = tmp_path / "watched_tree"
        watched.mkdir()
        with _lifespan_client(project_dir, shared_root, watch_dir=watched) as (_client, app):
            assert app.state.feedback_rel is None

    def test_relative_path_is_set_when_the_store_is_inside_the_watched_tree(
        self, project_dir, shared_root
    ):
        """Without ``watch_dir`` the two roots coincide, so concealment applies."""
        with _lifespan_client(project_dir, shared_root) as (_client, app):
            assert app.state.workspace_dir == shared_root.resolve()
            assert app.state.feedback_rel == Path("feedback")


class TestPanelsPayload:
    def test_configured_values_reach_the_browser(self, project_dir, shared_root):
        """The three UI-facing keys travel on ``GET /api/panels``."""
        overrides = {
            "web.docs_url": "https://docs.example.org/osprey",
            "web.feedback.github_repo": "facility/osprey-fork",
            "web.feedback.email": "controls@example.org",
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (client, _app):
            payload = client.get("/api/panels").json()
        assert payload["docs_url"] == "https://docs.example.org/osprey"
        assert payload["feedback_github_repo"] == "facility/osprey-fork"
        assert payload["feedback_email"] == "controls@example.org"

    def test_ceiling_is_not_exposed_to_the_browser(self, project_dir, shared_root):
        """The store ceiling is server-side only; nothing in the UI reads it."""
        with _lifespan_client(project_dir, shared_root) as (client, _app):
            payload = client.get("/api/panels").json()
        assert "feedback_max_store_bytes" not in payload

    def test_bare_app_state_still_serves_the_defaults(self):
        """The route never assumes the lifespan ran (getattr-with-default)."""
        app = FastAPI()
        app.include_router(panels_router)
        app.state.project_cwd = "/tmp"
        payload = TestClient(app).get("/api/panels").json()
        assert payload["docs_url"] == DEFAULT_DOCS_URL
        assert payload["feedback_github_repo"] == DEFAULT_FEEDBACK_GITHUB_REPO
        assert payload["feedback_email"] == DEFAULT_FEEDBACK_EMAIL
