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
from urllib.parse import unquote_plus

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.feedback_destination import (
    DEFAULT_DOCS_URL,
    DEFAULT_FEEDBACK_EMAIL,
    DEFAULT_FEEDBACK_GITHUB_REPO,
    DEFAULT_FEEDBACK_MAX_STORE_BYTES,
    coerce_config_str,
    coerce_feedback_trackers,
    coerce_store_ceiling,
    resolve_deployment_identity,
    resolve_feedback_destination,
    resolve_feedback_trackers,
    upstream_escalation_url,
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
            # The bare deployment still reaches the maintainers: the shipped
            # `github_repo` default is the one tracker on offer.
            assert app.state.feedback_trackers == [
                {"kind": "github", "label": "GitHub", "repo": DEFAULT_FEEDBACK_GITHUB_REPO}
            ]

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
            assert app.state.feedback_trackers == []

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
        assert payload["feedback_trackers"] == [
            {"kind": "github", "label": "GitHub", "repo": "facility/osprey-fork"}
        ]
        assert payload["feedback_email"] == "controls@example.org"
        # The sugar key is resolved into the tracker list server-side; the
        # browser never sees it on its own.
        assert "feedback_github_repo" not in payload

    def test_configured_trackers_reach_the_browser_in_order(self, project_dir, shared_root):
        """A facility-authored list travels as written, sugar tracker last."""
        overrides = {
            "web.feedback.trackers": [
                {"kind": "gitlab", "url": "https://git.example.org/ops/osprey", "label": "Ops"},
                {"kind": "github", "repo": "facility/fork", "label": "Fork"},
            ],
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (client, _app):
            payload = client.get("/api/panels").json()
        assert payload["feedback_trackers"] == [
            {"kind": "gitlab", "label": "Ops", "url": "https://git.example.org/ops/osprey"},
            {"kind": "github", "label": "Fork", "repo": "facility/fork"},
            {"kind": "github", "label": "GitHub", "repo": DEFAULT_FEEDBACK_GITHUB_REPO},
        ]

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
        assert payload["feedback_trackers"] == [
            {"kind": "github", "label": "GitHub", "repo": DEFAULT_FEEDBACK_GITHUB_REPO}
        ]
        assert payload["feedback_email"] == DEFAULT_FEEDBACK_EMAIL


GITLAB_URL = "https://git.example.org/controls/osprey"
UPSTREAM_TRACKER = {"kind": "github", "label": "GitHub", "repo": DEFAULT_FEEDBACK_GITHUB_REPO}


class TestFeedbackTrackers:
    """``web.feedback.trackers`` — the facility-authored outbound tracker list."""

    def test_absent_list_is_empty(self):
        assert coerce_feedback_trackers(None) == []

    def test_github_and_gitlab_entries_are_normalised(self):
        raw = [
            {"kind": "github", "repo": " facility/fork ", "label": " Fork "},
            {"kind": "gitlab", "url": GITLAB_URL + "/", "label": "Facility GitLab"},
        ]
        assert coerce_feedback_trackers(raw) == [
            {"kind": "github", "label": "Fork", "repo": "facility/fork"},
            {"kind": "gitlab", "label": "Facility GitLab", "url": GITLAB_URL},
        ]

    def test_label_defaults_to_the_kind_name(self):
        raw = [{"kind": "gitlab", "url": GITLAB_URL}, {"kind": "github", "repo": "a/b"}]
        assert [t["label"] for t in coerce_feedback_trackers(raw)] == ["GitLab", "GitHub"]

    @pytest.mark.parametrize(
        "bad",
        [
            "not a list",
            {"kind": "github", "repo": "a/b"},
            42,
        ],
    )
    def test_a_non_list_is_reported_and_dropped(self, bad):
        assert coerce_feedback_trackers(bad) == []

    @pytest.mark.parametrize(
        "entry",
        [
            "a/b",
            {"kind": "bitbucket", "url": GITLAB_URL},
            {"kind": "github"},
            {"kind": "github", "repo": ""},
            {"kind": "github", "repo": "no-slash"},
            {"kind": "github", "repo": "a/b c"},
            {"kind": "gitlab"},
            {"kind": "gitlab", "url": "git.example.org/x"},
            {"kind": "gitlab", "repo": "a/b"},
            {"repo": "a/b"},
        ],
    )
    def test_a_malformed_entry_is_dropped_and_the_rest_kept(self, entry):
        """One bad line must not take the whole channel list down with it."""
        raw = [entry, {"kind": "github", "repo": "keep/me"}]
        assert coerce_feedback_trackers(raw) == [
            {"kind": "github", "label": "GitHub", "repo": "keep/me"}
        ]

    def test_the_owner_tracker_is_appended_last(self):
        trackers = [{"kind": "gitlab", "label": "Ops", "url": GITLAB_URL}]
        assert resolve_feedback_trackers(trackers, UPSTREAM_TRACKER) == [
            {"kind": "gitlab", "label": "Ops", "url": GITLAB_URL},
            {"kind": "github", "label": "GitHub", "repo": "als-apg/osprey"},
        ]

    def test_no_owner_tracker_adds_nothing(self):
        """A retired owner channel (blank `github_repo`) appends nothing."""
        trackers = [{"kind": "gitlab", "label": "Ops", "url": GITLAB_URL}]
        assert resolve_feedback_trackers(trackers, None) == trackers
        assert resolve_feedback_trackers([], None) == []

    def test_a_listed_tracker_wins_over_the_owner_duplicate(self):
        """Listing the same repo with its own label must not render it twice."""
        trackers = [{"kind": "github", "label": "OSPREY upstream", "repo": "als-apg/osprey"}]
        assert resolve_feedback_trackers(trackers, UPSTREAM_TRACKER) == trackers

    def test_duplicates_inside_the_list_collapse_to_the_first(self):
        trackers = [
            {"kind": "gitlab", "label": "One", "url": GITLAB_URL},
            {"kind": "gitlab", "label": "Two", "url": GITLAB_URL},
        ]
        assert resolve_feedback_trackers(trackers, None) == [trackers[0]]

    def test_lifespan_resolves_the_list_plus_sugar(self, project_dir, shared_root):
        overrides = {
            "web.feedback.trackers": [{"kind": "gitlab", "url": GITLAB_URL, "label": "Ops"}],
            "web.feedback.github_repo": "facility/fork",
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (_client, app):
            assert app.state.feedback_trackers == [
                {"kind": "gitlab", "label": "Ops", "url": GITLAB_URL},
                {"kind": "github", "label": "GitHub", "repo": "facility/fork"},
            ]

    def test_lifespan_with_list_and_blank_sugar_is_the_list_alone(self, project_dir, shared_root):
        """The ALS posture: a self-hosted tracker and no upstream channel."""
        overrides = {
            "web.feedback.trackers": [{"kind": "gitlab", "url": GITLAB_URL, "label": "Ops"}],
            "web.feedback.github_repo": "",
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (_client, app):
            assert app.state.feedback_trackers == [
                {"kind": "gitlab", "label": "Ops", "url": GITLAB_URL}
            ]

    def test_malformed_list_in_config_leaves_the_sugar_tracker(self, project_dir, shared_root):
        overrides = {"web.feedback.trackers": "https://git.example.org/x"}
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (_client, app):
            assert app.state.feedback_trackers == [
                {"kind": "github", "label": "GitHub", "repo": DEFAULT_FEEDBACK_GITHUB_REPO}
            ]


class TestResolveFeedbackDestination:
    """One resolver behind both the lifespan and the ``/api/panels`` fallback.

    The duplication this replaced lived on the fallback path, so a drift
    between the two spellings only showed when app state was missing. These
    tests pin the two paths to each other rather than to literals, so the same
    class of drift cannot come back through a copy that merely *looks* right.
    """

    def test_no_arguments_is_the_unconfigured_deployment(self):
        """A deployment that configured nothing is owned by the OSPREY project."""
        destination = resolve_feedback_destination()
        assert destination.docs_url == DEFAULT_DOCS_URL
        assert destination.email == DEFAULT_FEEDBACK_EMAIL
        assert destination.github_repo == DEFAULT_FEEDBACK_GITHUB_REPO
        assert destination.max_store_bytes == DEFAULT_FEEDBACK_MAX_STORE_BYTES
        assert destination.trackers == [
            {"kind": "github", "label": "GitHub", "repo": DEFAULT_FEEDBACK_GITHUB_REPO}
        ]

    def test_the_sugar_expansion_happens_once(self):
        """``github_repo`` becomes a tracker entry inside the resolver, not outside it."""
        destination = resolve_feedback_destination(github_repo="facility/ops")
        assert destination.github_repo == "facility/ops"
        assert destination.trackers == [
            {"kind": "github", "label": "GitHub", "repo": "facility/ops"}
        ]

    def test_a_blank_repo_retires_the_github_channel(self):
        """The blank posture survives the move into the resolver."""
        destination = resolve_feedback_destination(github_repo="")
        assert destination.github_repo == ""
        assert destination.trackers == []

    def test_configured_trackers_precede_the_sugar(self):
        """Render order is the facility's list, then the ``github_repo`` sugar."""
        destination = resolve_feedback_destination(
            trackers=[{"kind": "gitlab", "url": GITLAB_URL, "label": "Ops"}],
            github_repo="facility/ops",
        )
        assert destination.trackers == [
            {"kind": "gitlab", "label": "Ops", "url": GITLAB_URL},
            {"kind": "github", "label": "GitHub", "repo": "facility/ops"},
        ]

    def test_one_unusable_value_does_not_drag_the_others_to_defaults(self):
        """Each field is coerced separately — the fail-open posture app.py states."""
        destination = resolve_feedback_destination(
            email="controls@example.org",
            max_store_bytes="256MB",
        )
        assert destination.email == "controls@example.org"
        assert destination.max_store_bytes == DEFAULT_FEEDBACK_MAX_STORE_BYTES

    def test_every_field_is_returned_fresh(self):
        """No caller can mutate the next caller's trackers."""
        first = resolve_feedback_destination()
        first.trackers.append({"kind": "github", "label": "X", "repo": "a/b"})
        assert len(resolve_feedback_destination().trackers) == 1

    def test_the_panels_fallback_is_the_resolver_and_not_a_copy(self):
        """The route's defaults ARE the resolver's output, field for field.

        This is the test the whole commit exists for: it fails if anyone
        re-types a default into the ``getattr`` fallbacks, however carefully.
        """
        app = FastAPI()
        app.include_router(panels_router)
        app.state.project_cwd = "/tmp"
        payload = TestClient(app).get("/api/panels").json()

        unconfigured = resolve_feedback_destination()
        assert payload["docs_url"] == unconfigured.docs_url
        assert payload["feedback_email"] == unconfigured.email
        assert payload["feedback_trackers"] == unconfigured.trackers

    def test_the_lifespan_resolves_the_same_way_the_fallback_does(self, project_dir, shared_root):
        """A configured lifespan and a direct resolve agree on every field."""
        overrides = {
            "web.docs_url": "https://docs.example.org",
            "web.feedback.email": "controls@example.org",
            "web.feedback.github_repo": "facility/ops",
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (client, app):
            payload = client.get("/api/panels").json()

        expected = resolve_feedback_destination(
            docs_url="https://docs.example.org",
            email="controls@example.org",
            github_repo="facility/ops",
        )
        assert payload["docs_url"] == expected.docs_url
        assert payload["feedback_email"] == expected.email
        assert payload["feedback_trackers"] == expected.trackers
        assert app.state.feedback_max_store_bytes == expected.max_store_bytes


OWNER_GITLAB = {"kind": "gitlab", "target": GITLAB_URL}


class TestFeedbackOwner:
    """``web.feedback.owner`` — email and tracker, moved together.

    The block exists because a facility that redirects feedback has to move
    both, and moving one is the failure it is meant to prevent. The two leaf
    keys still win where they are spelled, so no already-deployed profile
    changes meaning by upgrading into this.
    """

    def test_owner_supplies_the_address_when_the_leaf_key_is_absent(self):
        destination = resolve_feedback_destination(
            owner={"name": "ALS Controls", "email": "controls@als.example.org"}
        )
        assert destination.email == "controls@als.example.org"
        assert destination.owner_name == "ALS Controls"

    def test_the_leaf_key_still_wins(self):
        """An already-deployed profile spelling the leaf key keeps its meaning."""
        destination = resolve_feedback_destination(
            email="legacy@example.org",
            owner={"email": "controls@als.example.org"},
        )
        assert destination.email == "legacy@example.org"

    def test_a_blank_leaf_key_still_retires_the_channel(self):
        """Blank is a posture, not an absence — it outranks the owner block."""
        destination = resolve_feedback_destination(
            email="", owner={"email": "controls@als.example.org"}
        )
        assert destination.email == ""

    def test_an_owner_gitlab_tracker_becomes_the_destination(self):
        """A GitLab facility is first-class: no `trackers:` list required."""
        destination = resolve_feedback_destination(owner={"tracker": OWNER_GITLAB})
        assert destination.trackers == [{"kind": "gitlab", "label": "GitLab", "url": GITLAB_URL}]
        assert destination.github_repo == ""

    def test_an_owner_github_tracker_takes_owner_slash_name(self):
        destination = resolve_feedback_destination(
            owner={"tracker": {"kind": "github", "target": "facility/ops"}}
        )
        assert destination.trackers == [
            {"kind": "github", "label": "GitHub", "repo": "facility/ops"}
        ]
        assert destination.github_repo == "facility/ops"

    def test_an_owner_tracker_may_be_captioned(self):
        destination = resolve_feedback_destination(
            owner={"tracker": {**OWNER_GITLAB, "label": "Controls GitLab"}}
        )
        assert destination.trackers[0]["label"] == "Controls GitLab"

    def test_the_leaf_repo_wins_over_the_owner_tracker(self):
        destination = resolve_feedback_destination(
            github_repo="legacy/repo", owner={"tracker": OWNER_GITLAB}
        )
        assert destination.trackers == [
            {"kind": "github", "label": "GitHub", "repo": "legacy/repo"}
        ]

    def test_a_blank_leaf_repo_retires_the_channel_over_the_owner_tracker(self):
        destination = resolve_feedback_destination(github_repo="", owner={"tracker": OWNER_GITLAB})
        assert destination.trackers == []

    def test_the_facility_tracker_list_still_precedes_the_owner(self):
        destination = resolve_feedback_destination(
            trackers=[{"kind": "github", "repo": "facility/fork", "label": "Fork"}],
            owner={"tracker": OWNER_GITLAB},
        )
        assert destination.trackers == [
            {"kind": "github", "label": "Fork", "repo": "facility/fork"},
            {"kind": "gitlab", "label": "GitLab", "url": GITLAB_URL},
        ]

    def test_no_owner_block_is_the_osprey_project(self):
        """The unconfigured deployment's owner is unchanged by this key."""
        destination = resolve_feedback_destination()
        assert destination.email == DEFAULT_FEEDBACK_EMAIL
        assert destination.github_repo == DEFAULT_FEEDBACK_GITHUB_REPO
        assert destination.owner_name == ""

    @pytest.mark.parametrize("bad", ["nonsense", 3, [], True])
    def test_an_unusable_owner_block_falls_back_without_taking_anything_down(self, bad):
        destination = resolve_feedback_destination(owner=bad)
        assert destination.email == DEFAULT_FEEDBACK_EMAIL
        assert destination.github_repo == DEFAULT_FEEDBACK_GITHUB_REPO
        assert destination.owner_name == ""

    def test_a_bad_owner_tracker_does_not_take_the_owner_email_with_it(self):
        """Fields are coerced separately here too."""
        destination = resolve_feedback_destination(
            owner={
                "email": "controls@als.example.org",
                "tracker": {"kind": "svn", "target": "whatever"},
            }
        )
        assert destination.email == "controls@als.example.org"
        assert destination.trackers == [
            {"kind": "github", "label": "GitHub", "repo": DEFAULT_FEEDBACK_GITHUB_REPO}
        ]

    def test_the_owner_block_reaches_the_running_deployment(self, project_dir, shared_root):
        """End to end: the lifespan reads the block and /api/panels echoes it."""
        overrides = {
            "web.feedback.owner": {
                "name": "ALS Controls",
                "email": "controls@als.example.org",
                "tracker": OWNER_GITLAB,
            }
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (client, app):
            payload = client.get("/api/panels").json()
        assert payload["feedback_email"] == "controls@als.example.org"
        assert payload["feedback_trackers"] == [
            {"kind": "gitlab", "label": "GitLab", "url": GITLAB_URL}
        ]
        assert app.state.feedback_owner_name == "ALS Controls"


class TestDeploymentIdentityAndEscalation:
    """What a forwarded report carries, and when the forwarding link exists.

    A user files to whoever owns the deployment because a user cannot know
    whether a bug is OSPREY's code or the facility's configuration. That only
    works if the maintainer who *can* tell forwards the framework bugs — and
    they only will if forwarding is nearly free.
    """

    IDENTITY = {
        "osprey_version": "2026.9.0b1",
        "preset": "control-assistant",
        "preset_hash": "a3f91c7d2e8b4f6a1c9d0e2f",
        "channel_finder_mode": "hierarchical",
    }

    def test_build_lines_omit_the_version(self):
        """Both report builders already have their own source for it."""
        identity = resolve_deployment_identity(**self.IDENTITY)
        assert "OSPREY version" not in identity.build_lines()
        assert identity.as_metadata()["OSPREY version"] == "2026.9.0b1"

    def test_the_preset_and_its_hash_render_as_one_fact(self):
        identity = resolve_deployment_identity(**self.IDENTITY)
        assert identity.build_lines()["Preset"].startswith("control-assistant (a3f91c")
        assert identity.build_lines()["Channel finder"] == "hierarchical"

    def test_a_preset_with_no_hash_still_renders(self):
        identity = resolve_deployment_identity(preset="control-assistant")
        assert identity.build_lines() == {"Preset": "control-assistant"}

    @pytest.mark.parametrize("bad", [None, 3, [], {"a": 1}, "   "])
    def test_unusable_identity_fields_are_dropped_not_printed(self, bad):
        identity = resolve_deployment_identity(preset=bad, channel_finder_mode=bad)
        assert identity.build_lines() == {}

    def test_a_configured_deployment_gets_an_escalation_link(self):
        destination = resolve_feedback_destination(
            owner={"name": "ALS Controls", "email": "c@x.org", "tracker": OWNER_GITLAB}
        )
        url = upstream_escalation_url(resolve_deployment_identity(**self.IDENTITY), destination)
        assert url.startswith(f"https://github.com/{DEFAULT_FEEDBACK_GITHUB_REPO}/issues/new?")
        decoded = unquote_plus(url)
        assert "control-assistant" in decoded
        assert "hierarchical" in decoded
        assert "2026.9.0b1" in decoded
        assert "ALS Controls" in decoded

    def test_the_unconfigured_deployment_gets_no_link(self):
        """Its owner IS the OSPREY project — the link would point at itself."""
        url = upstream_escalation_url(
            resolve_deployment_identity(**self.IDENTITY), resolve_feedback_destination()
        )
        assert url == ""

    def test_a_facility_that_also_files_upstream_gets_no_link(self):
        """Reports already reach the project, so there is nothing to forward."""
        destination = resolve_feedback_destination(
            trackers=[{"kind": "github", "repo": DEFAULT_FEEDBACK_GITHUB_REPO, "label": "Up"}],
            owner={"email": "c@x.org", "tracker": OWNER_GITLAB},
        )
        assert upstream_escalation_url(resolve_deployment_identity(), destination) == ""

    def test_the_link_carries_no_user_content(self):
        """It is built once at startup, so nothing per-report can leak into it."""
        destination = resolve_feedback_destination(owner={"tracker": OWNER_GITLAB})
        url = upstream_escalation_url(resolve_deployment_identity(**self.IDENTITY), destination)
        decoded = unquote_plus(url)
        assert "session" not in decoded.lower()
        assert "paste what the user reported" in decoded

    def test_an_identity_that_cannot_be_read_still_yields_a_link(self):
        """A deployment with no provenance can still forward a bug."""
        destination = resolve_feedback_destination(owner={"tracker": OWNER_GITLAB})
        url = upstream_escalation_url(resolve_deployment_identity(), destination)
        assert url.startswith("https://github.com/")

    def test_the_running_deployment_publishes_both(self, project_dir, shared_root):
        """End to end: lifespan resolves them, /api/panels echoes them."""
        overrides = {
            "web.feedback.owner": {"name": "ALS Controls", "tracker": OWNER_GITLAB},
            "provenance.preset": "control-assistant",
            "provenance.preset_hash": "a3f91c7d2e8b4f6a1c9d0e2f",
            "channel_finder.pipeline_mode": "hierarchical",
        }
        with _lifespan_client(project_dir, shared_root, overrides=overrides) as (client, _app):
            payload = client.get("/api/panels").json()
        assert payload["feedback_deployment"]["Channel finder"] == "hierarchical"
        assert payload["feedback_deployment"]["Preset"].startswith("control-assistant (a3f91c")
        assert DEFAULT_FEEDBACK_GITHUB_REPO in payload["feedback_escalation_url"]

    def test_an_unconfigured_deployment_publishes_an_empty_link(self, project_dir, shared_root):
        with _lifespan_client(project_dir, shared_root) as (client, _app):
            payload = client.get("/api/panels").json()
        assert payload["feedback_escalation_url"] == ""

    def test_the_route_survives_a_lifespan_that_never_ran(self):
        """The getattr fallbacks cover the new fields too."""
        app = FastAPI()
        app.include_router(panels_router)
        app.state.project_cwd = "/tmp"
        payload = TestClient(app).get("/api/panels").json()
        assert payload["feedback_deployment"] == {}
        assert payload["feedback_escalation_url"] == ""
