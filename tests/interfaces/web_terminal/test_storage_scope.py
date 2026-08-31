"""Tests for the per-user browser-storage scope stamp (Task 3.1).

Multi-user deployments run one Web Terminal container per user behind a shared
nginx front door, each mounted at ``/u/<user>/``. Same origin, one shared
``localStorage`` — so every stored key (dock layout, rail position, recent
palette items, the active PTY session id, per-panel agent acks, …) collides
between users unless it is namespaced.

The namespace is decided **server side** and stamped onto ``<html>`` as
``data-osprey-storage-scope="<user>"``. That attribute is the single source of
truth every JS storage site reads; no JS parses ``location.pathname`` for the
mount user. Two properties make it usable as such:

- **Present exactly when there is a mount.** Under ``/u/<user>/`` the attribute
  carries the user; on a single-user / non-mounted deployment it is *absent*,
  not empty — so a reader can branch on presence alone, and the unscoped
  single-user markup stays byte-identical to what it was before this stamp.
- **On both served documents.** ``index.html`` and ``static/session.html`` are
  both Jinja-rendered, and the session page's module-import closure reaches the
  storage-owning modules (``terminal.js``, ``dock-workspace.js``,
  ``rail-position.js``, ``panel-agent-attention.js``, ``dock-reconcile.js``), so
  a stamp on only the index page would leave that page unscoped.
"""

from __future__ import annotations

import os
import re
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app, resolve_storage_scope

#: The attribute under test.
ATTR = "data-osprey-storage-scope"

# (page id, request path) -- both served HTML documents in scope, same set the
# prefix contract (test_prefix_injection.py) covers.
_PAGES = [
    ("index", "/"),
    ("session", "/static/session.html"),
]
_PAGE_IDS = [p[0] for p in _PAGES]


@pytest.fixture
def workspace_dir(tmp_path):
    """Temporary workspace directory for the app to watch."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _html_tag(body: str) -> str:
    """The document's opening ``<html …>`` tag.

    The stamp is only useful to a pre-paint reader if it sits on the root
    element, so the render assertions match against this rather than against
    the whole document.
    """
    match = re.search(r"<html\b[^>]*>", body)
    assert match, "served document has no <html> element"
    return match.group(0)


def _serve(workspace_dir, terminal_user: str | None):
    """Render both pages with ``OSPREY_TERMINAL_USER`` set to *terminal_user*.

    ``None`` removes the variable entirely (the single-user / non-mounted
    case). The env patch must wrap ``create_app`` as well as the requests: the
    URL prefix is computed at construction time and the terminal user is read
    during the lifespan, so a patch applied later would exercise neither.

    Returns:
        ``{page_id: response_body}`` for every page in :data:`_PAGES`.
    """
    env = {} if terminal_user is None else {"OSPREY_TERMINAL_USER": terminal_user}
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch.dict("os.environ", env),
    ):
        if terminal_user is None:
            os.environ.pop("OSPREY_TERMINAL_USER", None)
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            bodies = {}
            for page_id, path in _PAGES:
                response = client.get(path)
                assert response.status_code == 200, f"{page_id} did not render"
                bodies[page_id] = response.text
            return bodies


class TestResolveStorageScope:
    """Pure resolver: deployment identity -> storage namespace token."""

    def test_user_passes_through(self):
        assert resolve_storage_scope("alice") == "alice"

    def test_none_is_unscoped(self):
        """No mount user at all is the single-user deployment: no scope."""
        assert resolve_storage_scope(None) == ""

    def test_empty_is_unscoped(self):
        assert resolve_storage_scope("") == ""

    def test_whitespace_only_is_unscoped(self):
        """A blank env var means unset, exactly as ``compute_url_prefix`` reads it.

        The two must agree: a deployment that gets no ``/u/<user>`` prefix must
        also get no storage scope, or the attribute would claim a mount that
        does not exist.
        """
        assert resolve_storage_scope("   ") == ""

    def test_surrounding_whitespace_is_stripped(self):
        assert resolve_storage_scope("  alice\n") == "alice"

    def test_never_raises(self):
        for value in ("alice", "", "   ", None):
            try:
                resolve_storage_scope(value)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - failure path
                pytest.fail(f"resolve_storage_scope({value!r}) raised: {exc}")


class TestMountedRender:
    """``OSPREY_TERMINAL_USER=alice`` -> the scope is stamped on both pages."""

    @pytest.mark.parametrize("page_id", _PAGE_IDS)
    def test_scope_stamped_on_html_element(self, workspace_dir, page_id):
        body = _serve(workspace_dir, "alice")[page_id]
        assert f'{ATTR}="alice"' in _html_tag(body)

    @pytest.mark.parametrize("page_id", _PAGE_IDS)
    def test_stamped_exactly_once(self, workspace_dir, page_id):
        """One authoritative stamp — a second would make "the" source ambiguous."""
        body = _serve(workspace_dir, "alice")[page_id]
        assert body.count(ATTR) == 1

    def test_index_keeps_its_existing_stamps(self, workspace_dir):
        """The new attribute joins the existing ones; it does not displace them."""
        tag = _html_tag(_serve(workspace_dir, "alice")["index"])
        assert "data-ui-mode=" in tag
        assert "data-theme=" in tag
        assert "data-rail-position=" in tag


class TestUnmountedRender:
    """Single-user / non-mounted serving -> the attribute is absent entirely."""

    @pytest.mark.parametrize("page_id", _PAGE_IDS)
    def test_attribute_absent_when_user_unset(self, workspace_dir, page_id):
        body = _serve(workspace_dir, None)[page_id]
        assert ATTR not in body

    @pytest.mark.parametrize("page_id", _PAGE_IDS)
    def test_attribute_absent_when_user_blank(self, workspace_dir, page_id):
        """A blank value is unset, and must not stamp an empty scope.

        ``data-osprey-storage-scope=""`` would be present-but-meaningless: a
        reader branching on presence would namespace every key with the empty
        string instead of leaving storage unscoped.
        """
        body = _serve(workspace_dir, "   ")[page_id]
        assert ATTR not in body

    @pytest.mark.parametrize("page_id", _PAGE_IDS)
    def test_html_element_still_renders(self, workspace_dir, page_id):
        """Guard against the conditional swallowing the tag it decorates."""
        assert _html_tag(_serve(workspace_dir, None)[page_id]).startswith("<html")


class TestUnusualUsernames:
    """Roster-legal names must round-trip through the attribute exactly."""

    # Shapes a deploy roster can legitimately produce: dots, dashes,
    # underscores, digits, and mixed case (the scope is compared verbatim by
    # the JS readers, so case must survive).
    ROSTER_NAMES = [
        "first.last",
        "jean-luc",
        "op_2",
        "McTavish",
        "A.B-c_9",
    ]

    @pytest.mark.parametrize("user", ROSTER_NAMES)
    @pytest.mark.parametrize("page_id", _PAGE_IDS)
    def test_value_round_trips_verbatim(self, workspace_dir, user, page_id):
        body = _serve(workspace_dir, user)[page_id]
        assert f'{ATTR}="{user}"' in _html_tag(body)

    @pytest.mark.parametrize("page_id", _PAGE_IDS)
    def test_scope_matches_the_url_prefix_user(self, workspace_dir, page_id):
        """The stamped scope is the same user the ``/u/<user>`` prefix names.

        These are the two halves of one deployment identity; a divergence would
        namespace storage under a user the page is not actually mounted for.
        """
        body = _serve(workspace_dir, "first.last")[page_id]
        assert 'window.__OSPREY_PREFIX__ = "/u/first.last";' in body
        assert f'{ATTR}="first.last"' in _html_tag(body)

    @pytest.mark.parametrize("page_id", _PAGE_IDS)
    def test_value_is_html_escaped(self, workspace_dir, page_id):
        """Autoescape stays on for the value.

        Names come from the deploy roster, not from a request, so this is not
        the last line of defence — but a value that could close the attribute
        would rewrite the document's root element, and the template must never
        be the reason that is possible.
        """
        body = _serve(workspace_dir, 'a"><script>x</script>')[page_id]
        assert "<script>x</script>" not in body
        assert f'{ATTR}="a&#34;&gt;&lt;script&gt;' in _html_tag(body)
