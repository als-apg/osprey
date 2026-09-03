"""Server-rendered bar items: ordered shells, the adopted-node pool, the stamp.

The web terminal's header and status bar are item HOSTS: ``root()`` renders one
ordered run of ``.bar-item`` shells per bar from the effective layout, so the
operator's order is on the page before a single module script has run.

Five chrome nodes are ADOPTED — they are resolved by id by other modules
(``#docs-link``, ``#command-palette-btn``, ``#display-menu-settings``,
``#logout-btn``, the identity trigger and menu), so
they are rendered on **every** request whichever way the
layout falls: into their shell when it places them, into ``#bar-item-pool``
when it does not. That is the invariant this file exists for (FR6), and it is
asserted in both directions:

* the pool never **subtracts** — a saved layout naming none of the adopted
  types still resolves every one of them, in the pool;
* the pool never **adds** — an id this deployment would not render today
  (an identity block with no user) is absent from the shells and from the
  pool alike.

Two fixtures carry that: a plain single-user app, and a configured one
(``terminal_user`` + ``landing_url`` + the SYSTEM panel enabled).
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    ADOPTED_BAR_ITEM_TYPES,
    BAR_ITEM_AVAILABILITY,
    BAR_ITEM_GATES,
    BAR_ITEM_OPTIONS,
    BAR_ITEM_TYPES,
    BAR_LAYOUT_VERSION,
    DEFAULT_BAR_LAYOUT,
    STATIC_DIR,
    SYSTEM_HEALTH_PANEL_ID,
    bar_item_available,
    bar_render_plan,
    create_app,
)

#: Ids the terminal renders on every deployment, whatever the layout says.
_UNIVERSAL_IDS = (
    "docs-link",
    "command-palette-btn",
    "display-menu-settings",
)

#: Ids that additionally exist once the deployment knows a user and has a
#: landing page to send them back to.
_IDENTITY_IDS = (
    "header-identity-trigger",
    "header-identity-menu",
    "logout-btn",
    "display-menu-logout-btn",
)

#: A saved layout naming none of the adopted types: every adopted node must
#: therefore be resolvable in the pool instead of in a bar.
_LAYOUT_WITHOUT_ADOPTED_ITEMS = {
    "version": 1,
    "rev": 7,
    "header": [{"type": "control-target"}, {"type": "space"}],
    "status": [{"type": "clock"}],
    "status_visible": True,
}


@pytest.fixture
def workspace_dir(tmp_path):
    """A temporary workspace directory for the app to watch."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "README.md").write_text("# Test workspace\n")
    return ws


def _build_app(workspace_dir, *, enabled_panels=None, custom_panels=None, env=None):
    """Boot an app the way the other route tests do, and return (app, client).

    Args:
        workspace_dir: The watched directory.
        enabled_panels: Enabled built-in panel ids, or None for the universal set.
        custom_panels: Config-declared panel dicts, or None for none.
        env: Environment overrides applied across ``create_app`` and lifespan.
    """
    panels = {"artifacts"} if enabled_panels is None else set(enabled_panels)
    return (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(panels, list(custom_panels or []), None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
        ),
        patch.dict("os.environ", env or {}, clear=False),
    )


@pytest.fixture
def plain_app(workspace_dir):
    """A single-user deployment: no terminal user, no landing page, no SYSTEM panel."""
    cfg, panels, launch, env = _build_app(workspace_dir)
    with cfg, panels, launch, env:
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            yield app, client


@pytest.fixture
def configured_app(workspace_dir):
    """A multi-user deployment with an identity, a way out and the SYSTEM panel."""
    cfg, panels, launch, env = _build_app(
        workspace_dir,
        enabled_panels={"artifacts", "system-health"},
        env={
            "OSPREY_TERMINAL_USER": "alice",
            "OSPREY_TERMINAL_LANDING_URL": "https://facility.example/portal",
        },
    )
    with cfg, panels, launch, env:
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            yield app, client


def _body(client) -> str:
    """GET the hub page and return its markup, asserting it rendered at all."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


def _shell_types(body: str, host: str) -> list[str]:
    """The ordered item types of one host's shell run.

    Args:
        body: The rendered document.
        host: ``header`` or ``status``.

    Returns:
        Every ``data-bar-item`` value inside that host, in document order.
    """
    if host == "header":
        opening = body.index('data-bar-host="header"')
        closing = body.index("</header>", opening)
    else:
        opening = body.index('data-bar-host="status"')
        closing = body.index("</footer>", opening)
    return re.findall(r'data-bar-item="([^"]+)"', body[opening:closing])


class _SubtreeExtractor(HTMLParser):
    """Collect the markup of one element by id, bounded by its own nesting.

    Bounding the pool on the document's whitespace (the comment that follows
    it, say) makes every pool assertion in this file depend on the template's
    indentation, and over-reads into live markup the moment an element is added
    between the two. Depth-counting bounds it on the element itself.
    """

    def __init__(self, element_id: str) -> None:
        super().__init__(convert_charrefs=False)
        self._wanted = element_id
        self._depth = 0
        self._chunks: list[str] = []
        self.found = False

    _VOID_TAGS = frozenset("area base br col embed hr img input link meta source track wbr".split())

    def handle_starttag(self, tag, attrs):
        if tag in self._VOID_TAGS:
            # HTMLParser never emits an end tag for these, so counting them
            # into the depth would run the subtree to EOF.
            self.handle_startendtag(tag, attrs)
            return
        if self._depth:
            self._depth += 1
        elif dict(attrs).get("id") == self._wanted:
            self._depth = 1
            self.found = True
            return
        if self._depth:
            self._chunks.append(self.get_starttag_text() or "")

    def handle_startendtag(self, tag, attrs):
        if self._depth:
            self._chunks.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag):
        if not self._depth:
            return
        self._depth -= 1
        if self._depth:
            self._chunks.append(f"</{tag}>")

    def handle_data(self, data):
        if self._depth:
            self._chunks.append(data)

    @property
    def markup(self) -> str:
        return "".join(self._chunks)


def _subtree(body: str, element_id: str) -> str:
    """The inner markup of the element with *element_id*."""
    parser = _SubtreeExtractor(element_id)
    parser.feed(body)
    assert parser.found, f"#{element_id} is not in the document"
    return parser.markup


def _pool(body: str) -> str:
    """The markup inside ``#bar-item-pool``, bounded by the pool element."""
    return _subtree(body, "bar-item-pool")


def _has_id(markup: str, element_id: str) -> bool:
    """Whether *markup* declares ``id="<element_id>"``."""
    return f'id="{element_id}"' in markup


def _html_tag(body: str) -> str:
    """The document's opening ``<html …>`` tag, where the pre-paint layout
    attributes are stamped. Matched on the tag rather than on the whole
    document, which also mentions the attribute in prose."""
    return body[body.index("<html ") : body.index(">", body.index("<html "))]


class TestShellRuns:
    """The two hosts render the effective layout's order."""

    def test_header_run_matches_the_default_layout(self, configured_app):
        _, client = configured_app
        expected = [item["type"] for item in DEFAULT_BAR_LAYOUT["header"]]
        assert _shell_types(_body(client), "header") == expected

    def test_status_run_matches_the_default_layout(self, configured_app):
        """The shipped default is nearly bare: a space, the system-health dot
        and the clock at the right edge. Spelled out rather than read back
        from the constant, so a change to what a fresh deployment shows is a
        change to this file."""
        _, client = configured_app
        assert _shell_types(_body(client), "status") == ["space", "system-health", "clock"]

    def test_the_plain_status_run_drops_the_health_dot(self, plain_app):
        """``system-health`` is in the default because a deployment that
        enables the SYSTEM panel should show it; one that does not renders the
        rest and no empty shell in its place."""
        _, client = plain_app
        assert _shell_types(_body(client), "status") == ["space", "clock"]

    def test_saved_layout_is_rendered_in_its_own_order(self, plain_app):
        """The seam the phase-2 store plugs into, exercised end to end."""
        app, client = plain_app
        app.state.bar_layout = {
            "version": 1,
            "rev": 3,
            "header": [{"type": "display"}, {"type": "space"}, {"type": "logo"}],
            "status": [{"type": "docs"}, {"type": "stopwatch"}],
            "status_visible": True,
        }
        body = _body(client)
        assert _shell_types(body, "header") == ["display", "space", "logo"]
        assert _shell_types(body, "status") == ["docs", "stopwatch"]

    def test_shells_name_the_item_they_follow(self, configured_app):
        """`data-follows` is how the identity middot knows it sits after the
        logo — the one ordering fact an item is allowed to have, stamped by the
        host rather than inferred by a CSS sibling combinator."""
        body = _body(configured_app[1])
        header = body[body.index('data-bar-host="header"') : body.index("</header>")]
        first, second = re.findall(r'<div class="bar-item"[^>]*>', header)[:2]
        assert "data-follows" not in first
        assert 'data-follows="logo"' in second

    def test_a_layout_this_build_cannot_read_falls_back_to_the_default(self, plain_app):
        """A document from a newer build is refused here as well as on the
        client. Rendering it anyway would paint one arrangement and hydrate
        into another — with the pool computed from a layout the client has
        already discarded."""
        app, client = plain_app
        app.state.bar_layout = {
            "version": BAR_LAYOUT_VERSION + 98,
            "rev": 12,
            "header": [{"type": "display"}],
            "status": [],
            "header_visible": False,
            "status_visible": False,
        }
        body = _body(client)
        assert _shell_types(body, "header") == [
            item["type"] for item in DEFAULT_BAR_LAYOUT["header"] if item["type"] != "identity"
        ]
        assert "data-status-bar" not in _html_tag(body)
        assert "data-header-bar" not in _html_tag(body)


class TestUnavailableItemsAreAbsentNotEmpty:
    """The catalog's words: "absent rather than empty"."""

    def test_the_plain_header_has_no_identity_shell_at_all(self, plain_app):
        """An empty shell is not free: bars.css keys the identity middot off
        `[data-follows="logo"]` on the SHELL, so an empty identity shell would
        paint `OSPREY ·` with nothing after it on every single-user
        deployment — and cost the run another gap."""
        body = _body(plain_app[1])
        assert _shell_types(body, "header") == [
            "logo",
            "space",
            "control-target",
            "search",
            "display",
        ]

    def test_follows_chains_off_the_previous_emitted_item(self, plain_app):
        """With identity skipped, the middot belongs to whatever actually sits
        after the logo — here the space item — not to a shell that is no longer
        the logo's neighbour."""
        body = _body(plain_app[1])
        header = body[body.index('data-bar-host="header"') : body.index("</header>")]
        shells = re.findall(r'<div class="bar-item"[^>]*>', header)
        assert 'data-bar-item="space"' in shells[1]
        assert 'data-follows="logo"' in shells[1]
        assert body.count('data-follows="logo"') == 1

    def test_the_plain_status_bar_has_no_system_health_shell(self, plain_app):
        """The SYSTEM panel is not enabled here, so the item is absent — not an
        empty shell with nothing to read."""
        assert "system-health" not in _shell_types(_body(plain_app[1]), "status")


class TestKnownTypes:
    """Any known type renders in either bar; an unknown one renders nowhere."""

    def test_a_type_named_in_the_status_bar_renders_there(self, plain_app):
        """The wordmark used to be refused from the status bar; every type may
        now sit in either bar, and the SSR paints it where the layout says."""
        app, client = plain_app
        app.state.bar_layout = {
            "version": BAR_LAYOUT_VERSION,
            "rev": 4,
            "header": [],
            "status": [{"type": "logo"}, {"type": "clock"}],
            "status_visible": True,
        }
        body = _body(client)
        assert _shell_types(body, "status") == ["logo", "clock"]
        footer = body[body.index('data-bar-host="status"') : body.index("</footer>")]
        assert "header-logo" in footer

    def test_an_adopted_item_in_the_status_bar_is_rendered_not_pooled(self, configured_app):
        """The node moves with its shell: placed in the status bar it is
        rendered there, and the pool holds no second copy of its ids."""
        app, client = configured_app
        app.state.bar_layout = {
            "version": BAR_LAYOUT_VERSION,
            "rev": 5,
            "header": [{"type": "space"}],
            "status": [{"type": "identity"}],
            "status_visible": True,
        }
        body = _body(client)
        assert "identity" in _shell_types(body, "status")
        footer = body[body.index('data-bar-host="status"') : body.index("</footer>")]
        for element_id in ("header-identity-trigger", "header-identity-menu", "logout-btn"):
            assert _has_id(footer, element_id), element_id
            assert not _has_id(_pool(body), element_id), element_id

    def test_an_unknown_type_renders_nothing(self, plain_app):
        """The client's normalizer drops an unknown type; so does this."""
        app, client = plain_app
        app.state.bar_layout = {
            "version": BAR_LAYOUT_VERSION,
            "rev": 6,
            "header": [{"type": "logo"}, {"type": "not-a-real-item"}],
            "status": [],
            "status_visible": True,
        }
        body = _body(client)
        assert _shell_types(body, "header") == ["logo"]
        assert "not-a-real-item" not in body

    def test_the_type_table_mirrors_the_js_catalog(self):
        """The server's copy exists because it renders first; a copy that
        drifts is worse than none, so it is pinned against the source, in the
        catalog's own order."""
        source = (STATIC_DIR / "js" / "bar-catalog.js").read_text()
        declared = re.findall(r"^  '?([a-z-]+)'?: \{\n    type: '", source, re.MULTILINE)
        assert declared, "could not read the catalog's type declarations"
        assert list(BAR_ITEM_TYPES) == declared

    def test_no_entry_declares_a_placement_axis(self):
        """`hosts` is gone from both sides; an entry that grew it back would
        refuse a bar the server had already painted it in."""
        source = (STATIC_DIR / "js" / "bar-catalog.js").read_text()
        assert re.search(r"^    hosts:", source, re.MULTILINE) is None


#: One catalog entry's ``options:`` declaration — either ``NO_OPTIONS`` or an
#: ``Object.freeze({...})`` block. The block never nests braces (``enumSpec``'s
#: values are a bracketed array), so the first ``})`` closes it.
_CATALOG_OPTIONS = re.compile(
    r"^  '?([a-z-]+)'?: \{\n(?:.*\n)*?    options: (NO_OPTIONS|Object\.freeze\(\{[^}]*\}\)),",
    re.MULTILINE,
)

#: One option inside such a block: ``name: numberSpec(...)`` and friends. The
#: argument list allows one level of nesting, which is what ``enumSpec``'s
#: ``Object.freeze([...])`` needs.
_CATALOG_OPTION = re.compile(
    r"(\w+): (numberSpec|booleanSpec|enumSpec)\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
)


def _js_args(raw: str) -> list[str]:
    """Split a JS argument list on its top-level commas."""
    args: list[str] = []
    depth = 0
    current = ""
    for char in raw:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            args.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        args.append(current.strip())
    return args


def _js_value(raw: str) -> object:
    """One JS literal as its Python equivalent."""
    if raw in ("true", "false"):
        return raw == "true"
    if raw == "null":
        return None
    if raw.startswith(("'", '"')):
        return raw[1:-1]
    return int(raw)


def _catalog_option_specs() -> dict[str, dict[str, dict]]:
    """Every option spec in ``bar-catalog.js``, as the Python table spells them.

    Reproduces what the catalog's three builders return, which is why their
    signatures are pinned alongside: a default that moved from ``step = 1``
    would otherwise change every number spec with nothing to catch it.
    """
    source = (STATIC_DIR / "js" / "bar-catalog.js").read_text()
    assert "function numberSpec(min, max, fallback, unit = null, step = 1)" in source
    assert "function booleanSpec(fallback)" in source
    assert "function enumSpec(values, fallback)" in source

    declared: dict[str, dict[str, dict]] = {}
    for item_type, block in _CATALOG_OPTIONS.findall(source):
        specs: dict[str, dict] = {}
        for name, builder, raw_args in _CATALOG_OPTION.findall(block):
            args = _js_args(raw_args)
            if builder == "booleanSpec":
                specs[name] = {"kind": "boolean", "default": _js_value(args[0])}
            elif builder == "numberSpec":
                values = [_js_value(arg) for arg in args]
                specs[name] = {
                    "kind": "number",
                    "min": values[0],
                    "max": values[1],
                    "step": values[4] if len(values) > 4 else 1,
                    "unit": values[3] if len(values) > 3 else None,
                    "default": values[2],
                }
            else:
                inner = args[0][args[0].index("[") + 1 : args[0].rindex("]")]
                specs[name] = {
                    "kind": "enum",
                    "values": tuple(_js_value(v) for v in _js_args(inner)),
                    "default": _js_value(args[1]),
                }
        if specs:
            declared[item_type] = specs
    return declared


class TestOptionSpecsMirrorTheJsCatalog:
    """``BAR_ITEM_OPTIONS`` is the second half of the catalog mirror.

    The store validates a saved layout against these specs, so a bound that
    drifts is worse than one-sided: the customize sheet would offer a value the
    store then refuses with a 422 the client cannot explain. Pinned here rather
    than restated, exactly as the type table is.
    """

    def test_the_scrape_finds_the_types_that_declare_options(self):
        """A scrape that silently matched nothing would pass every assertion
        below."""
        assert set(_catalog_option_specs()) == {
            "clock",
            "space",
            "system-health",
            "bluesky-queue",
        }

    def test_the_option_table_mirrors_the_js_catalog(self):
        assert BAR_ITEM_OPTIONS == _catalog_option_specs()

    def test_every_other_type_declares_no_options(self):
        """``NO_OPTIONS`` in the catalog is absence here, which
        ``bar_item_vocabulary()`` reads as an empty spec mapping."""
        assert set(BAR_ITEM_OPTIONS) < set(BAR_ITEM_TYPES)
        assert "logo" not in BAR_ITEM_OPTIONS


class TestAdoptedNodesArePresentOnEveryDeployment:
    """The FR6 invariant, in both fixtures and in both layouts."""

    def test_plain_deployment_renders_its_universal_ids(self, plain_app):
        body = _body(plain_app[1])
        for element_id in _UNIVERSAL_IDS:
            assert _has_id(body, element_id), element_id

    def test_configured_deployment_adds_identity(self, configured_app):
        body = _body(configured_app[1])
        for element_id in (*_UNIVERSAL_IDS, *_IDENTITY_IDS):
            assert _has_id(body, element_id), element_id

    @pytest.mark.parametrize("fixture", ["plain_app", "configured_app"])
    def test_a_layout_naming_none_of_them_still_resolves_every_one(self, fixture, request):
        """Nothing is subtracted: what leaves the bars lands in the pool."""
        app, client = request.getfixturevalue(fixture)
        app.state.bar_layout = _LAYOUT_WITHOUT_ADOPTED_ITEMS
        body = _body(client)
        pool = _pool(body)

        expected = list(_UNIVERSAL_IDS)
        if fixture == "configured_app":
            expected += list(_IDENTITY_IDS)
        for element_id in expected:
            assert _has_id(body, element_id), element_id
            assert _has_id(pool, element_id), f"{element_id} is live, not pooled"

    @pytest.mark.parametrize("fixture", ["plain_app", "configured_app"])
    def test_a_parked_item_is_a_shell_exactly_as_it_is_in_a_bar(self, fixture, request):
        """bar-host.js indexes ``.bar-item[data-bar-item]`` and nothing else.

        A bare body parked in the pool would be invisible to the reconcile:
        placing the item back would build a fresh empty shell while the real
        node stayed orphaned here, and every id it owns would point at a node
        no bar contains. So the pooled branch emits the same shell the host
        run does — one per parked type, with the body inside it.
        """
        app, client = request.getfixturevalue(fixture)
        app.state.bar_layout = _LAYOUT_WITHOUT_ADOPTED_ITEMS
        pool = _pool(_body(client))

        parked = re.findall(r'<div class="bar-item" data-bar-item="([^"]+)"', pool)
        expected = ["logo", "search", "display", "docs"]
        if fixture == "configured_app":
            expected += ["identity"]
        assert sorted(parked) == sorted(expected)
        # The bodies sit INSIDE those shells: the first thing in the pool is a
        # shell, never an adopted node.
        assert pool.index('<div class="bar-item"') < pool.index('id="docs-link"')

    @pytest.mark.parametrize("fixture", ["plain_app", "configured_app"])
    def test_every_adopted_node_is_rendered_exactly_once(self, fixture, request):
        """A duplicated id is worse than a missing one — getElementById picks
        one silently. Asserted under a layout that names every adopted type
        twice, in both hosts."""
        app, client = request.getfixturevalue(fixture)
        doubled = [{"type": item_type} for item_type in ADOPTED_BAR_ITEM_TYPES] * 2
        app.state.bar_layout = {
            "version": 1,
            "rev": 1,
            "header": doubled,
            "status": doubled,
            "status_visible": True,
        }
        body = _body(client)
        expected = list(_UNIVERSAL_IDS)
        if fixture == "configured_app":
            expected += list(_IDENTITY_IDS)
        for element_id in expected:
            assert body.count(f'id="{element_id}"') == 1, element_id

    def test_the_pool_element_exists_even_when_it_is_empty(self, plain_app):
        """A pool that appears only when something is missing is a pool every
        reader has to check for before it can use it."""
        app, client = plain_app
        # Place every adopted type this deployment renders, so nothing is parked.
        app.state.bar_layout = {
            "version": 1,
            "rev": 1,
            "header": [{"type": t} for t in ("logo", "search", "display")],
            "status": [{"type": "docs"}],
            "status_visible": True,
        }
        body = _body(client)
        assert 'id="bar-item-pool"' in body
        assert _pool(body).strip() == ""


class TestNothingIsAddedThatTheDeploymentWouldNotRender:
    """The other half of FR6 — the pool is not a place to invent ids."""

    def test_no_identity_block_without_a_user_or_a_deployment_name(self, plain_app):
        app, client = plain_app
        app.state.bar_layout = _LAYOUT_WITHOUT_ADOPTED_ITEMS
        body = _body(client)
        for element_id in _IDENTITY_IDS:
            assert not _has_id(body, element_id), element_id

    def test_no_logout_without_a_landing_url(self, workspace_dir):
        """A user with nowhere to return to is still identified — but the two
        logout controls are the ACTION, and the action needs a destination."""
        cfg, panels, launch, env = _build_app(workspace_dir, env={"OSPREY_TERMINAL_USER": "alice"})
        with cfg, panels, launch, env:
            app = create_app(shell_command="echo")
            with TestClient(app) as client:
                body = _body(client)
        assert _has_id(body, "header-identity-trigger")
        assert not _has_id(body, "logout-btn")
        assert not _has_id(body, "display-menu-logout-btn")
        assert "data-landing-url=" not in body

    def test_the_landing_url_is_stamped_on_html_beside_the_buttons(self, configured_app):
        """The command palette's Log out reads the landing URL off ``<html>``,
        so it works with both logout buttons removed from the bars. Stamped
        under exactly the condition that renders the buttons."""
        body = _body(configured_app[1])
        assert _has_id(body, "logout-btn")
        assert 'data-landing-url="https://facility.example/portal"' in body

    def test_no_landing_url_on_html_without_a_user(self, plain_app):
        body = _body(plain_app[1])
        assert "data-landing-url=" not in body


class TestStatusBarStamp:
    """``html[data-status-bar="hidden"]`` — a pre-paint fact, like the theme."""

    def test_absent_while_the_bar_is_shown(self, plain_app):
        assert "data-status-bar" not in _html_tag(_body(plain_app[1]))

    def test_stamped_when_the_layout_hides_the_bar(self, configured_app):
        app, client = configured_app
        app.state.bar_layout = {**DEFAULT_BAR_LAYOUT, "status_visible": False}
        body = _body(client)
        assert 'data-status-bar="hidden"' in _html_tag(body)
        # Hiding the bar must not empty it: the items are still rendered, so
        # showing it again is a CSS fact rather than a re-render.
        assert _shell_types(body, "status") == [
            item["type"] for item in DEFAULT_BAR_LAYOUT["status"]
        ]

    @pytest.mark.parametrize("stored", [False, None, 0, "", "false"])
    def test_the_flag_is_coerced_not_identity_compared(self, plain_app, stored):
        """The document is JSON from a store: `null` (a key written but never
        set) and `0` both reach here meaning "not visible" to every other
        reader, and a bar that stayed shown for them would disagree with the
        client silently, before the first paint. A non-empty string is truthy
        and shows the bar — the same answer `Boolean()` gives on the client."""
        app, client = plain_app
        app.state.bar_layout = {**DEFAULT_BAR_LAYOUT, "status_visible": stored}
        hidden = 'data-status-bar="hidden"' in _html_tag(_body(client))
        assert hidden is not bool(stored)


class TestHeaderBarStamp:
    """``html[data-header-bar="hidden"]`` — the header goes the same way."""

    def test_absent_while_the_bar_is_shown(self, plain_app):
        assert "data-header-bar" not in _html_tag(_body(plain_app[1]))

    def test_stamped_when_the_layout_hides_the_bar(self, configured_app):
        app, client = configured_app
        app.state.bar_layout = {**DEFAULT_BAR_LAYOUT, "header_visible": False}
        body = _body(client)
        assert 'data-header-bar="hidden"' in _html_tag(body)
        assert "data-status-bar" not in _html_tag(body)
        # Hidden, not emptied: the items are still rendered, so showing the bar
        # again is a CSS fact rather than a re-render.
        assert _shell_types(body, "header") == [
            item["type"] for item in DEFAULT_BAR_LAYOUT["header"]
        ]

    @pytest.mark.parametrize("stored", [False, None, 0, "", "false"])
    def test_the_flag_is_coerced_not_identity_compared(self, plain_app, stored):
        app, client = plain_app
        app.state.bar_layout = {**DEFAULT_BAR_LAYOUT, "header_visible": stored}
        hidden = 'data-header-bar="hidden"' in _html_tag(_body(client))
        assert hidden is not bool(stored)


#: A saved layout naming the identity block, which only a deployment that
#: knows who is signed in can render.
_LAYOUT_WITH_IDENTITY = {
    "version": 1,
    "rev": 4,
    "header": [{"type": "logo"}, {"type": "identity"}],
    "status": [{"type": "clock"}],
    "status_visible": True,
}

#: One catalog entry's ``available:`` declaration: ``ALWAYS`` for the items
#: every deployment renders, an arrow taking ``ctx`` for the ones that ask a
#: deployment fact.
_CATALOG_AVAILABLE = re.compile(
    r"^  '?([a-z-]+)'?: \{\n(?:.*\n)*?    available: (ALWAYS|\(ctx\))",
    re.MULTILINE,
)


def _catalog_entry(source: str, item_type: str) -> str:
    """One ``bar-catalog.js`` entry's body, sliced on its own indentation.

    Sliced rather than matched: an entry-spanning regex over a 500-line file
    backtracks catastrophically, and the catalog's two-space entry indent is a
    more stable landmark than any pattern over its contents.
    """
    for opening in (f"\n  {item_type}: {{", f"\n  '{item_type}': {{"):
        start = source.find(opening)
        if start != -1:
            end = source.index("\n  },", start)
            return source[start:end]
    raise AssertionError(f"{item_type} is not declared in bar-catalog.js")


def _context(body: str) -> dict:
    """The deployment context stamped on ``<html>``, parsed.

    Single-quoted in the template because Jinja's ``tojson`` escapes ``'``
    (and ``<``, ``>``, ``&``) but leaves ``"`` alone, so a single-quoted
    attribute is the one that cannot be broken out of.
    """
    match = re.search(r"data-bar-context='([^']*)'", _html_tag(body))
    assert match, "the deployment context is not stamped on <html>"
    return json.loads(match.group(1))


class TestDeploymentContextIsServerSupplied:
    """The availability facts are evaluated once, on the server, and stamped.

    The client cannot infer them. Reading them off the rendered shells makes a
    deployment that OFFERS an item but does not PLACE it look like one that
    cannot render it at all — and an ``unavailable`` drop latches the whole
    document read-only, so saving goes silently dead for the session. The
    server knows all three facts; it says them.
    """

    def test_the_stamp_carries_exactly_the_keys_the_catalog_asks_for(self, plain_app):
        """``available(ctx)`` in ``bar-catalog.js`` defines this vocabulary."""
        assert set(_context(_body(plain_app[1]))) == {
            "identityAvailable",
            "blueskyAvailable",
            "systemHealthAvailable",
        }

    def test_a_plain_deployment_stamps_what_it_does_not_have(self, plain_app):
        assert _context(_body(plain_app[1])) == {
            "identityAvailable": False,
            "blueskyAvailable": False,
            "systemHealthAvailable": False,
        }

    def test_a_configured_deployment_stamps_its_identity_and_its_panel(self, configured_app):
        """Truthful in both halves: the stamp says what the page renders."""
        body = _body(configured_app[1])
        assert _context(body) == {
            "identityAvailable": True,
            "blueskyAvailable": False,
            "systemHealthAvailable": True,
        }
        assert "system-health" in _shell_types(body, "status")

    def test_no_shell_for_an_item_this_deployment_cannot_render(self, plain_app):
        """An unavailable item is ABSENT, not empty: a single-user deployment
        paints no ``identity`` shell — an empty box on first paint that the
        client would only take away again."""
        app, client = plain_app
        app.state.bar_layout = _LAYOUT_WITH_IDENTITY
        assert _shell_types(_body(client), "header") == ["logo"]

    def test_the_same_item_renders_once_the_deployment_has_an_identity(self, configured_app):
        app, client = configured_app
        app.state.bar_layout = _LAYOUT_WITH_IDENTITY
        body = _body(client)
        assert _shell_types(body, "header") == ["logo", "identity"]
        assert _context(body)["identityAvailable"] is True

    def test_an_unrenderable_item_is_not_seeded_into_the_pool_either(self, plain_app):
        """The pool holds the adopted nodes; an item this deployment cannot
        render has no node anywhere in the document."""
        app, client = plain_app
        app.state.bar_layout = _LAYOUT_WITH_IDENTITY
        body = _body(client)
        assert 'data-bar-item="identity"' not in body

    def test_a_second_copy_of_a_single_node_type_renders_no_shell(self, plain_app):
        """One node, one home — and no empty box for the copy either. The
        client's normalizer drops the duplicate, so a shell for it would only
        be taken away on the first reconcile."""
        app, client = plain_app
        app.state.bar_layout = {
            "version": 1,
            "rev": 4,
            "header": [{"type": "logo"}, {"type": "docs"}, {"type": "separator"}],
            "status": [{"type": "docs"}, {"type": "separator"}, {"type": "clock"}],
            "status_visible": True,
        }
        body = _body(client)
        assert _shell_types(body, "header") == ["logo", "docs", "separator"]
        assert _shell_types(body, "status") == ["separator", "clock"]
        assert body.count('data-bar-item="docs"') == 1

    def test_the_bluesky_fact_is_the_declared_panel(self, workspace_dir):
        """The plan-queue item reads the queue through the Bluesky panel's
        proxy, so the panel's declaration (``web.panels.bluesky``) is the one
        fact it is offered on — the same declaration that is the bridge
        entitlement, so no second key for the same fact."""
        cfg, panels, launch, env = _build_app(
            workspace_dir,
            custom_panels=[{"id": "bluesky", "label": "BLUESKY", "url": "http://bluesky-web:8080"}],
        )
        with cfg, panels, launch, env:
            app = create_app(shell_command="echo")
            with TestClient(app) as client:
                assert app.state.bluesky_available is True
                assert _context(_body(client))["blueskyAvailable"] is True

    def test_the_plan_queue_renders_where_the_panel_is_declared(self, workspace_dir):
        """An available, JS-built item gets its shell on first paint; the
        same layout on a deployment without the panel paints no shell."""
        layout = {
            "version": BAR_LAYOUT_VERSION,
            "rev": 7,
            "header": [{"type": "logo"}],
            "status": [{"type": "bluesky-queue"}, {"type": "clock"}],
            "status_visible": True,
        }
        cfg, panels, launch, env = _build_app(
            workspace_dir, custom_panels=[{"id": "bluesky", "url": "http://bluesky-web:8080"}]
        )
        with cfg, panels, launch, env:
            app = create_app(shell_command="echo")
            with TestClient(app) as client:
                app.state.bar_layout = layout
                assert _shell_types(_body(client), "status") == ["bluesky-queue", "clock"]
        cfg, panels, launch, env = _build_app(workspace_dir)
        with cfg, panels, launch, env:
            app = create_app(shell_command="echo")
            with TestClient(app) as client:
                app.state.bar_layout = layout
                assert _shell_types(_body(client), "status") == ["clock"]

    def test_the_availability_table_mirrors_the_js_catalog(self):
        """The server's copy of ``available()`` exists because it renders
        first; a copy that drifts is worse than none."""
        source = (STATIC_DIR / "js" / "bar-catalog.js").read_text()
        declared = dict(_CATALOG_AVAILABLE.findall(source))
        assert declared, "could not read the catalog's availability declarations"
        conditional = {item_type for item_type, form in declared.items() if form != "ALWAYS"}
        assert set(BAR_ITEM_AVAILABILITY) == conditional

    def test_each_side_gates_on_the_same_context_fact(self):
        """Which types are conditional is only half the mirror; WHICH FACT
        each one asks is the half that decides whether the two halves agree.

        The keys below are what the Python predicates in
        ``BAR_ITEM_AVAILABILITY`` read; this asserts the catalog's predicates
        read the same ones. It does not pin the comparison itself — an
        inverted test would still pass — which is what the shared
        context/expectation fixture in the follow-up ledger would close.
        """
        source = (STATIC_DIR / "js" / "bar-catalog.js").read_text()
        expected = {
            "identity": {"identityAvailable"},
            "bluesky-queue": {"blueskyAvailable"},
            "system-health": {"systemHealthAvailable"},
        }
        assert set(expected) == set(BAR_ITEM_AVAILABILITY), "a gated type grew or vanished"
        assert set(BAR_ITEM_GATES) == set(BAR_ITEM_AVAILABILITY), (
            "every gated type names what it needs, for the web.bar_items warning"
        )
        # The build-time half: every panel-gated type is judged by `osprey
        # build` too, and the one runtime-gated type (identity) is the only one
        # it leaves to the server.
        from osprey.cli.build_profile_panels import BAR_ITEM_PANEL_GATES

        assert set(BAR_ITEM_AVAILABILITY) - set(BAR_ITEM_PANEL_GATES) == {"identity"}
        assert BAR_ITEM_PANEL_GATES["system-health"] == SYSTEM_HEALTH_PANEL_ID
        for item_type, keys in expected.items():
            entry = _catalog_entry(source, item_type)
            predicate = entry[entry.index("available:") :]
            assert set(re.findall(r"ctx\.(\w+)", predicate)) == keys


class TestAssetsGoThroughThePrefix:
    """The new stylesheet is a ``<link href>`` — the import map never sees it."""

    def test_bars_css_is_prefixed(self, workspace_dir):
        cfg, panels, launch, env = _build_app(workspace_dir, env={"OSPREY_TERMINAL_USER": "alice"})
        with cfg, panels, launch, env:
            app = create_app(shell_command="echo")
            with TestClient(app) as client:
                body = _body(client)
        assert 'href="/u/alice/static/css/bars.css"' in body
        assert 'href="/static/css/bars.css"' not in body


class TestTheDefaultIsRenderableByConstruction:
    """The rev-0 answer never names an item the stamp says is unavailable.

    The shipped default places ``identity`` and ``system-health``, both gated.
    ``bar_render_plan`` always dropped them on paint, but the document behind
    the paint — the one ``GET /api/bar-items`` answers at ``rev`` 0 and the one
    a reset returns to — still carried them, and the browser's normalizer read
    that as lost content and latched Customize read-only on every deployment
    without the SYSTEM panel. The default is therefore filtered once, at the
    lifespan, by the same context the page stamps, so the served document and
    the paint agree.
    """

    def test_the_rev_0_answer_names_no_item_the_stamp_says_is_unavailable(self, plain_app):
        _, client = plain_app
        context = _context(_body(client))
        layout = client.get("/api/bar-items").json()

        assert layout["rev"] == 0
        for host in ("header", "status"):
            for item in layout[host]:
                assert bar_item_available(item["type"], context), (host, item["type"])

    def test_the_plain_default_degrades_exactly_as_its_docstring_promises(self, plain_app):
        """No SYSTEM panel and no identity: ``space · clock`` and a header
        without the identity block — the served document, not just the paint."""
        _, client = plain_app
        layout = client.get("/api/bar-items").json()

        assert [item["type"] for item in layout["status"]] == ["space", "clock"]
        assert "identity" not in [item["type"] for item in layout["header"]]

    def test_a_configured_deployment_keeps_the_gated_items(self, configured_app):
        _, client = configured_app
        layout = client.get("/api/bar-items").json()

        assert [item["type"] for item in layout["status"]] == ["space", "system-health", "clock"]
        assert "identity" in [item["type"] for item in layout["header"]]

    def test_the_paint_drops_nothing_further_from_the_served_default(self, plain_app):
        """The render plan and the document describe the same bars: every item
        the deployment default holds is a shell on the page."""
        app, client = plain_app
        body = _body(client)
        plan = bar_render_plan(app.state.bar_layout, context=_context(body))

        for host in ("header", "status"):
            assert [shell["type"] for shell in getattr(plan, host)] == [
                item["type"] for item in app.state.bar_layout[host]
            ]
            assert _shell_types(body, host) == [item["type"] for item in app.state.bar_layout[host]]
