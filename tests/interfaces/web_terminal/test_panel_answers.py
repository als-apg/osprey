"""How a panel's static files are answered: validated, compressed, read once.

Three contracts, each through the whole app:

* a page the proxy rewrites names the hub's shared style sheets, images and
  fonts at the fingerprinted address the hub's own pages use, so the browser
  never asks for them again; the design system's modules stay panel-scoped;
* the hub's own files under a panel are stored and revalidated, sent gzipped,
  and read from disk once per content;
* a backend's static file gets a validator of the bytes the browser receives
  and a ``304`` on a match, while a page or an API answer stays unstored.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from osprey.interfaces._assets import VERSION_SEGMENT, AssetSet
from osprey.interfaces.web_terminal import panel_answers
from osprey.interfaces.web_terminal.routes.proxy import _rewrite_content
from tests.interfaces.web_terminal._proxy_fakes import panel_app

DESIGN_SYSTEM_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
)

NEVER_STORED = "no-cache, no-store, must-revalidate"


@pytest.fixture(autouse=True)
def _nothing_kept():
    """The kept bodies are process-wide; each test starts and ends without any."""
    panel_answers._FILES.clear()
    panel_answers._PACKED.clear()
    yield
    panel_answers._FILES.clear()
    panel_answers._PACKED.clear()


@pytest.fixture
def app_and_client(workspace_dir):
    custom = [{"id": "my-dash", "label": "DASH", "url": "http://localhost:9"}]
    yield from panel_app(workspace_dir, custom)


def _backend(app, *, body, content_type, headers=None, status_code=200):
    """Make the panel backend answer every request with one response; return the mock."""

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        return httpx.Response(
            status_code=status_code,
            content=body,
            headers={"content-type": content_type, **(extra or {})},
        )

    extra = headers
    mock = AsyncMock(side_effect=fake_request)
    app.state.proxy_client.request = mock
    return mock


# ---------------------------------------------------------------------------
# The rewrite
# ---------------------------------------------------------------------------


@pytest.fixture
def shared(tmp_path) -> AssetSet:
    design = tmp_path / "design"
    (design / "css").mkdir(parents=True)
    (design / "css" / "tokens.css").write_text(":root{}", encoding="utf-8")
    (design / "img").mkdir()
    (design / "img" / "mark.svg").write_text("<svg/>", encoding="utf-8")
    (design / "js").mkdir()
    (design / "js" / "theme.js").write_text("export {}", encoding="utf-8")
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    (fonts / "fonts.css").write_text("@font-face{}", encoding="utf-8")
    return AssetSet({"/static/fonts": fonts, "/design-system": design})


class TestSharedFilesAreNamedAtTheShellAddress:
    def test_a_shared_sheet_image_and_font_get_the_fingerprinted_address(self, shared):
        fingerprint = shared.version()
        body = (
            '<link href="/design-system/css/tokens.css">'
            '<img src="/design-system/img/mark.svg">'
            '<link href="/static/fonts/fonts.css">'
        )

        result = _rewrite_content(body, "dash", "/u/op", shared=shared)

        assert f'"/u/op/design-system/{VERSION_SEGMENT}/{fingerprint}/css/tokens.css"' in result
        assert f'"/u/op/design-system/{VERSION_SEGMENT}/{fingerprint}/img/mark.svg"' in result
        assert f'"/u/op/static/fonts/{VERSION_SEGMENT}/{fingerprint}/fonts.css"' in result
        assert "/panel/dash/" not in result

    def test_a_query_string_stays_behind_the_fingerprinted_address(self, shared):
        result = _rewrite_content('"/design-system/css/tokens.css?v=1"', "dash", "", shared=shared)

        assert result == (
            f'"/design-system/{VERSION_SEGMENT}/{shared.version()}/css/tokens.css?v=1"'
        )

    def test_a_design_system_module_stays_panel_scoped(self, shared):
        """Modules import each other root-absolute; only the panel copy resolves that."""
        result = _rewrite_content('import "/design-system/js/theme.js"', "dash", "", shared=shared)

        assert result == 'import "/panel/dash/design-system/js/theme.js"'

    @pytest.mark.parametrize(
        "literal",
        [
            "/design-system/css/absent.css",
            "/design-system/css/",
            "/design-system/css/../js/theme.js",
            "/static/fonts/absent.ttf",
        ],
    )
    def test_a_name_that_is_no_file_of_the_set_stays_panel_scoped(self, shared, literal):
        result = _rewrite_content(f'"{literal}"', "dash", "", shared=shared)

        assert result == f'"/panel/dash{literal}"'

    def test_without_an_asset_set_every_literal_is_panel_scoped(self):
        result = _rewrite_content('"/design-system/css/tokens.css"', "dash", "")

        assert result == '"/panel/dash/design-system/css/tokens.css"'

    def test_a_rewritten_address_is_answered_immutable_by_the_app(self, app_and_client):
        """The address the rewrite emits is one the shell really serves, for a year."""
        app, client = app_and_client
        _backend(app, body=b'<link href="/design-system/css/tokens.css">', content_type="text/html")

        page = client.get("/panel/my-dash/")

        fingerprint = app.state.assets.version()
        address = f"/design-system/{VERSION_SEGMENT}/{fingerprint}/css/tokens.css"
        assert f'href="{address}"' in page.text
        sheet = client.get(address)
        assert sheet.status_code == 200
        assert "immutable" in sheet.headers["cache-control"]
        assert sheet.content == (DESIGN_SYSTEM_DIR / "css" / "tokens.css").read_bytes()


class TestUrlOfFile:
    def test_names_a_file_of_the_set(self, shared):
        assert shared.url_of_file("/design-system/css/tokens.css") == (
            f"/design-system/{VERSION_SEGMENT}/{shared.version()}/css/tokens.css"
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/design-system/css/absent.css",
            "/design-system/css",
            "/design-system/",
            "/design-system/css/../css/tokens.css",
            f"/design-system/{VERSION_SEGMENT}/x/css/tokens.css",
            "/elsewhere/tokens.css",
        ],
    )
    def test_names_nothing_else(self, shared, path):
        assert shared.url_of_file(path) is None


# ---------------------------------------------------------------------------
# The hub's own files under a panel
# ---------------------------------------------------------------------------

HUB_MODULE = "/panel/my-dash/design-system/js/clipboard.js"
HUB_SHEET = "/panel/my-dash/design-system/css/tokens.css"


class TestHubFilesUnderAPanel:
    def test_stored_and_revalidated_never_unstorable(self, app_and_client):
        _, client = app_and_client

        resp = client.get(HUB_SHEET)

        assert resp.headers["cache-control"] == "no-cache"
        assert "no-store" not in resp.headers["cache-control"]
        assert resp.headers["etag"].startswith('"')

    def test_a_matching_validator_is_answered_304_without_a_body(self, app_and_client):
        _, client = app_and_client
        first = client.get(HUB_SHEET)

        again = client.get(HUB_SHEET, headers={"If-None-Match": first.headers["etag"]})

        assert again.status_code == 304
        assert again.content == b""
        assert again.headers["etag"] == first.headers["etag"]
        assert again.headers["cache-control"] == "no-cache"

    def test_sent_gzipped_to_a_browser_that_accepts_it(self, app_and_client):
        _, client = app_and_client
        on_disk = (DESIGN_SYSTEM_DIR / "css" / "tokens.css").read_bytes()

        with client.stream("GET", HUB_SHEET, headers={"Accept-Encoding": "gzip"}) as resp:
            raw = b"".join(resp.iter_raw())

        assert resp.headers["content-encoding"] == "gzip"
        assert "accept-encoding" in resp.headers["vary"].lower()
        assert len(raw) < len(on_disk) / 2
        assert gzip.decompress(raw) == on_disk

    def test_sent_plain_to_a_browser_that_does_not(self, app_and_client):
        _, client = app_and_client

        resp = client.get(HUB_SHEET, headers={"Accept-Encoding": "identity"})

        assert "content-encoding" not in resp.headers
        assert resp.content == (DESIGN_SYSTEM_DIR / "css" / "tokens.css").read_bytes()

    def test_the_gzipped_and_the_plain_form_carry_different_validators(self, app_and_client):
        _, client = app_and_client

        plain = client.get(HUB_SHEET, headers={"Accept-Encoding": "identity"})
        packed = client.get(HUB_SHEET, headers={"Accept-Encoding": "gzip"})

        assert plain.headers["etag"] != packed.headers["etag"]
        for tag in (plain.headers["etag"], packed.headers["etag"]):
            assert client.get(HUB_SHEET, headers={"If-None-Match": tag}).status_code == 304

    def test_a_file_is_read_once_however_often_it_is_asked_for(self, app_and_client, monkeypatch):
        _, client = app_and_client
        reads = []
        read_file = panel_answers._read_file

        def counting(path, *args):
            reads.append(path)
            return read_file(path, *args)

        monkeypatch.setattr(panel_answers, "_read_file", counting)

        bodies = {client.get(HUB_MODULE).content for _ in range(5)}

        assert len(bodies) == 1
        assert len(reads) == 1

    def test_a_module_is_still_rewritten_into_the_panel(self, app_and_client):
        _, client = app_and_client

        resp = client.get("/panel/my-dash/design-system/js/channel-combobox.js")

        assert resp.status_code == 200
        assert "'/design-system/js/" not in resp.text
        assert '"/design-system/js/' not in resp.text


class TestHubFileCache:
    async def test_a_changed_file_is_read_again(self, tmp_path):
        path = tmp_path / "a.css"
        path.write_text("a{}", encoding="utf-8")
        ask = httpx.Headers({})

        first = await panel_answers.serve_hub_file(
            ask, "GET", path, media_type="text/css", headers={}
        )
        path.write_text("b{color:red}", encoding="utf-8")
        second = await panel_answers.serve_hub_file(
            ask, "GET", path, media_type="text/css", headers={}
        )

        assert first.body == b"a{}"
        assert second.body == b"b{color:red}"
        assert first.headers["etag"] != second.headers["etag"]

    async def test_one_file_under_two_variants_is_kept_apart(self, tmp_path):
        path = tmp_path / "a.js"
        path.write_text("x", encoding="utf-8")
        ask = httpx.Headers({})

        answers = [
            await panel_answers.serve_hub_file(
                ask,
                "GET",
                path,
                media_type="text/javascript",
                headers={},
                variant=variant,
                transform=lambda body, variant=variant: body + variant.encode(),
            )
            for variant in ("one", "two", "one")
        ]

        assert [answer.body for answer in answers] == [b"xone", b"xtwo", b"xone"]

    async def test_a_missing_file_is_404(self, tmp_path):
        answer = await panel_answers.serve_hub_file(
            httpx.Headers({}), "GET", tmp_path / "gone.css", media_type="text/css", headers={}
        )

        assert answer.status_code == 404

    def test_the_least_recently_used_body_is_dropped_beyond_the_bound(self):
        keep = panel_answers._Keep(limit=10)
        keep.put("a", panel_answers.Kept(None, b"1234", '"a"'))
        keep.put("b", panel_answers.Kept(None, b"1234", '"b"'))
        assert keep.get("a") is not None  # now the most recently used

        keep.put("c", panel_answers.Kept(None, b"1234", '"c"'))
        keep.put("huge", panel_answers.Kept(None, b"x" * 11, '"huge"'))

        assert keep.get("b") is None
        assert keep.get("a") is not None
        assert keep.get("c") is not None
        assert keep.get("huge") is None


# ---------------------------------------------------------------------------
# A backend's files
# ---------------------------------------------------------------------------

BIG_SCRIPT = b"const a = '/api/state';\n" * 200


class TestRelayedStaticFiles:
    def test_a_rewritten_script_is_validated_by_the_bytes_sent(self, app_and_client):
        """The backend's tag names the body before the rewrite, so it is replaced."""
        app, client = app_and_client
        _backend(
            app,
            body=BIG_SCRIPT,
            content_type="text/javascript",
            headers={"etag": '"backend"', "last-modified": "Wed, 21 Oct 2015 07:28:00 GMT"},
        )

        resp = client.get("/panel/my-dash/app.js", headers={"Accept-Encoding": "identity"})

        assert "/panel/my-dash/api/state" in resp.text
        assert panel_answers.is_hub_tag(resp.headers["etag"])
        assert "last-modified" not in resp.headers
        assert resp.headers["cache-control"] == "no-cache"

    def test_a_matching_validator_is_answered_304(self, app_and_client):
        app, client = app_and_client
        _backend(app, body=BIG_SCRIPT, content_type="text/javascript")
        first = client.get("/panel/my-dash/app.js")

        again = client.get(
            "/panel/my-dash/app.js", headers={"If-None-Match": first.headers["etag"]}
        )

        assert again.status_code == 304
        assert again.content == b""
        assert again.headers["cache-control"] == "no-cache"

    def test_a_changed_backend_file_is_sent_again(self, app_and_client):
        app, client = app_and_client
        _backend(app, body=BIG_SCRIPT, content_type="text/javascript")
        first = client.get("/panel/my-dash/app.js")
        _backend(app, body=BIG_SCRIPT + b"// new\n", content_type="text/javascript")

        again = client.get(
            "/panel/my-dash/app.js", headers={"If-None-Match": first.headers["etag"]}
        )

        assert again.status_code == 200
        assert again.text.endswith("// new\n")

    def test_the_hubs_validator_does_not_reach_the_backend(self, app_and_client):
        """The backend never issued it; asked conditionally it could only answer wrongly."""
        app, client = app_and_client
        backend = _backend(app, body=BIG_SCRIPT, content_type="text/javascript")
        first = client.get("/panel/my-dash/app.js")

        client.get(
            "/panel/my-dash/app.js",
            headers={
                "If-None-Match": first.headers["etag"],
                "If-Modified-Since": "Wed, 21 Oct 2015 07:28:00 GMT",
            },
        )

        sent = {name.lower() for name in backend.call_args.kwargs["headers"]}
        assert "if-none-match" not in sent
        assert "if-modified-since" not in sent

    def test_a_backends_validator_reaches_it_and_its_304_comes_back(self, app_and_client):
        app, client = app_and_client
        backend = _backend(
            app, body=b"", content_type="image/png", headers={"etag": '"backend"'}, status_code=304
        )

        resp = client.get("/panel/my-dash/logo.png", headers={"If-None-Match": '"backend"'})

        sent = {k.lower(): v for k, v in backend.call_args.kwargs["headers"].items()}
        assert sent["if-none-match"] == '"backend"'
        assert resp.status_code == 304
        assert resp.headers["etag"] == '"backend"'
        assert "no-store" not in resp.headers["cache-control"]

    def test_an_untouched_file_keeps_the_backends_validator(self, app_and_client):
        app, client = app_and_client
        _backend(
            app,
            body=b"\x89PNG" + b"\x00" * 64,
            content_type="image/png",
            headers={"etag": '"backend"', "last-modified": "Wed, 21 Oct 2015 07:28:00 GMT"},
        )

        resp = client.get("/panel/my-dash/logo.png")

        assert resp.headers["etag"] == '"backend"'
        assert resp.headers["last-modified"] == "Wed, 21 Oct 2015 07:28:00 GMT"
        assert resp.headers["cache-control"] == "no-cache"
        assert "content-encoding" not in resp.headers

    def test_a_backends_own_cache_control_is_kept(self, app_and_client):
        app, client = app_and_client
        _backend(
            app,
            body=BIG_SCRIPT,
            content_type="text/javascript",
            headers={"cache-control": "public, max-age=60"},
        )

        resp = client.get("/panel/my-dash/app.js")

        assert resp.headers["cache-control"] == "public, max-age=60"

    def test_a_large_text_answer_leaves_gzipped(self, app_and_client):
        app, client = app_and_client
        _backend(app, body=BIG_SCRIPT, content_type="text/javascript")

        with client.stream(
            "GET", "/panel/my-dash/app.js", headers={"Accept-Encoding": "gzip"}
        ) as resp:
            raw = b"".join(resp.iter_raw())

        assert resp.headers["content-encoding"] == "gzip"
        assert "accept-encoding" in resp.headers["vary"].lower()
        assert len(raw) < len(BIG_SCRIPT) / 4
        assert b"/panel/my-dash/api/state" in gzip.decompress(raw)

    def test_a_small_answer_is_left_alone(self, app_and_client):
        app, client = app_and_client
        _backend(app, body=b"a{}", content_type="text/css")

        resp = client.get("/panel/my-dash/a.css", headers={"Accept-Encoding": "gzip"})

        assert "content-encoding" not in resp.headers
        assert resp.content == b"a{}"


class TestRelayedPagesAndApiAnswers:
    @pytest.mark.parametrize("content_type", ["text/html", "application/json"])
    def test_never_stored_and_not_validated(self, app_and_client, content_type):
        app, client = app_and_client
        _backend(app, body=b'{"a": "/api/x"}', content_type=content_type)

        resp = client.get("/panel/my-dash/api/state")

        assert resp.headers["cache-control"] == NEVER_STORED
        assert "etag" not in resp.headers

    def test_a_large_api_answer_leaves_gzipped_and_still_unstored(self, app_and_client):
        app, client = app_and_client
        body = b'{"rows": [' + b'"value",' * 500 + b'"end"]}'
        _backend(app, body=body, content_type="application/json")

        with client.stream(
            "GET", "/panel/my-dash/api/rows", headers={"Accept-Encoding": "gzip"}
        ) as resp:
            raw = b"".join(resp.iter_raw())

        assert resp.headers["content-encoding"] == "gzip"
        assert resp.headers["cache-control"] == NEVER_STORED
        assert gzip.decompress(raw) == body

    def test_a_post_answer_is_not_validated(self, app_and_client):
        app, client = app_and_client
        _backend(app, body=BIG_SCRIPT, content_type="text/javascript")

        resp = client.post("/panel/my-dash/app.js")

        assert "etag" not in resp.headers
        assert resp.headers["cache-control"] == NEVER_STORED


# ---------------------------------------------------------------------------
# A backend cannot pass for the hub
# ---------------------------------------------------------------------------


async def _relayed(body: bytes, *, etag: str | None = None, content_type="text/css", gzip_ok=True):
    """One untouched backend answer through ``shape_relayed``, as a browser gets it."""
    from starlette.datastructures import Headers

    return await panel_answers.shape_relayed(
        Headers({"accept-encoding": "gzip" if gzip_ok else "identity"}),
        "GET",
        body=body,
        status_code=200,
        headers={"ETag": etag} if etag is not None else {},
        media_type=content_type,
        rewritten=False,
        default_cache_control=NEVER_STORED,
    )


def _sent(response) -> bytes:
    body = bytes(response.body)
    return gzip.decompress(body) if response.headers.get("content-encoding") == "gzip" else body


FIRST_SHEET = b".first { color: red; }\n" * 200
OTHER_SHEET = b".other { color: blue; }\n" * 200


class TestABackendCannotPassForTheHub:
    @pytest.mark.parametrize(
        "spelling",
        ['"h-{d}"', 'W/"h-{d}"', 'w/"h-{d}"', ' W/ "h-{d}"', '"h-{d}-gz"', '"h-anything"'],
    )
    async def test_a_backend_tag_spelled_as_the_hubs_is_not_relayed(self, spelling):
        # The tag the hub would give FIRST_SHEET, claimed for another body.
        claimed = spelling.format(d=panel_answers._tag(FIRST_SHEET)[3:-1])

        resp = await _relayed(OTHER_SHEET, etag=claimed)

        assert resp.headers["etag"] == panel_answers._compressed_tag(
            panel_answers._tag(OTHER_SHEET), ours=True
        )
        assert _sent(resp) == OTHER_SHEET

    async def test_a_claimed_hub_tag_is_never_served_the_body_that_tag_names(self):
        first = await _relayed(FIRST_SHEET)  # the hub tags it and keeps its gzipped copy
        assert _sent(first) == FIRST_SHEET
        plain_tag = panel_answers._tag(FIRST_SHEET)

        for claimed in (plain_tag, first.headers["etag"], f"W/{plain_tag}"):
            resp = await _relayed(OTHER_SHEET, etag=claimed)
            assert _sent(resp) == OTHER_SHEET, claimed

    async def test_a_claimed_hub_tag_plants_nothing_for_the_body_that_tag_names(self):
        claimed = panel_answers._tag(FIRST_SHEET)
        planted = await _relayed(OTHER_SHEET, etag=claimed)
        assert _sent(planted) == OTHER_SHEET

        resp = await _relayed(FIRST_SHEET)

        assert _sent(resp) == FIRST_SHEET

    async def test_a_claimed_hub_tag_is_dropped_from_an_answer_that_is_not_validated(self):
        resp = await _relayed(b"{}" * 2000, etag='"h-anything"', content_type="application/json")

        assert "etag" not in resp.headers
        assert _sent(resp) == b"{}" * 2000

    async def test_a_backends_own_tag_keeps_nothing(self):
        resp = await _relayed(FIRST_SHEET, etag='"backend-1"')

        assert resp.headers["etag"] == 'W/"backend-1"'
        assert not panel_answers._PACKED._held

    async def test_one_backend_tag_on_two_bodies_sends_each_its_own(self):
        first = await _relayed(FIRST_SHEET, etag='"same"')
        other = await _relayed(OTHER_SHEET, etag='"same"')

        assert (_sent(first), _sent(other)) == (FIRST_SHEET, OTHER_SHEET)

    async def test_two_bodies_never_share_a_kept_copy(self):
        await _relayed(FIRST_SHEET)
        await _relayed(OTHER_SHEET)
        await _relayed(FIRST_SHEET, content_type="text/javascript")

        held = panel_answers._PACKED._held
        assert set(held) == {
            (panel_answers._digest(FIRST_SHEET), len(FIRST_SHEET)),
            (panel_answers._digest(OTHER_SHEET), len(OTHER_SHEET)),
        }
        for (digest, _length), kept in held.items():
            assert panel_answers._digest(gzip.decompress(kept.packed)) == digest

    async def test_bodies_that_agree_in_the_short_tag_are_still_kept_apart(self, monkeypatch):
        """The key is the whole digest and the length, not the tag sent to the browser."""
        monkeypatch.setattr(panel_answers, "_tag", lambda _body, _digest=None: '"h-collides"')

        first = await _relayed(FIRST_SHEET)
        other = await _relayed(OTHER_SHEET)

        assert first.headers["etag"] == other.headers["etag"]
        assert (_sent(first), _sent(other)) == (FIRST_SHEET, OTHER_SHEET)

    def test_through_the_proxy_a_claimed_hub_tag_is_replaced(self, app_and_client):
        app, client = app_and_client
        claimed = panel_answers._tag(FIRST_SHEET)
        _backend(app, body=OTHER_SHEET, content_type="image/svg+xml", headers={"etag": claimed})

        resp = client.get("/panel/my-dash/mark.svg")

        assert resp.content == OTHER_SHEET
        assert resp.headers["etag"].strip('"').removesuffix("-gz") == (
            panel_answers._tag(OTHER_SHEET).strip('"')
        )

    def test_is_hub_tag_reads_weak_forms_in_any_case(self):
        assert panel_answers.is_hub_tag('w/"h-abc"')
        assert panel_answers.is_hub_tag('"other", W/ "h-abc-gz"')
        assert not panel_answers.is_hub_tag('"backend-h-abc"')
