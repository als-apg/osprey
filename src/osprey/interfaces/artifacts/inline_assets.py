"""Self-contained pages for the artifact gallery's sandboxed content routes.

Every content route answers with ``Content-Security-Policy: sandbox
allow-scripts allow-downloads`` (see :mod:`.content_sandbox`), so the page runs
in an opaque origin. Such a page sends no cookie, and on a multi-user
deployment every ``/u/<user>/`` asset sits behind the proxy's ``auth_request``:
a ``/static`` or ``/design-system`` subresource answers 401 inside the
sandbox. The pages the gallery builds itself therefore carry what they need
inline:

* design-system scripts and stylesheets (``theme-boot.js``, ``tokens.css``)
  become inline ``<script>`` / ``<style>`` blocks;
* ES modules (``theme-manager.js``, the renderer ``md-render.js``) become
  ``data:`` URL modules whose own imports are rewritten the same way, so no
  module graph reaches back to the server;
* vendor libraries (marked, KaTeX, highlight.js, DOMPurify, Plotly) are read
  from the vendor directory ``osprey vendor fetch`` fills when the deployment
  is offline, and keep their CDN URLs otherwise;
* KaTeX's stylesheet is inlined with its fonts as ``data:`` URIs.

Nothing here touches agent-authored markup beyond the Plotly bundle: the
rewrite of Plotly ``<script src>`` tags (CDN in offline mode, a local
``/static`` path in either mode) to one inline copy per page.
"""

from __future__ import annotations

import base64
import json
import logging
import posixpath
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: ``osprey/interfaces`` — the base every vendor-manifest target is relative to.
INTERFACES_DIR = Path(__file__).resolve().parent.parent

#: Where ``osprey vendor fetch`` writes the manifest's targets (its default
#: ``base_dir``). Tests point this at a fixture vendor tree.
VENDOR_ROOT: Path = INTERFACES_DIR

ARTIFACTS_STATIC = Path(__file__).resolve().parent / "static"
DESIGN_SYSTEM_STATIC = INTERFACES_DIR / "design_system" / "static"

#: URL prefixes of first-party modules and the directories they are served from.
_MODULE_ROOTS: tuple[tuple[str, Path], ...] = (
    ("/design-system/", DESIGN_SYSTEM_STATIC),
    ("/static/", ARTIFACTS_STATIC),
)

PLOTLY_ASSET = "Plotly.js"

# ---------------------------------------------------------------------------
# File reads
# ---------------------------------------------------------------------------

_read_cache: dict[Path, tuple[int, int, bytes]] = {}


def _read_bytes(path: Path) -> bytes:
    """Read a file, cached until its mtime or size changes (Plotly is 4.8 MB)."""
    st = path.stat()
    hit = _read_cache.get(path)
    if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    data = path.read_bytes()
    _read_cache[path] = (st.st_mtime_ns, st.st_size, data)
    return data


def _first_party_file(url_path: str) -> Path:
    """The on-disk file behind a first-party ``/design-system/`` or ``/static/`` URL."""
    for prefix, root in _MODULE_ROOTS:
        if url_path.startswith(prefix):
            candidate = (root / url_path[len(prefix) :]).resolve()
            if candidate.is_relative_to(root.resolve()) and candidate.is_file():
                return candidate
            break
    raise FileNotFoundError(f"no first-party asset behind {url_path!r}")


# ---------------------------------------------------------------------------
# Vendor assets
# ---------------------------------------------------------------------------


def _manifest() -> dict:
    from osprey.interfaces.vendor import MANIFEST_PATH

    manifest: dict = json.loads(_read_bytes(MANIFEST_PATH))
    return manifest


def _asset(name: str) -> dict:
    asset: dict
    for asset in _manifest().get("assets", []):
        if asset.get("name") == name:
            return asset
    raise KeyError(f"Unknown vendor asset name: {name!r}")


def vendor_file(name: str) -> Path | None:
    """The fetched local copy of manifest asset ``name``, from any of its targets.

    An asset may ship to several interfaces' vendor directories; any fetched
    copy is the same file, checksum-pinned by the manifest.
    """
    asset = _asset(name)
    for target in asset["targets"]:
        path: Path = VENDOR_ROOT / target / asset["filename"]
        if path.is_file():
            return path
    return None


def vendor_text(name: str) -> str | None:
    """The fetched text of manifest asset ``name``, or ``None`` when not fetched."""
    path = vendor_file(name)
    if path is None:
        logger.warning(
            "Vendor asset %r is not fetched (run `osprey vendor fetch`); "
            "sandboxed pages render without it",
            name,
        )
        return None
    return _read_bytes(path).decode("utf-8", errors="replace")


def cdn_url(name: str) -> str:
    return str(_asset(name)["url"])


# ---------------------------------------------------------------------------
# Inline blocks
# ---------------------------------------------------------------------------

_SCRIPT_CLOSE_RE = re.compile(r"</(script)", re.IGNORECASE)
_STYLE_CLOSE_RE = re.compile(r"</(style)", re.IGNORECASE)


def inline_script(js: str, *, module: bool = False) -> str:
    """``js`` as an inline ``<script>`` block that cannot close itself early."""
    body = _SCRIPT_CLOSE_RE.sub(r"<\\/\1", js).replace("<!--", "<\\!--")
    kind = ' type="module"' if module else ""
    return f"<script{kind}>{body}</script>"


def inline_style(css: str) -> str:
    body = _STYLE_CLOSE_RE.sub(r"<\\/\1", css)
    return f"<style>{body}</style>"


def data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def vendor_script(name: str, *, offline: bool) -> str:
    """A ``<script>`` for vendor asset ``name``: CDN ``src`` online, inline offline.

    Offline with the asset not fetched, nothing is emitted (a credentialed
    ``/static`` path would only answer 401 inside the sandbox); the renderers
    feature-guard every library and degrade to plain text.
    """
    if not offline:
        return f'<script src="{cdn_url(name)}"></script>'
    text = vendor_text(name)
    return inline_script(text) if text is not None else ""


def vendor_stylesheet_href(name: str, *, offline: bool) -> str:
    """An ``href`` for vendor stylesheet ``name``: its CDN URL, or a ``data:`` URL offline."""
    if not offline:
        return cdn_url(name)
    text = vendor_text(name)
    return data_url("text/css", (text or "").encode("utf-8"))


# ---------------------------------------------------------------------------
# KaTeX stylesheet with inlined fonts
# ---------------------------------------------------------------------------

_FONT_SRC_RE = re.compile(r"src:([^;}]+)")
_FONT_URL_RE = re.compile(r"url\(\s*['\"]?([^)'\"]+)['\"]?\s*\)")


def _katex_fonts_dir() -> Path | None:
    for bundle in _manifest().get("font_bundles", []):
        if "katex" in bundle.get("base_url", "").lower():
            fonts_dir: Path = VENDOR_ROOT / bundle["target"]
            return fonts_dir
    return None


def _inline_font_src(match: re.Match[str], fonts_dir: Path | None) -> str:
    """Keep only the ``woff2`` sources of one ``src:`` list, as ``data:`` URIs."""
    kept: list[str] = []
    for entry in match.group(1).split(","):
        url = _FONT_URL_RE.search(entry)
        if url is None or not url.group(1).endswith(".woff2") or fonts_dir is None:
            continue
        font = fonts_dir / posixpath.basename(url.group(1))
        if not font.is_file():
            continue
        uri = data_url("font/woff2", _read_bytes(font))
        kept.append(_FONT_URL_RE.sub(lambda _m, u=uri: f"url({u})", entry.strip(), count=1))
    return "src:" + ",".join(kept) if kept else "src:local('')"


def katex_stylesheet(*, offline: bool) -> str:
    """KaTeX's stylesheet: a CDN ``<link>`` online, an inline ``<style>`` offline.

    Offline, every ``@font-face`` keeps only its ``woff2`` source, inlined as a
    ``data:`` URI, so the math fonts load inside the sandbox; the ``woff`` and
    ``ttf`` fallbacks (relative URLs that would reach the server) are dropped.
    """
    if not offline:
        return f'<link rel="stylesheet" href="{cdn_url("KaTeX CSS")}">'
    css = vendor_text("KaTeX CSS")
    if css is None:
        return ""
    fonts_dir = _katex_fonts_dir()
    return inline_style(_FONT_SRC_RE.sub(lambda m: _inline_font_src(m, fonts_dir), css))


# ---------------------------------------------------------------------------
# First-party modules as data: URLs
# ---------------------------------------------------------------------------

# Static import / re-export specifiers: `import x from '...'`, `import '...'`,
# `export { a } from '...'`. Only root-relative and relative specifiers are
# first-party; bare and absolute-URL specifiers are left alone.
_IMPORT_SPEC_RE = re.compile(r"""(\bfrom\s*|\bimport\s*)(["'])(\.{0,2}/[^"'\n]*)\2""")


def _rewrite_imports(source: str, base_url: str | None, stack: tuple[str, ...]) -> str:
    def replace(m: re.Match[str]) -> str:
        spec = m.group(3)
        if spec.startswith("/"):
            target = spec
        elif base_url is not None:
            target = posixpath.normpath(posixpath.join(posixpath.dirname(base_url), spec))
        else:
            return m.group(0)
        return f"{m.group(1)}{m.group(2)}{module_data_url(target, _stack=stack)}{m.group(2)}"

    return _IMPORT_SPEC_RE.sub(replace, source)


def module_data_url(url_path: str, *, _stack: tuple[str, ...] = ()) -> str:
    """A ``data:`` URL holding first-party module ``url_path`` and, recursively, its imports.

    A ``data:`` module cannot resolve a relative or root-relative specifier
    against the server, so every first-party import inside it is itself
    replaced by a ``data:`` URL. Raises ``FileNotFoundError`` for an unknown
    path and ``ValueError`` on an import cycle.
    """
    if url_path in _stack:
        raise ValueError(f"import cycle through {url_path!r}")
    source = _read_bytes(_first_party_file(url_path)).decode("utf-8")
    rewritten = _rewrite_imports(source, url_path, (*_stack, url_path))
    return data_url("text/javascript", rewritten.encode("utf-8"))


# ---------------------------------------------------------------------------
# First-party page snippets
# ---------------------------------------------------------------------------

_DS_SCRIPT_RE = re.compile(
    r"""<script\s+src=["'](/design-system/[^"']+\.js)["']\s*>\s*</script>""", re.IGNORECASE
)
_DS_STYLESHEET_RE = re.compile(
    r"""<link\s+rel=["']stylesheet["']\s+href=["'](/design-system/[^"']+\.css)["']\s*/?>""",
    re.IGNORECASE,
)
_MODULE_BLOCK_RE = re.compile(
    r"""(<script\s+type=["']module["']\s*>)(.*?)(</script>)""", re.IGNORECASE | re.DOTALL
)


def self_contained(snippet: str) -> str:
    """Make one of the gallery's own HTML snippets reference no server path.

    Design-system classic scripts and stylesheets are inlined, and every
    first-party import inside an inline module script becomes a ``data:``
    URL module. Only for markup the gallery itself writes -- never applied to
    agent-authored HTML.
    """
    snippet = _DS_SCRIPT_RE.sub(
        lambda m: inline_script(_read_bytes(_first_party_file(m.group(1))).decode("utf-8")),
        snippet,
    )
    snippet = _DS_STYLESHEET_RE.sub(
        lambda m: inline_style(_read_bytes(_first_party_file(m.group(1))).decode("utf-8")),
        snippet,
    )
    return _MODULE_BLOCK_RE.sub(
        lambda m: m.group(1) + _rewrite_imports(m.group(2), None, ()) + m.group(3), snippet
    )


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

# A whole external Plotly <script> tag: the public CDN (any version), or a
# first-party `/static/...plotly...js` path.
_CDN_PLOTLY_TAG_RE = re.compile(
    r"""<script\b[^>]*?\bsrc\s*=\s*["']https://cdn\.plot\.ly/plotly[^"']*\.js["'][^>]*>\s*</script\s*>""",
    re.IGNORECASE,
)
_LOCAL_PLOTLY_TAG_RE = re.compile(
    r"""<script\b[^>]*?\bsrc\s*=\s*["'](?:[^"':]*/)?static/[^"']*plotly[^"']*\.js["'][^>]*>\s*</script\s*>""",
    re.IGNORECASE,
)
_PLOTLY_BANNER = "plotly.js v"


def plotly_script(*, offline: bool) -> str:
    """The one Plotly bundle for a full-view page: CDN ``src`` online, inline offline."""
    if not offline:
        return f'<script src="{cdn_url(PLOTLY_ASSET)}"></script>'
    text = vendor_text(PLOTLY_ASSET)
    return inline_script(text) if text is not None else ""


def self_contain_plotly(html: str, *, offline: bool) -> str:
    """Rewrite external Plotly tags to at most one inline copy per page.

    Offline, every CDN tag (any version, SRI attributes included) is replaced;
    in either mode, so is every first-party ``/static`` Plotly tag, which the
    sandbox could not load. The first replaced tag becomes the inline bundle
    -- unless the page already inlines one -- and the rest are dropped,
    whatever their count or version.
    """
    patterns = [_LOCAL_PLOTLY_TAG_RE] + ([_CDN_PLOTLY_TAG_RE] if offline else [])
    combined = re.compile("|".join(p.pattern for p in patterns), re.IGNORECASE)
    pending = [_PLOTLY_BANNER not in html]  # one inline copy, unless the page has its own

    def replace(_m: re.Match[str]) -> str:
        if pending[0]:
            pending[0] = False
            return _plotly_inline_or_cdn(offline)
        return ""

    return combined.sub(replace, html)


def _plotly_inline_or_cdn(offline: bool) -> str:
    text = vendor_text(PLOTLY_ASSET)
    if text is not None:
        return inline_script(text)
    # Nothing fetched to inline: online, the CDN still serves it.
    return "" if offline else f'<script src="{cdn_url(PLOTLY_ASSET)}"></script>'
