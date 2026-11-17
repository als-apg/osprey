"""Which login shows which panel: the answer ``osprey profile panels`` prints.

A login's tab strip is decided in several places: the ``web_panels`` list of
the deployment and of the login's own file, ``web.panels.<id>.hidden`` under
either ``config:``, ``default_panel``, and the ``apps:`` entries that show an
application's panel or keep it off the sidebar. Each is read where the build
reads it, from the resolved profile of every login, so the table is what the
renders will carry and not a second reading of the files.

:func:`panel_matrix` is the derivation, :func:`format_panel_matrix` the plain
table, :func:`panel_matrix_html` the same table as one self-contained page and
:func:`panel_matrix_bundle` that page as the ``roster`` panel of the web terminal.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from osprey.profiles.web_panels import (
    BUILD_PANELS,
    ROSTER_PANEL_ID,
    UNIVERSAL_PANELS,
    in_catalog_order,
    panel_label,
)

__all__ = [
    "STATE_DEFAULT",
    "STATE_MENU",
    "STATE_OFF",
    "STATE_TAB",
    "format_panel_matrix",
    "panel_matrix",
    "panel_matrix_bundle",
    "panel_matrix_html",
]

#: The tab the login opens on.
STATE_DEFAULT = "default"
#: A tab on the sidebar.
STATE_TAB = "tab"
#: Selected but hidden: reachable from the panel menu, no tab until added.
STATE_MENU = "menu"
#: Not selected: the login has no such panel.
STATE_OFF = "off"

#: Where a fact with no authored line in the repo comes from.
_SOURCE_INHERITED = "inherited"
_SOURCE_ALWAYS = "always"

_MARKS = {STATE_DEFAULT: "DEFAULT", STATE_TAB: "tab", STATE_MENU: "menu", STATE_OFF: "-"}


def _raw_document(path: Path) -> Mapping[str, Any]:
    """The mapping *path* authors, or an empty one when it does not parse to one."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _authored_panel_leaf(document: Mapping[str, Any], panel_id: str, leaf: str) -> bool:
    """Whether *document*'s ``config:`` writes ``web.panels.<panel_id>.<leaf>``."""
    from .build_profile_panels import panel_leaf_spellings

    config = document.get("config")
    if not isinstance(config, Mapping):
        return False
    return bool(panel_leaf_spellings(config, panel_id, leaf))


def _first_author(
    documents: list[tuple[str, Mapping[str, Any]]], authored: Any, fallback: str
) -> str:
    """The first of *documents*, nearest the login first, for which *authored* holds."""
    for shown, document in documents:
        if authored(document):
            return shown
    return fallback


def _logins(config: Mapping[str, Any]) -> dict[str, list[str]]:
    """``render name -> who opens it``: roster people by tier, a shared card by its access."""
    from osprey.deployment.web_terminals.roster import (
        authored_people,
        effective_tier,
        shared_app_name,
    )

    from .build_profile_emit import effective_web_terminals

    web_terminals = effective_web_terminals(config)
    if not web_terminals.get("enabled"):
        return {}
    opened: dict[str, list[str]] = {}
    for entry in authored_people(web_terminals):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            continue
        tier = effective_tier(
            entry,
            web_terminals.get("authorization"),
            web_terminals.get("default_tier"),
            strict=False,
        )
        if tier:
            opened.setdefault(tier, []).append(entry["name"])
    shared = web_terminals.get("shared")
    for item in shared if isinstance(shared, list) else []:
        name = shared_app_name(item)
        if name is None:
            continue
        access = item.get("access") if isinstance(item, Mapping) else None
        who = [str(a) for a in access] if isinstance(access, list) and access else ["everyone"]
        opened.setdefault(name, []).extend(f"card: {w}" for w in who)
    return opened


def panel_matrix(repo_root: Path) -> dict[str, Any]:
    """Every login of the repo at *repo_root* against every panel any of them selects.

    Args:
        repo_root: The deployment repo: ``profile.yml`` and its layer files.

    Returns:
        ``{"panels": [{id, label}], "renders": [{name, file, logins, default_panel,
        panels: {id: {state, selected_in, hidden_in}}}]}``. The first render is
        the deployment's own profile; ``selected_in`` and ``hidden_in`` name the
        file whose line decides the cell, ``inherited`` when none in the repo does.

    Raises:
        BuildProfileError: A profile or layer that does not resolve.
    """
    from .build_profile import resolve_build_profile
    from .build_profile_emit import effective_config_subtree
    from .profile_root import PROFILE_FILENAME, layer_files

    root_path = repo_root / PROFILE_FILENAME
    root_document = (PROFILE_FILENAME, _raw_document(root_path))
    sources: list[tuple[str, Path, list[tuple[str, Mapping[str, Any]]]]] = [
        (PROFILE_FILENAME, root_path, [root_document])
    ]
    for _kind, path in layer_files(repo_root):
        shown = str(path.relative_to(repo_root))
        sources.append((shown, path, [(shown, _raw_document(path)), root_document]))

    root_profile, _ = resolve_build_profile(root_path, None)
    logins = _logins(root_profile.config)

    renders: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for shown, path, documents in sources:
        profile, profile_dir = resolve_build_profile(path, None)
        panels_config = effective_config_subtree(profile.config, ("web", "panels"))
        app_panels = profile.shown_app_panels(profile_dir)
        cells: dict[str, dict[str, str | None]] = {}

        for panel_id in list(dict.fromkeys(profile.web_panels)) + sorted(UNIVERSAL_PANELS):
            if panel_id in cells:
                continue
            spec = panels_config.get(panel_id)
            hidden = isinstance(spec, Mapping) and bool(spec.get("hidden", False))
            labels.setdefault(panel_id, panel_label(panel_id, panels_config))
            if panel_id in profile.web_panels:
                selected_in = _first_author(
                    documents,
                    lambda d, pid=panel_id: pid in (d.get("web_panels") or []),
                    _SOURCE_INHERITED,
                )
            else:
                selected_in = _SOURCE_ALWAYS
            cells[panel_id] = {
                "state": STATE_MENU if hidden else STATE_TAB,
                "selected_in": selected_in,
                "hidden_in": _first_author(
                    documents,
                    lambda d, pid=panel_id: _authored_panel_leaf(d, pid, "hidden"),
                    _SOURCE_INHERITED,
                )
                if hidden
                else None,
            }

        for panel_id, (app_name, label) in app_panels.items():
            hidden = profile.app_panel_hidden(app_name)
            labels.setdefault(panel_id, label)

            def names_app(d: Mapping[str, Any], name: str = app_name) -> bool:
                apps = d.get("apps")
                return isinstance(apps, Mapping) and name in apps

            author = _first_author(documents, names_app, _SOURCE_INHERITED)
            cells[panel_id] = {
                "state": STATE_MENU if hidden else STATE_TAB,
                "selected_in": author,
                "hidden_in": author if hidden else None,
            }

        default = profile.default_panel
        if default in cells and cells[default]["state"] == STATE_TAB:
            cells[default]["state"] = STATE_DEFAULT
        renders.append(
            {
                "name": profile.name if shown == PROFILE_FILENAME else path.stem,
                "file": shown,
                "logins": logins.get(path.stem, []) if shown != PROFILE_FILENAME else [],
                "default_panel": default,
                "panels": cells,
            }
        )

    every = {pid for render in renders for pid in render["panels"]}
    off: dict[str, str | None] = {"state": STATE_OFF, "selected_in": None, "hidden_in": None}
    for render in renders:
        for panel_id in every:
            render["panels"].setdefault(panel_id, dict(off))
    return {
        "panels": [{"id": pid, "label": labels[pid]} for pid in in_catalog_order(every)],
        "renders": renders,
    }


def format_panel_matrix(matrix: Mapping[str, Any]) -> list[str]:
    """*matrix* as a plain table, one row per panel and one column per login.

    Args:
        matrix: What :func:`panel_matrix` returns.

    Returns:
        The lines to print: the table, a legend, and who opens each login.
    """
    renders = matrix["renders"]
    header = ["panel"] + [render["name"] for render in renders]
    rows = [
        [f"{panel['label']} ({panel['id']})"]
        + [_MARKS[render["panels"][panel["id"]]["state"]] for render in renders]
        for panel in matrix["panels"]
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(row: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()

    lines = [line(header), line(["-" * width for width in widths]), *map(line, rows), ""]
    lines.append(
        "DEFAULT opens first · tab on the sidebar · menu in the panel menu only · - not there"
    )
    opened = [
        f"  {render['name']}: {', '.join(render['logins'])}"
        for render in renders
        if render["logins"]
    ]
    if opened:
        lines += ["", "who opens what", *opened]
    return lines


_STYLE = """*,*::before,*::after{box-sizing:border-box}
body{margin:0;padding:var(--space-5) var(--space-4);background:var(--surface-canvas);
  color:var(--text-primary);font:var(--weight-regular) var(--text-lg)/var(--leading-normal) var(--font-display)}
header{display:flex;align-items:baseline;gap:var(--space-3);flex-wrap:wrap;margin-bottom:var(--space-4)}
h1{margin:0;font-size:var(--text-3xl);font-weight:var(--weight-semibold)}
header p{margin:0;color:var(--text-secondary);flex:1 1 240px}
button{font:inherit;font-size:var(--text-md);color:var(--text-secondary);background:var(--surface-card);
  border:1px solid var(--border-default);border-radius:var(--radius-md);padding:var(--space-1) var(--space-3);cursor:pointer}
button:hover{background:var(--surface-hover);color:var(--text-primary)}
.card{background:var(--surface-card);border:1px solid var(--border-default);border-radius:var(--radius-lg);overflow-x:auto}
table{border-collapse:collapse;width:100%}
th,td{padding:var(--space-2) var(--space-3);text-align:center;white-space:nowrap;
  border-bottom:1px solid var(--surface-hairline)}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--surface-hover)}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--surface-card)}
thead th{vertical-align:bottom;background:var(--surface-well);font-weight:var(--weight-semibold)}
thead th:first-child{background:var(--surface-well);font-size:var(--text-sm);letter-spacing:1px;
  text-transform:uppercase;color:var(--text-muted)}
small{display:block;font:var(--weight-regular) var(--text-base)/var(--leading-tight) var(--font-mono);color:var(--text-muted)}
.pill{display:inline-block;min-width:64px;padding:2px var(--space-2);border-radius:var(--radius-full);
  font-size:var(--text-md);border:1px solid transparent}
.default .pill{background:var(--color-accent);color:var(--color-on-accent);font-weight:var(--weight-semibold)}
.tab .pill{background:var(--accent-tint-12);color:var(--color-accent-light);border-color:var(--accent-tint-25)}
.menu .pill{background:var(--accent-secondary-tint-12);color:var(--color-accent-secondary);
  border-color:var(--accent-secondary-tint-25)}
.off .pill{color:var(--text-muted)}
td[data-why]{cursor:pointer}
td[data-why]:focus-visible,td.picked{outline:2px solid var(--surface-focus);outline-offset:-2px}
.legend{display:flex;gap:var(--space-4);flex-wrap:wrap;margin:var(--space-3) 0;color:var(--text-secondary);
  font-size:var(--text-md)}
.legend span{display:inline-flex;align-items:center;gap:var(--space-2)}
#why{margin-top:var(--space-3);padding:var(--space-3) var(--space-4);min-height:44px;background:var(--surface-well);
  border:1px solid var(--border-default);border-radius:var(--radius-lg);color:var(--text-secondary)}
"""

_INTRO = (
    "<h1>Panels by login</h1>"
    "<p>What each login of this deployment shows. Select a cell to see which file decides it.</p>"
)

_TABLE = """<div class="card"><table><thead><tr><th>Panel</th>__HEAD__</tr></thead><tbody>__BODY__</tbody></table></div>
<div class="legend">
<span class="default"><span class="pill">default</span>opens first</span>
<span class="tab"><span class="pill">tab</span>on the sidebar</span>
<span class="menu"><span class="pill">menu</span>in the panel menu only</span>
<span class="off"><span class="pill">&ndash;</span>not there</span></div>
<div id="why">Select a cell.</div>
"""

_PICK = """const data=__DATA__;
function pick(td){document.querySelectorAll('td.picked').forEach(c=>c.classList.remove('picked'));
  td.classList.add('picked');document.getElementById('why').textContent=td.dataset.why;}
document.querySelectorAll('td[data-why]').forEach(td=>{
  td.addEventListener('click',()=>pick(td));
  td.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick(td);}});});
"""

#: The page a maintainer opens from disk: it links nothing, so it carries the
#: token blocks and its own light/dark button.
_PAGE = (
    """<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panels by login</title>
<style>
__TOKENS__
"""
    + _STYLE
    + "</style></head><body>\n<header>"
    + _INTRO
    + """
<button type="button" id="theme">Light</button></header>
"""
    + _TABLE
    + """<script>
const root=document.documentElement,button=document.getElementById('theme');
function theme(name){root.dataset.theme=name;button.textContent=name==='dark'?'Light':'Dark';}
theme(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark');
button.addEventListener('click',()=>theme(root.dataset.theme==='dark'?'light':'dark'));
"""
    + _PICK
    + "</script></body></html>\n"
)

#: The same table as a panel of the web terminal: it links the design system by
#: the panel contract's canonical tags and follows the terminal's theme, so it
#: has no button of its own.
_PANEL_PAGE = (
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="/design-system/js/theme-boot.js"></script>
<link rel="stylesheet" href="/design-system/css/tokens.css">
<link rel="stylesheet" href="/design-system/css/base.css">
<link rel="stylesheet" href="/design-system/css/primitives.css">
<link rel="stylesheet" href="/static/fonts/fonts.css">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Panels by login</title>
<script type="module">
  import { initTheme } from '/design-system/js/theme-manager.js';
  import { applyEmbedded } from '/design-system/js/frame-params.js';
  initTheme({ role: 'follower' });
  applyEmbedded();
</script>
<style>
html,body{height:auto;overflow:auto}
"""
    + _STYLE
    + "</style></head><body>\n<header>"
    + _INTRO
    + "</header>\n"
    + _TABLE
    + "<script>\n"
    + _PICK
    + "</script></body></html>\n"
)

_PAGE_MARKS = {STATE_DEFAULT: "default", STATE_TAB: "tab", STATE_MENU: "menu", STATE_OFF: "\u2013"}


def _theme_tokens() -> str:
    """The design system's dark and light token blocks, for a page that links nothing.

    The page is one file a maintainer opens from disk, so it cannot link
    ``tokens.css``; it carries the two blocks of it instead, and so wears the
    colours and scale every OSPREY panel wears.
    """
    from importlib.resources import files

    text = (
        files("osprey.interfaces.design_system")
        .joinpath("static", "css", "tokens.css")
        .read_text(encoding="utf-8")
    )
    blocks = []
    for opener in (':root, [data-theme="dark"] {', '[data-theme="light"] {'):
        start = text.index(opener)
        blocks.append(text[start : text.index("\n}", start) + 2])
    return "\n".join(blocks)


def _why(render: Mapping[str, Any], panel: Mapping[str, str]) -> str:
    """One sentence saying which file decides *panel* on *render*."""
    cell = render["panels"][panel["id"]]
    who = f"{panel['label']} on {render['name']}"
    if cell["state"] == STATE_OFF:
        return f"{who}: not selected. No web_panels list or apps entry this login reads names it."
    said = f"{who}: selected in {cell['selected_in']}"
    if cell["state"] == STATE_MENU:
        return f"{said}; kept in the panel menu by {cell['hidden_in']}."
    if cell["state"] == STATE_DEFAULT:
        return f"{said}; default_panel of {render['file']} opens on it."
    return f"{said}."


def panel_matrix_html(matrix: Mapping[str, Any]) -> str:
    """*matrix* as one self-contained page: the table, each cell saying what decides it.

    Args:
        matrix: What :func:`panel_matrix` returns.

    Returns:
        The page's HTML. It loads nothing from the network.
    """
    return _filled(_PAGE.replace("__TOKENS__", _theme_tokens()), matrix)


def _filled(page: str, matrix: Mapping[str, Any]) -> str:
    """*page* with the table of *matrix* and its data written in."""
    esc = html.escape
    head = "".join(
        f"<th>{esc(render['name'])}<small>{esc(render['file'])}</small>"
        + "".join(f"<small>{esc(login)}</small>" for login in render["logins"])
        + "</th>"
        for render in matrix["renders"]
    )
    body = ""
    for panel in matrix["panels"]:
        body += f"<tr><td>{esc(panel['label'])}<small>{esc(panel['id'])}</small></td>"
        for render in matrix["renders"]:
            state = render["panels"][panel["id"]]["state"]
            body += (
                f'<td class="{state}" tabindex="0" data-why="{esc(_why(render, panel), quote=True)}">'
                f'<span class="pill">{esc(_PAGE_MARKS[state])}</span></td>'
            )
        body += "</tr>"
    data = json.dumps(matrix).replace("</", "<\\/")
    return page.replace("__HEAD__", head).replace("__BODY__", body).replace("__DATA__", data)


def panel_matrix_bundle(matrix: Mapping[str, Any]) -> dict[str, str]:
    """*matrix* as the files of the ``roster`` panel's bundle.

    The bundle is what the panel contract asks of any static panel
    (:mod:`osprey.interfaces.design_system.panels.validator`): a manifest and
    an entry page that links the design system. ``osprey build`` writes it into
    a render that selects the panel, and the web terminal serves it from there.

    Args:
        matrix: What :func:`panel_matrix` returns.

    Returns:
        File name to content: ``manifest.json`` and ``index.html``.
    """
    manifest = {
        "id": ROSTER_PANEL_ID,
        "label": BUILD_PANELS[ROSTER_PANEL_ID],
        "entry": "index.html",
        "version": 1,
    }
    return {
        "manifest.json": json.dumps(manifest, indent=2) + "\n",
        "index.html": _filled(_PANEL_PAGE, matrix),
    }
