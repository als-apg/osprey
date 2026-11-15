"""Which login shows which panel: the answer ``osprey profile panels`` prints.

A login's tab strip is decided in several places: the ``web_panels`` list of
the deployment and of the login's own file, ``web.panels.<id>.hidden`` under
either ``config:``, ``default_panel``, and the ``apps:`` entries that show an
application's panel or keep it off the sidebar. Each is read where the build
reads it, from the resolved profile of every login, so the table is what the
renders will carry and not a second reading of the files.

:func:`panel_matrix` is the derivation, :func:`format_panel_matrix` the plain
table and :func:`panel_matrix_html` the same table as one self-contained page.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from osprey.profiles.web_panels import UNIVERSAL_PANELS, in_catalog_order, panel_label

__all__ = [
    "STATE_DEFAULT",
    "STATE_MENU",
    "STATE_OFF",
    "STATE_TAB",
    "format_panel_matrix",
    "panel_matrix",
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
    from osprey.deployment.web_terminals.roster import effective_tier, shared_app_name

    from .build_profile_emit import effective_web_terminals

    web_terminals = effective_web_terminals(config)
    if not web_terminals.get("enabled"):
        return {}
    opened: dict[str, list[str]] = {}
    users = web_terminals.get("users")
    for entry in users if isinstance(users, list) else []:
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


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panels by login</title>
<style>
:root{--bg:#fff;--fg:#1a1d21;--dim:#667085;--line:#e3e6ea;--tab:#d7ecff;--def:#1f6feb;--deffg:#fff;--menu:#fdf0cc}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e6e8eb;--dim:#98a2b3;--line:#2b3138;--tab:#17406b;--def:#4c9aff;--deffg:#0b1320;--menu:#5a4712}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}p{color:var(--dim);margin:0 0 16px}
.wrap{overflow-x:auto}table{border-collapse:collapse}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:center;white-space:nowrap}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--bg)}
thead th{vertical-align:bottom}th small,td small{display:block;color:var(--dim);font-weight:400}
.tab{background:var(--tab)}.default{background:var(--def);color:var(--deffg);font-weight:600}
.menu{background:var(--menu)}.off{color:var(--dim)}
#why{margin-top:16px;min-height:3em;color:var(--dim)}td[data-why]{cursor:pointer}
</style></head><body>
<h1>Panels by login</h1>
<p>What each login of this deployment shows. Select a cell to see which file decides it.</p>
<div class="wrap"><table><thead><tr><th>panel</th>__HEAD__</tr></thead><tbody>__BODY__</tbody></table></div>
<div id="why"></div>
<script>
const data=__DATA__;
document.querySelectorAll('td[data-why]').forEach(td=>td.addEventListener('click',()=>{
  document.getElementById('why').textContent=td.dataset.why;}));
</script></body></html>
"""


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
    esc = html.escape
    head = "".join(
        f"<th>{esc(render['name'])}<small>{esc(render['file'])}</small>"
        f"<small>{esc(', '.join(render['logins']))}</small></th>"
        for render in matrix["renders"]
    )
    body = ""
    for panel in matrix["panels"]:
        body += f"<tr><td>{esc(panel['label'])}<small>{esc(panel['id'])}</small></td>"
        for render in matrix["renders"]:
            state = render["panels"][panel["id"]]["state"]
            body += (
                f'<td class="{state}" data-why="{esc(_why(render, panel), quote=True)}">'
                f"{esc(_MARKS[state])}</td>"
            )
        body += "</tr>"
    data = json.dumps(matrix).replace("</", "<\\/")
    return _PAGE.replace("__HEAD__", head).replace("__BODY__", body).replace("__DATA__", data)
