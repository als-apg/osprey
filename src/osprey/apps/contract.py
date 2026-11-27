"""The application contract: the number an ``app.yml`` declares, and its check.

An application states, with ``contract: N`` in its ``app.yml``, the version of
the contract it is written for. This OSPREY implements :data:`APP_CONTRACT`; a
manifest declaring any other number, or none, is refused before anything else
in it is read.
"""

from __future__ import annotations

import re
from typing import Any

from osprey.apps.errors import Violation
from osprey.docs_links import APP_CONTRACT_URL
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from osprey.utils.owner_header import CALLER_HEADER, OWNER_HEADER, SESSION_HEADER

#: The version of the application contract this OSPREY implements.
APP_CONTRACT = 1

#: The key of ``app.yml`` that declares the number, and the refusal's field and rule.
CONTRACT_KEY = "contract"

#: The HTTP headers an application reads from the requests OSPREY sends it: the
#: signed-in owner, the agent session, the calling agent, and the panel's
#: launch token.
APP_HEADERS: tuple[str, ...] = (OWNER_HEADER, SESSION_HEADER, CALLER_HEADER, LAUNCH_TOKEN_HEADER)


def contract_violation(data: dict[str, Any], *, app: str, source: str | None) -> Violation | None:
    """The refusal of a manifest whose ``contract`` is not :data:`APP_CONTRACT`.

    Args:
        data: The manifest as loaded, already known to be a mapping.
        app: The application's name, used in the refusal.
        source: The file a refusal names when it is not ``apps/<app>/app.yml``.

    Returns:
        ``None`` when the manifest declares :data:`APP_CONTRACT`, else the one
        violation that refuses it.
    """
    n = APP_CONTRACT
    declare = f'this OSPREY implements contract {n}. Add "contract: {n}" to app.yml.'
    if CONTRACT_KEY not in data:
        message = f"not declared; {declare}"
    else:
        value = data[CONTRACT_KEY]
        if type(value) is not int:
            message = f"{value!r} is not an integer; {declare}"
        elif value == n:
            return None
        elif value < n:
            message = (
                f"the app is written for contract {value}; this OSPREY implements {n}. "
                f"Update the app to contract {n}."
            )
        else:
            message = (
                f"the app is written for contract {value}; this OSPREY implements {n}. "
                "Upgrade OSPREY."
            )
    return Violation(
        app, CONTRACT_KEY, CONTRACT_KEY, f"{message} See {APP_CONTRACT_URL}.", source=source
    )


#: The files that define each section of :func:`surface`, relative to the
#: repository root. A change to one of them may change what apps rely on.
SURFACE_SOURCES: dict[str, tuple[str, ...]] = {
    "manifest": ("src/osprey/apps/manifest.py",),
    "agent_frontmatter": ("src/osprey/apps/agents.py",),
    "sdk": ("src/osprey/apps/sdk/__init__.py",),
    "sdk_testing": ("src/osprey/apps/sdk/testing/__init__.py",),
    "kinds": ("packages/osprey-facility/src/osprey_facility/kinds.py",),
    "page_ready": ("src/osprey/interfaces/design_system/static/js/page-ready.js",),
    "primitives": ("src/osprey/interfaces/design_system/static/css/primitives.css",),
    "panel_messages": ("src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js",),
    "headers": ("src/osprey/utils/owner_header.py", "src/osprey/utils/launch_token.py"),
    "environment": ("src/osprey/apps/projection.py",),
}

#: A top-level ``export`` declaration of a module; prose in comments is not one.
_JS_EXPORT = re.compile(
    r"^export\s+(?:async\s+)?(?:const|let|var|function\*?|class)\s+([A-Za-z_$][\w$]*)", re.MULTILINE
)

#: A CSS block comment.
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

#: A design-system class, and one compounded with a modifier (``.osprey-btn.ghost``).
_CSS_CLASS = re.compile(r"\.(osprey-[\w-]+)")
_CSS_COMPOUND = re.compile(r"\.(osprey-[\w-]+)\.([\w-]+)")

#: The ``type`` of a message the panel host posts to a panel frame.
_PANEL_MESSAGE = re.compile(r"""\btype:\s*['"](osprey-[\w-]+)['"]""")


def surface() -> dict[str, Any]:
    """The names an application relies on, read from the live code.

    The ``manifest`` section maps each dotted ``app.yml`` key path to its JSON
    type set (``integer|null``); every other section is a sorted list of names.

    Returns:
        One entry per key of :data:`SURFACE_SOURCES`.
    """
    # Imported here, not at module level: ``manifest.py`` imports this module,
    # and the SDK's import graph must stay out of a manifest import.
    from importlib.resources import files

    import osprey.apps.sdk as sdk
    import osprey.apps.sdk.testing as sdk_testing
    import osprey_facility.kinds as kinds
    from osprey.apps.agents import AppAgent
    from osprey.apps.manifest import AppManifest
    from osprey.apps.projection import APP_ENVIRONMENT

    design_system = files("osprey.interfaces.design_system") / "static"
    web_terminal = files("osprey.interfaces.web_terminal") / "static"
    page_ready = (design_system / "js" / "page-ready.js").read_text(encoding="utf-8")
    primitives = (design_system / "css" / "primitives.css").read_text(encoding="utf-8")
    messages = (web_terminal / "js" / "panel-iframe-sync.js").read_text(encoding="utf-8")

    return {
        "manifest": _schema_paths(AppManifest.model_json_schema()),
        "agent_frontmatter": sorted(
            field.alias or name for name, field in AppAgent.model_fields.items()
        ),
        "sdk": sorted(sdk.__all__),
        "sdk_testing": sorted(sdk_testing.__all__),
        "kinds": sorted(kinds.__all__),
        "page_ready": sorted(set(_JS_EXPORT.findall(page_ready))),
        "primitives": _css_classes(primitives),
        "panel_messages": sorted(set(_PANEL_MESSAGE.findall(messages))),
        "headers": sorted(APP_HEADERS),
        "environment": sorted(APP_ENVIRONMENT),
    }


def _css_classes(css: str) -> list[str]:
    """Every ``osprey-*`` class of a stylesheet, and every ``osprey-<base>.<modifier>``.

    Comments are stripped first, so a class named only in prose is not recorded.
    """
    css = _CSS_COMMENT.sub("", css)
    names = set(_CSS_CLASS.findall(css))
    names.update(f"{base}.{modifier}" for base, modifier in _CSS_COMPOUND.findall(css))
    return sorted(names)


def _schema_paths(schema: dict[str, Any]) -> dict[str, str]:
    """Every key path of a JSON schema with its bare type set (``null`` last), by path.

    ``$ref`` is resolved against ``$defs``; a mapping's values are written
    ``<key>.*``, a list's items ``<key>[]``; every arm of ``anyOf`` and
    ``oneOf`` is walked. Bounds, patterns, enums and defaults are dropped.
    """
    defs = schema.get("$defs", {})
    found: dict[str, set[str]] = {}

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        while "$ref" in node:
            node = defs[node["$ref"].rsplit("/", 1)[-1]]
        return node

    def arms(node: dict[str, Any]) -> list[dict[str, Any]]:
        node = resolve(node)
        union = node.get("anyOf", []) + node.get("oneOf", [])
        if not union:
            return [node]
        return [leaf for arm in union for leaf in arms(arm)]

    def walk(node: dict[str, Any], path: str) -> None:
        for arm in arms(node):
            kind = arm.get("type", "any")
            if path:
                found.setdefault(path, set()).update([kind] if isinstance(kind, str) else kind)
            prefix = f"{path}." if path else ""
            for key, child in arm.get("properties", {}).items():
                walk(child, prefix + key)
            values = arm.get("additionalProperties")
            if isinstance(values, dict):
                walk(values, f"{path}.*")
            if isinstance(arm.get("items"), dict):
                walk(arm["items"], f"{path}[]")

    walk(schema, "")

    def spell(kinds: set[str]) -> str:
        return "|".join(sorted(kinds, key=lambda kind: (kind == "null", kind)))

    return {path: spell(found[path]) for path in sorted(found)}
