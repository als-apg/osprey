"""Artifact forms: the one grouping level the gallery lists artifacts by.

A *form* says what an artifact is to the operator — a plot, a trend, a table,
a document — independent of the file format it is stored in. It is computed
server-side from what a writer's code recorded on the entry (``kind``,
``artifact_type``, mime, ``tool_source``, ``metadata``), never from agent input.

:func:`form_of` is an entry's *intrinsic* form: it looks at that entry alone and
never returns ``runs``. Whether an entry heads a run is decided at read time
from its run group, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.stores.artifact_store import ArtifactEntry

BASE_FORMS: tuple[str, ...] = (
    "runs",
    "plots",
    "trends",
    "tables",
    "documents",
    "dashboards",
    "images",
    "data",
)
"""Every base form, in listing order. ``runs`` is assigned only to the head of a
run group at read time; :func:`form_of` never returns it."""

_FORM_LABELS: dict[str, str] = {
    "runs": "Runs",
    "plots": "Plots",
    "trends": "Trends",
    "tables": "Tables",
    "documents": "Documents",
    "dashboards": "Dashboards",
    "images": "Images",
    "data": "Data",
}

FALLBACK_FORM = "data"
"""The form of an entry nothing else claims: :func:`form_of` is total."""

_TREND_TOOLS = frozenset({"archiver_read", "channel_read"})
_IMAGE_TOOLS = frozenset({"screenshot_capture", "phoebus_snapshot"})
_VIZ_TOOLS = frozenset({"create_static_plot", "create_interactive_plot"})

_FORM_BY_TYPE: dict[str, str] = {
    "plot_html": "plots",
    "plot_png": "plots",
    "table_html": "tables",
    "dashboard_html": "dashboards",
    "html": "documents",
    "markdown": "documents",
    "text": "documents",
    "notebook": "documents",
    "json": "data",
    "file": "data",
    "binary": "data",
}

_DOCUMENT_MIMES = frozenset(
    {
        "application/pdf",
        "text/html",
        "text/markdown",
        "text/plain",
        "application/x-ipynb+json",
    }
)
_DATA_MIMES = frozenset({"text/csv", "application/json"})

RUN_RECORD_KEY = "run_record"
"""``metadata`` key, set to True, on an artifact that records a script run rather
than holding its result: the approval hook's approval preview, and the
executor's notebook and result record. Plots and files the script saves carry no
such key. The web terminal's sidebar lists these under "Script runs" (it asks the
gallery for them through ``GET /api/artifacts?run=true``) and keeps them out of its
Artifacts list; :func:`is_run_record` is the verdict both listings share."""


def is_run_record(entry: ArtifactEntry) -> bool:
    """Whether ``entry`` records a script run rather than holding its result.

    True when the entry carries :data:`RUN_RECORD_KEY` in its metadata, or — for
    entries saved before that key existed — when it has what the writers always
    gave a run record: the ``code_output`` category, the ``osprey_approval`` tool
    source, or a notebook titled ``Notebook: …`` from ``execute`` or
    ``execute_file``. The gallery sends this verdict to the browser as the
    ``run_record`` field so no client re-derives it.
    """
    if entry.metadata.get(RUN_RECORD_KEY) is True:
        return True
    if entry.category == "code_output":
        return True
    if entry.tool_source == "osprey_approval":
        return True
    return (
        entry.artifact_type == "notebook"
        and entry.tool_source in ("execute", "execute_file")
        and entry.title.startswith("Notebook: ")
    )


def form_label(key: str, registry: Any = None) -> str:
    """Human label of form ``key``.

    A base form has its fixed label. An extension key (``registry`` declares it
    with an eligible base form, see :func:`_registered_base_form`) has the label
    the registry declares; anything else is title-cased as is.
    """
    if key in _FORM_LABELS:
        return _FORM_LABELS[key]
    if _registered_base_form(key, registry) is not None:
        label = _declared_label(key, registry)
        if label:
            return label
    return key.replace("_", " ").title()


def _declared_label(kind: str, registry: Any) -> str | None:
    """The ``label`` ``registry`` declares for ``kind``, if it is a usable string."""
    try:
        declared = registry.get(kind)
    except Exception:  # a broken registry never breaks a listing
        return None
    if isinstance(declared, dict):
        label = declared.get("label")
    else:
        label = getattr(declared, "label", None)
    return label if isinstance(label, str) and label else None


def extension_key(entry: Any, registry: Any) -> str | None:
    """``entry.kind`` when it is an extension key in ``registry``, else ``None``."""
    kind = getattr(entry, "kind", "") or ""
    return kind if _registered_base_form(kind, registry) is not None else None


def _registered_base_form(kind: str, registry: Any) -> str | None:
    """Base form an extension type ``kind`` declares in ``registry``, if any.

    ``registry`` is duck-typed: anything with ``.get(kind)`` returning an object
    (or mapping) that carries ``base_form``. ``None`` or an unknown kind yields
    ``None``, and so does a declared ``runs`` or an unknown base form, so the
    intrinsic mapping takes over.
    """
    if not kind or registry is None:
        return None
    getter = getattr(registry, "get", None)
    if getter is None:
        return None
    try:
        declared = getter(kind)
    except Exception:  # a broken registry never breaks a listing
        return None
    if declared is None:
        return None
    if isinstance(declared, dict):
        base = declared.get("base_form")
    else:
        base = getattr(declared, "base_form", None)
    if base in BASE_FORMS and base != "runs":
        return str(base)
    return None


def form_of(entry: ArtifactEntry, registry: Any = None) -> str:
    """The intrinsic form of ``entry``: one of :data:`BASE_FORMS`, never ``runs``.

    Precedence: a registered extension ``kind`` → its declared base form;
    archiver and channel readings (or ``metadata.data_type == "timeseries"``) →
    trends; screenshots and widget snapshots → images; then by
    ``artifact_type``, with images split into plots (executor figures stamped
    ``metadata.figure=True``, and the viz tools' figures) and images (everything
    else); then by mime; anything left → :data:`FALLBACK_FORM`.
    """
    registered = _registered_base_form(getattr(entry, "kind", "") or "", registry)
    if registered is not None:
        return registered

    metadata = getattr(entry, "metadata", None) or {}
    tool_source = getattr(entry, "tool_source", "") or ""
    artifact_type = getattr(entry, "artifact_type", "") or ""
    mime = (getattr(entry, "mime_type", "") or "").split(";", 1)[0].strip().lower()

    if tool_source in _TREND_TOOLS or metadata.get("data_type") == "timeseries":
        return "trends"
    if tool_source in _IMAGE_TOOLS:
        return "images"
    if artifact_type == "image":
        if metadata.get("figure") is True or tool_source in _VIZ_TOOLS:
            return "plots"
        return "images"
    if artifact_type == "file" and mime == "application/pdf":
        return "documents"
    if artifact_type in _FORM_BY_TYPE:
        return _FORM_BY_TYPE[artifact_type]
    if mime.startswith("image/"):
        return "images"
    if mime in _DATA_MIMES:
        return "data"
    if mime in _DOCUMENT_MIMES or mime.startswith("text/"):
        return "documents"
    return FALLBACK_FORM


def is_interactive(entry: ArtifactEntry) -> bool:
    """Whether ``entry`` is an interactive (plotly) figure."""
    return (getattr(entry, "artifact_type", "") or "") == "plot_html"
