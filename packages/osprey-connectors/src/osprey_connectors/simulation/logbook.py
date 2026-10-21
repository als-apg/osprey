"""The logbook entries a scenario narrates, and the pictures they carry.

:class:`ScenarioLogEntry` is the one entry shape the logbook seed reads.
An entry's pictures are shipped image files or :class:`PlotSpec` records the
seeder draws against the entry's timestamp; :func:`parse_log_attachments`
resolves and checks an entry's ``attachments`` list, and
:func:`parse_plot_spec` validates one decoded plot spec.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey_connectors.relative_time import RelativeTimestamp

__all__ = [
    "IMAGE_SIGNATURES",
    "SIGNATURE_BYTES",
    "PlotSeries",
    "PlotSpec",
    "ScenarioLogEntry",
    "matches_signature",
    "parse_log_attachments",
    "parse_plot_spec",
]

# Logbook attachment-item schema: closed. An item
# carries exactly one of these keys: 'path' names a shipped picture, 'plot' a
# plot spec drawn at seed time against the entry's own timestamp.
_ATTACHMENT_KEYS = ("path", "plot")

# Plot-spec schema: closed. Every key but 'ylim' is required.
_PLOT_SPEC_KEYS = ("filename", "title", "ylabel", "hours_before", "series", "ylim")
_PLOT_SERIES_KEYS = ("label", "values")

# Picture formats a logbook attachment may name: suffix -> accepted file
# signatures as (byte offset, magic bytes).
IMAGE_SIGNATURES: dict[str, tuple[tuple[int, bytes], ...]] = {
    ".png": ((0, b"\x89PNG\r\n\x1a\n"),),
    ".jpg": ((0, b"\xff\xd8\xff"),),
    ".jpeg": ((0, b"\xff\xd8\xff"),),
    ".gif": ((0, b"GIF87a"), (0, b"GIF89a")),
    ".webp": ((8, b"WEBP"),),
}
SIGNATURE_BYTES = 16


@dataclass(frozen=True)
class PlotSeries:
    """One named line of a :class:`PlotSpec`, one value per axis point."""

    label: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class PlotSpec:
    """A time-series picture drawn when its entry is seeded, not shipped as a file.

    The time axis is relative: ``hours_before`` counts back from the instant the
    entry is written (``0.0``), so the drawn picture carries that entry's real
    dates whatever anchor the narrative resolves against. Every series has one
    value per point of that shared axis.

    Attributes:
        filename: Name the drawn picture is stored under (a ``.png`` file name).
        title: Figure title.
        ylabel: Y-axis label.
        hours_before: Shared time axis, non-increasing, ending at ``0.0``.
        series: The lines drawn, in legend order.
        ylim: Fixed y-axis range ``(low, high)``, or ``None`` to fit the data.
    """

    filename: str
    title: str
    ylabel: str
    hours_before: tuple[float, ...]
    series: tuple[PlotSeries, ...]
    ylim: tuple[float, float] | None = None


@dataclass(frozen=True)
class ScenarioLogEntry:
    """A logbook entry owned by a scenario.

    The one entry shape the ARIEL DB seed (via ``apply``) reads, whether the
    entry came from a scenario bundle's ``logbook.json`` or a facility
    scenario's ``logbook`` block, so the telemetry overlay and its narrative
    ship together.

    ``attachments`` are the pictures the entry carries, in the order its
    ``attachments`` list names them: a shipped picture as an absolute path
    inside the scenario directory (checked at load time to exist and to be an
    image), or a :class:`PlotSpec` the seeder draws against the entry's
    timestamp.
    """

    entry_id: str
    when: RelativeTimestamp
    author: str
    title: str
    text: str
    tags: tuple[str, ...]
    categories: tuple[str, ...]
    loto_tag: str | None
    extra: dict[str, Any]
    attachments: tuple[Path | PlotSpec, ...] = ()


def parse_log_attachments(prefix: str, raw: Any, bundle: Path) -> tuple[Path | PlotSpec, ...]:
    """Resolve a logbook entry's ``attachments`` list to the pictures it carries.

    Each item holds exactly one key. ``{"path": "<relative path>"}`` names a
    shipped picture: it must name an existing file with an image suffix that
    starts with that image format's signature. ``{"plot": "<relative path>"}``
    names a plot spec: a ``.json`` file holding a valid spec (see
    :func:`parse_plot_spec`). Either path is relative to the bundle directory
    and must stay inside it, and no two pictures of one entry may share a file
    name, so a misnamed, missing or malformed picture is a load error rather
    than a seed failure.
    """
    if not isinstance(raw, list):
        raise ValueError(f"{prefix}: 'attachments' must be a list, got {raw!r}")
    root = bundle.resolve()
    pictures: list[Path | PlotSpec] = []
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{prefix}: each attachment must be a mapping, got {item!r}")
        unknown = sorted(set(item) - set(_ATTACHMENT_KEYS))
        if unknown:
            raise ValueError(f"{prefix}: attachment has unknown keys {unknown}")
        if len(item) != 1:
            raise ValueError(
                f"{prefix}: an attachment names exactly one of 'path' or 'plot', got {item!r}"
            )
        (key, rel), *_ = item.items()
        path = _bundle_file(prefix, key, rel, root)
        picture = (
            _shipped_picture(prefix, rel, path) if key == "path" else _plot_spec(prefix, rel, path)
        )
        name = picture.name if isinstance(picture, Path) else picture.filename
        if name in names:
            raise ValueError(f"{prefix}: two attachments are both named {name!r}")
        names.add(name)
        pictures.append(picture)
    return tuple(pictures)


def _bundle_file(prefix: str, key: str, rel: Any, root: Path) -> Path:
    """Resolve an attachment item's relative path, which must stay inside ``root``."""
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"{prefix}: attachment {key!r} must be a non-empty string, got {rel!r}")
    path = (root / rel).resolve()
    if Path(rel).is_absolute() or not path.is_relative_to(root):
        raise ValueError(
            f"{prefix}: attachment path {rel!r} must be relative to the scenario directory"
        )
    return path


def _shipped_picture(prefix: str, rel: str, path: Path) -> Path:
    """Check that ``path`` is an existing image file whose bytes match its suffix."""
    signatures = IMAGE_SIGNATURES.get(path.suffix.lower())
    if signatures is None:
        raise ValueError(
            f"{prefix}: attachment {rel!r} is not a picture "
            f"(accepted: {', '.join(sorted(IMAGE_SIGNATURES))})"
        )
    if not path.is_file():
        raise ValueError(f"{prefix}: attachment file {rel!r} not found at {path}")
    with path.open("rb") as handle:
        head = handle.read(SIGNATURE_BYTES)
    if not any(matches_signature(head, sig) for sig in signatures):
        raise ValueError(
            f"{prefix}: attachment {rel!r} does not hold {path.suffix.lower()} image data"
        )
    return path


def _plot_spec(prefix: str, rel: str, path: Path) -> PlotSpec:
    """Read and validate the plot spec at ``path``."""
    if path.suffix.lower() != ".json":
        raise ValueError(f"{prefix}: plot spec {rel!r} must be a .json file")
    if not path.is_file():
        raise ValueError(f"{prefix}: plot spec {rel!r} not found at {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{prefix}: plot spec {rel!r} is not valid JSON: {exc}") from exc
    return parse_plot_spec(raw, f"{prefix}: plot spec {rel!r}")


def parse_plot_spec(raw: Any, where: str = "plot spec") -> PlotSpec:
    """Validate a decoded plot spec and return it as a :class:`PlotSpec`.

    The schema is closed: ``filename`` (a ``.png`` file name, no directory),
    ``title``, ``ylabel``, ``hours_before`` (at least two finite numbers,
    non-negative, non-increasing, ending at ``0``), ``series`` (a non-empty
    list of ``{"label", "values"}`` with unique non-empty labels and one finite
    number per ``hours_before`` point) and optional ``ylim`` (``[low, high]``
    with ``low < high``).

    Args:
        raw: The decoded JSON value.
        where: How error messages name the spec.

    Raises:
        ValueError: If the spec breaks any of those rules.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: must be a JSON object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_PLOT_SPEC_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")
    missing = [key for key in _PLOT_SPEC_KEYS if key != "ylim" and key not in raw]
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")

    filename = raw["filename"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or Path(filename).suffix.lower() != ".png"
        or Path(filename).stem == ""
    ):
        raise ValueError(f"{where}: 'filename' must be a .png file name, got {filename!r}")
    for key in ("title", "ylabel"):
        if not isinstance(raw[key], str):
            raise ValueError(f"{where}: {key!r} must be a string, got {raw[key]!r}")

    hours = _finite_numbers(where, "'hours_before'", raw["hours_before"])
    if len(hours) < 2:
        raise ValueError(f"{where}: 'hours_before' needs at least two points")
    if any(later > earlier for earlier, later in zip(hours, hours[1:], strict=False)):
        raise ValueError(f"{where}: 'hours_before' must be non-increasing")
    if hours[-1] != 0.0:
        raise ValueError(f"{where}: 'hours_before' must end at 0, got {hours[-1]!r}")

    series_raw = raw["series"]
    if not isinstance(series_raw, list) or not series_raw:
        raise ValueError(f"{where}: 'series' must be a non-empty list")
    series: list[PlotSeries] = []
    for item in series_raw:
        if not isinstance(item, dict) or set(item) != set(_PLOT_SERIES_KEYS):
            raise ValueError(
                f"{where}: each series is a mapping with exactly 'label' and 'values', got {item!r}"
            )
        label = item["label"]
        if not isinstance(label, str) or not label:
            raise ValueError(f"{where}: series 'label' must be a non-empty string, got {label!r}")
        if any(existing.label == label for existing in series):
            raise ValueError(f"{where}: two series are both labelled {label!r}")
        values = _finite_numbers(where, f"series {label!r} 'values'", item["values"])
        if len(values) != len(hours):
            raise ValueError(
                f"{where}: series {label!r} has {len(values)} values for "
                f"{len(hours)} 'hours_before' points"
            )
        series.append(PlotSeries(label=label, values=values))

    ylim = None
    if "ylim" in raw:
        bounds = _finite_numbers(where, "'ylim'", raw["ylim"])
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError(f"{where}: 'ylim' must be [low, high] with low < high")
        ylim = (bounds[0], bounds[1])

    return PlotSpec(
        filename=filename,
        title=raw["title"],
        ylabel=raw["ylabel"],
        hours_before=hours,
        series=tuple(series),
        ylim=ylim,
    )


def _finite_numbers(where: str, key: str, raw: Any) -> tuple[float, ...]:
    """``raw`` as a tuple of floats, refusing anything but a list of finite numbers."""
    if not isinstance(raw, list) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in raw
    ):
        raise ValueError(f"{where}: {key} must be a list of finite numbers")
    return tuple(float(v) for v in raw)


def matches_signature(head: bytes, signature: tuple[int, bytes]) -> bool:
    """Whether ``head`` carries ``signature`` (an ``(offset, bytes)`` pair)."""
    offset, magic = signature
    return head[offset : offset + len(magic)] == magic
