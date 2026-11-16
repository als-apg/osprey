"""The logbook entries a scenario narrates, and the pictures they carry.

:class:`ScenarioLogEntry` is the one entry shape the logbook seed reads.
An entry's pictures are shipped image files or :class:`PlotSpec` records the
seeder draws against the entry's timestamp; :func:`parse_log_attachments`
resolves and checks an entry's ``attachments`` list, and
:func:`parse_plot_spec` validates one decoded plot spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BeforeValidator, Field, ValidationError, ValidationInfo, field_validator
from pydantic_core import ErrorDetails, PydanticCustomError

from osprey_connectors.model import FrozenModel
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


def _listed(value: object) -> object:
    """A JSON list as the tuple a field holds; anything else is no list."""
    if isinstance(value, list):
        return tuple(value)
    raise PydanticCustomError("not_a_list", "Input should be a list")


#: A list of finite numbers, each an ``int`` or a ``float`` and never a ``bool``.
_Finite: TypeAlias = Annotated[
    tuple[Annotated[float, Field(allow_inf_nan=False)], ...], BeforeValidator(_listed)
]


class PlotSeries(FrozenModel):
    """One named line of a :class:`PlotSpec`, one value per axis point."""

    label: str = Field(min_length=1)
    values: _Finite


class PlotSpec(FrozenModel):
    """A time-series picture drawn when its entry is seeded, not shipped as a file.

    The time axis is relative: ``hours_before`` counts back from the instant the
    entry is written (``0.0``), so the drawn picture carries that entry's real
    dates whatever anchor the narrative resolves against. Every series has one
    value per point of that shared axis. :func:`parse_plot_spec` reads one from
    its decoded JSON, and names the key and the rule a spec breaks.

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
    hours_before: _Finite
    series: Annotated[tuple[PlotSeries, ...], BeforeValidator(_listed)]
    ylim: _Finite | None = None

    @field_validator("filename")
    @classmethod
    def _png_name(cls, filename: str) -> str:
        path = Path(filename)
        if path.name != filename or path.suffix.lower() != ".png" or path.stem == "":
            raise ValueError(f"'filename' must be a .png file name, got {filename!r}")
        return filename

    @field_validator("hours_before")
    @classmethod
    def _axis(cls, hours: tuple[float, ...]) -> tuple[float, ...]:
        if len(hours) < 2:
            raise ValueError("'hours_before' needs at least two points")
        if any(later > earlier for earlier, later in zip(hours, hours[1:], strict=False)):
            raise ValueError("'hours_before' must be non-increasing")
        if hours[-1] != 0.0:
            raise ValueError(f"'hours_before' must end at 0, got {hours[-1]!r}")
        return hours

    @field_validator("series")
    @classmethod
    def _lines(
        cls, series: tuple[PlotSeries, ...], info: ValidationInfo[object]
    ) -> tuple[PlotSeries, ...]:
        if not series:
            raise ValueError("'series' must be a non-empty list")
        hours = info.data.get("hours_before")
        labels: set[str] = set()
        for line in series:
            if line.label in labels:
                raise ValueError(f"two series are both labelled {line.label!r}")
            labels.add(line.label)
            if isinstance(hours, tuple) and len(line.values) != len(hours):
                raise ValueError(
                    f"series {line.label!r} has {len(line.values)} values for "
                    f"{len(hours)} 'hours_before' points"
                )
        return series

    @field_validator("ylim", mode="before")
    @classmethod
    def _stated_range(cls, ylim: object) -> object:
        """A stated ``ylim`` is a list; only one left out is ``None``."""
        if ylim is None:
            raise PydanticCustomError("not_a_list", "Input should be a list")
        return ylim

    @field_validator("ylim")
    @classmethod
    def _range(cls, ylim: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if ylim is not None and (len(ylim) != 2 or not ylim[0] < ylim[1]):
            raise ValueError("'ylim' must be [low, high] with low < high")
        return ylim


class ScenarioLogEntry(FrozenModel):
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
    extra: dict[str, object]
    attachments: tuple[Path | PlotSpec, ...] = ()


def parse_log_attachments(prefix: str, raw: object, bundle: Path) -> tuple[Path | PlotSpec, ...]:
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


def _bundle_file(prefix: str, key: str, rel: object, root: Path) -> Path:
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
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{prefix}: plot spec {rel!r} is not valid JSON: {exc}") from exc
    return parse_plot_spec(raw, f"{prefix}: plot spec {rel!r}")


def parse_plot_spec(raw: object, where: str = "plot spec") -> PlotSpec:
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
        ValueError: If the spec breaks any of those rules; the message names
            the key and the rule.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: must be a JSON object, got {type(raw).__name__}")
    try:
        return PlotSpec.model_validate(raw)
    except ValidationError as refusal:
        raise ValueError(f"{where}: {_plot_refusal(refusal.errors(), raw)}") from None


def _plot_refusal(errors: list[ErrorDetails], raw: dict[object, object]) -> str:
    """The rule a plot spec breaks, as :func:`parse_plot_spec` words it."""
    unknown = sorted(
        str(error["loc"][0])
        for error in errors
        if error["type"] == "extra_forbidden" and len(error["loc"]) == 1
    )
    if unknown:
        return f"unknown keys {unknown}"
    missing = [key for key in _PLOT_SPEC_KEYS if key != "ylim" and key not in raw]
    if missing:
        return f"missing keys {missing}"
    error = errors[0]
    loc = error["loc"]
    if error["type"] == "value_error" and len(loc) == 1:
        return str(error["ctx"]["error"])
    key = str(loc[0])
    if key == "filename":
        return f"'filename' must be a .png file name, got {raw['filename']!r}"
    if key in ("title", "ylabel"):
        return f"{key!r} must be a string, got {raw[key]!r}"
    if key in ("hours_before", "ylim"):
        return f"'{key}' must be a list of finite numbers"
    if len(loc) == 1:
        return "'series' must be a non-empty list"
    item = error["input"] if len(loc) == 2 else _series_item(raw, loc[1])
    if len(loc) == 2 or not isinstance(item, dict) or set(item) != {"label", "values"}:
        return f"each series is a mapping with exactly 'label' and 'values', got {item!r}"
    if loc[2] == "label":
        return f"series 'label' must be a non-empty string, got {item['label']!r}"
    return f"series {item['label']!r} 'values' must be a list of finite numbers"


def _series_item(raw: dict[object, object], index: object) -> object:
    """The item of a plot spec's ``series`` list at ``index``."""
    series = raw["series"]
    return series[index] if isinstance(series, list) and isinstance(index, int) else None


def matches_signature(head: bytes, signature: tuple[int, bytes]) -> bool:
    """Whether ``head`` carries ``signature`` (an ``(offset, bytes)`` pair)."""
    offset, magic = signature
    return head[offset : offset + len(magic)] == magic
