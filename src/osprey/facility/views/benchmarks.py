"""The benchmarks view: the channel-finder benchmark query file a render carries.

The facility tree ships one scored query set per pipeline family under
``data/facility/benchmarks/cross_paradigm/queries/``: ``in_context_queries.json``
for the ``in_context`` pipeline and ``tree_queries.json`` for every other mode.
A render whose ``config.yml`` states ``channel_finder.benchmark.dataset_path``
carries the one its ``channel_finder.pipeline_mode`` selects, byte-equal, at
that path; ``osprey channel-finder benchmark`` reads it from there.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from osprey.facility.views import PIPELINE_MODE_KEY

if TYPE_CHECKING:
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.views import ViewInputs

__all__ = [
    "BENCHMARK_DATASET_KEY",
    "QUERY_SOURCES",
    "benchmarks_wanted",
    "query_source_name",
    "write_benchmarks_view",
]

#: The rendered config key that names where the render carries the query file,
#: relative to the render root.
BENCHMARK_DATASET_KEY = "channel_finder.benchmark.dataset_path"

#: The query sources' directory, relative to the content root (the facility
#: tree of a single-tree deployment).
QUERY_SOURCES = "benchmarks/cross_paradigm/queries"

#: The pipeline mode whose query set is its own file; every other mode reads the
#: tree set.
_IN_CONTEXT_MODE = "in_context"


def query_source_name(mode: str | None) -> str:
    """The query source file a pipeline mode reads.

    Args:
        mode: The rendered ``channel_finder.pipeline_mode``.

    Returns:
        ``in_context_queries.json`` for ``in_context``, else ``tree_queries.json``.
    """
    return "in_context_queries.json" if mode == _IN_CONTEXT_MODE else "tree_queries.json"


def _channel_finder(config: Mapping[str, Any]) -> Mapping[str, Any]:
    block = config.get("channel_finder")
    return block if isinstance(block, Mapping) else {}


def _dataset_path(config: Mapping[str, Any]) -> str | None:
    benchmark = _channel_finder(config).get("benchmark")
    if not isinstance(benchmark, Mapping):
        return None
    value = benchmark.get("dataset_path")
    return str(value) if value else None


def benchmarks_wanted(inputs: ViewInputs) -> tuple[bool, str]:
    """Whether a render carries the benchmark query file, and what decided it.

    Args:
        inputs: The render's view inputs.

    Returns:
        True with the key when the render states
        ``channel_finder.benchmark.dataset_path`` and the facility ships query
        sources; otherwise False with the key, or the sources' directory when the
        key is stated and the facility ships none.
    """
    if _dataset_path(inputs.rendered_config) is None:
        return False, BENCHMARK_DATASET_KEY
    if not (inputs.content_root / QUERY_SOURCES).is_dir():
        return False, f"data/facility/{QUERY_SOURCES}"
    return True, BENCHMARK_DATASET_KEY


def _refusal(kind: str, record_id: str, detail: str, remedy: str) -> FacilityBuildError:
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.served import CONFIG_SOURCE

    return FacilityBuildError(
        kind, record_id, [CONFIG_SOURCE], remedy, record_kind="path", detail=detail
    )


def write_benchmarks_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Copy the mode's query source to the render's stated dataset path.

    Args:
        root: The render's ``data`` directory; the dataset path is relative to
            its parent, the render root.
        inputs: The render's view inputs.

    Returns:
        The file written.

    Raises:
        FacilityBuildError: ``profile-invalid`` when the dataset path is absolute
            or outside the render's ``data/``; ``view-unsupported`` when the
            facility ships query sources but none for the mode. Both before
            anything is written.
    """
    stated = _dataset_path(inputs.rendered_config)
    relative = PurePosixPath(stated or "")
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:1] != ("data",)
        or len(relative.parts) < 2
    ):
        raise _refusal(
            "profile-invalid",
            BENCHMARK_DATASET_KEY,
            f"`{stated}` is not a path under the render's data/",
            f"set `{BENCHMARK_DATASET_KEY}` to a path under data/, "
            "such as `data/benchmarks/queries.json`",
        )
    mode = _channel_finder(inputs.rendered_config).get("pipeline_mode")
    source = inputs.content_root / QUERY_SOURCES / query_source_name(mode)
    if not source.is_file():
        raise _refusal(
            "view-unsupported",
            PIPELINE_MODE_KEY,
            f"selects {mode} and the facility ships no data/facility/{QUERY_SOURCES}/{source.name}",
            f"add {source.name} to data/facility/{QUERY_SOURCES}/, "
            "or select another channel_finder_mode",
        )
    target = root.parent.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return [target]
