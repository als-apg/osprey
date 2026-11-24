"""Operator screens generated from a built facility and its rule tables.

:mod:`~osprey_panels.document` reads and checks the facility file;
:mod:`~osprey_panels.tables` reads and checks the rule tables;
:mod:`~osprey_panels.model` joins them with the facility file into
the :class:`PanelGraph` every later step reads through;
:mod:`~osprey_panels.health` turns values into states;
:mod:`~osprey_panels.view` and :mod:`~osprey_panels.describe`
say what each screen contains, without layout;
:mod:`~osprey_panels.strip` arranges a place's devices in beam order;
:mod:`~osprey_panels.layout` draws a view as a Screen-IR document with
the symbols and theme roles of :mod:`~osprey_panels.symbols`;
:mod:`~osprey_panels.emit` writes the screens to disk and checks a
written set against the model; :mod:`~osprey_panels.classes` groups the
written screens into classes, each judged by one standard screen.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from osprey_panels.describe import (
    channel_list,
    checkout,
    describe,
    device_card,
    health_rollup,
    place_view,
    rack_elevation,
    set_overview,
    targets,
)
from osprey_panels.document import FacilityDocument, FacilityDocumentError
from osprey_panels.health import (
    State,
    channel_state,
    rollup,
    row_matches,
    rule_channels,
    supported,
    unsupported,
    worst,
)
from osprey_panels.model import (
    Channel,
    Citation,
    Edge,
    Pair,
    PanelGraph,
    Thing,
    load_graph,
)
from osprey_panels.tables import (
    HealthRule,
    PanelTableError,
    PanelTables,
    Pose,
    read_tables,
)
from osprey_panels.view import Target, View

# The version comes from the package's own osprey-panels-v* calendar tags (see
# pyproject.toml): a built distribution answers from its own metadata, and a
# bare source tree — no dist installed, no build hook run — answers honestly
# that it is unbuilt.
try:
    __version__ = _dist_version("osprey-panels")
except PackageNotFoundError:  # pragma: no cover — source tree without an install
    __version__ = "0.0.0+unbuilt"

__all__ = [
    "Channel",
    "Citation",
    "Edge",
    "FacilityDocument",
    "FacilityDocumentError",
    "HealthRule",
    "Pair",
    "PanelGraph",
    "PanelTableError",
    "PanelTables",
    "Pose",
    "State",
    "Target",
    "Thing",
    "View",
    "__version__",
    "channel_state",
    "checkout",
    "describe",
    "channel_list",
    "device_card",
    "health_rollup",
    "load_graph",
    "place_view",
    "rack_elevation",
    "read_tables",
    "rollup",
    "row_matches",
    "rule_channels",
    "set_overview",
    "supported",
    "targets",
    "unsupported",
    "worst",
]
