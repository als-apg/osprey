"""
Hierarchical Channel Database

Loads and navigates hierarchical channel structures for iterative LLM-based refinement.
Supports flexible hierarchy with arbitrary mixing of:
- Tree levels (semantic categories)
- Instance levels (numbered/patterned expansions)

The implementation is split by concern across sibling ``_hierarchical_*`` modules
(loading/validation, channel-name construction, query/navigation, tree preview,
and writes). :class:`HierarchicalChannelDatabase` composes them into the public
class; every method of the ``BaseDatabase`` contract plus the extra public API
remains available directly on this class.
"""

from ..core.base_database import BaseDatabase
from ._hierarchical_loading import _HierarchicalLoadingMixin
from ._hierarchical_preview import _HierarchicalPreviewMixin
from ._hierarchical_query import _HierarchicalQueryMixin
from ._hierarchical_write import _HierarchicalWriteMixin

__all__ = ["HierarchicalChannelDatabase"]


class HierarchicalChannelDatabase(
    _HierarchicalLoadingMixin,
    _HierarchicalPreviewMixin,
    _HierarchicalWriteMixin,
    _HierarchicalQueryMixin,
    BaseDatabase,
):
    """Database for hierarchical channel naming schemes.

    Supports flexible hierarchy with arbitrary mixing of tree levels
    (semantic categories) and instance levels (numbered/patterned expansions).

    The behavior lives in the concern-specific mixins:

    - :class:`~osprey.services.channel_finder.databases._hierarchical_loading._HierarchicalLoadingMixin`
      -- parsing, schema validation, and serialization.
    - :class:`~osprey.services.channel_finder.databases._hierarchical_naming._HierarchicalNamingMixin`
      -- channel-name construction and flat channel-map expansion.
    - :class:`~osprey.services.channel_finder.databases._hierarchical_query._HierarchicalQueryMixin`
      -- level-option enumeration, navigation, and channel lookup.
    - :class:`~osprey.services.channel_finder.databases._hierarchical_preview._HierarchicalPreviewMixin`
      -- compact text previews of the tree.
    - :class:`~osprey.services.channel_finder.databases._hierarchical_write._HierarchicalWriteMixin`
      -- node/expansion mutation and persistence.
    """

    def __init__(self, db_path: str):
        super().__init__(db_path)
