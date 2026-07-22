"""Mutating operations on the hierarchical database.

Adds, edits, and deletes tree nodes; reads and edits instance-level expansion
configs; and reports the impact of a deletion. Every mutation navigates to the
target via ``_navigate_for_write``, then rebuilds the flat channel map (via
:class:`_HierarchicalNamingMixin`) and persists to disk.
"""

from ..core.base_database import DatabaseWriteError
from ._hierarchical_naming import _HierarchicalNamingMixin


class _HierarchicalWriteMixin(_HierarchicalNamingMixin):
    """Node/expansion mutation, impact analysis, and persistence."""

    @property
    def _level_types(self) -> dict[str, str]:
        """Map level names to their type (tree/instances)."""
        return {name: cfg["type"] for name, cfg in self.hierarchy_config["levels"].items()}

    def _navigate_for_write(self, selections: dict[str, str]) -> dict:
        """Navigate the hierarchical tree to a parent node for write operations.

        Instance-type levels are traversed by entering the child that owns
        an ``_expansion`` key (the selection value is ignored as a tree key).

        Args:
            selections: Mapping of level_name -> selected_value for parent path.

        Returns:
            The target subtree dict.

        Raises:
            DatabaseWriteError: If navigation path is invalid.
        """
        node = self.tree
        for lvl in self.hierarchy_levels:
            val = selections.get(lvl)
            if val is None:
                break
            ltype = self._level_types.get(lvl, "tree")
            if ltype == "instances":
                found = False
                for key, child in node.items():
                    if key.startswith("_") or not isinstance(child, dict):
                        continue
                    if "_expansion" in child:
                        node = child
                        found = True
                        break
                if not found:
                    raise DatabaseWriteError(
                        f"No expansion container found at instance level '{lvl}'",
                        "invalid_path",
                    )
            else:
                if val not in node:
                    raise DatabaseWriteError(
                        f"Node '{val}' not found at level '{lvl}'", "not_found"
                    )
                node = node[val]
                if not isinstance(node, dict):
                    raise DatabaseWriteError(
                        f"Node '{val}' at level '{lvl}' is not navigable",
                        "invalid_path",
                    )
        return node

    @staticmethod
    def _normalize_expansion(exp: dict) -> dict:
        """Normalize expansion keys for the API layer.

        The database engine uses underscore-prefixed keys (``_pattern``, ``_range``).
        The API/frontend uses clean names (``pattern``, ``range``).
        """
        return {
            "pattern": exp.get("_pattern") or exp.get("pattern", ""),
            "range": exp.get("_range") or exp.get("range", [1, 1]),
        }

    @classmethod
    def _collect_impact(
        cls,
        node: dict,
        remaining_levels: list[str],
        level_types: dict[str, str],
    ) -> dict[str, int]:
        """Walk the tree and count descendants at each remaining hierarchy level."""
        counts: dict[str, int] = {}
        if not remaining_levels or not isinstance(node, dict):
            return counts

        current_level = remaining_levels[0]
        rest = remaining_levels[1:]
        ltype = level_types.get(current_level, "tree")

        if ltype == "instances":
            for _k, child in node.items():
                if not isinstance(child, dict) or _k.startswith("_"):
                    continue
                if "_expansion" in child:
                    inst_count = cls._expansion_instance_count(child["_expansion"])
                    counts[current_level] = inst_count
                    sub = cls._collect_impact(child, rest, level_types)
                    for lvl, c in sub.items():
                        counts[lvl] = c * inst_count
                    break
        else:
            children = cls._child_keys(node)
            counts[current_level] = len(children)
            for ck in children:
                child = node[ck]
                if isinstance(child, dict):
                    sub = cls._collect_impact(child, rest, level_types)
                    for lvl, c in sub.items():
                        counts[lvl] = counts.get(lvl, 0) + c

        return counts

    def add_node(
        self,
        level: str,
        parent_selections: dict[str, str],
        name: str,
        description: str = "",
        channel_part: str | None = None,
    ) -> dict:
        """Add a new node at a given hierarchy level.

        Args:
            level: The level name where the node will be added.
            parent_selections: Dict of level->value for parent navigation.
            name: Name of the new node.
            description: Optional description.
            channel_part: Optional channel part override.

        Returns:
            Success dict with node info.
        """
        parent = self._navigate_for_write(parent_selections)

        if name in parent:
            raise DatabaseWriteError(f"Node '{name}' already exists at this level", "duplicate")

        new_node: dict = {}
        if description:
            new_node["_description"] = description
        if channel_part:
            new_node["_channel_part"] = channel_part

        parent[name] = new_node

        self._commit()

        return {"success": True, "name": name, "level": level}

    def edit_node(
        self,
        level: str,
        selections: dict[str, str],
        old_name: str,
        new_name: str | None = None,
        description: str | None = None,
    ) -> dict:
        """Edit a node's name and/or description at a hierarchy level.

        Args:
            level: The level where the node exists.
            selections: Parent selections (excluding the target level).
            old_name: Current name of the node.
            new_name: New name (None to keep current).
            description: New description (None to leave unchanged, "" to clear).

        Returns:
            Success dict.
        """
        parent = self._navigate_for_write(selections)

        if old_name not in parent:
            raise DatabaseWriteError(f"Node '{old_name}' not found", "not_found")

        effective_name = new_name if new_name and new_name != old_name else old_name

        if effective_name != old_name:
            if effective_name in parent:
                raise DatabaseWriteError(f"Node '{effective_name}' already exists", "duplicate")
            parent[effective_name] = parent.pop(old_name)

        if description is not None:
            node = parent[effective_name]
            if isinstance(node, dict):
                if description:
                    node["_description"] = description
                else:
                    node.pop("_description", None)

        self._commit()

        return {"success": True, "old_name": old_name, "new_name": effective_name}

    def delete_node(
        self,
        level: str,
        selections: dict[str, str],
        name: str,
    ) -> dict:
        """Delete a node (and all descendants) at a hierarchy level.

        Args:
            level: The level where the node exists.
            selections: Parent selections (excluding the target level).
            name: Name of the node to delete.

        Returns:
            Success dict with affected channel count.
        """
        parent = self._navigate_for_write(selections)

        if name not in parent:
            raise DatabaseWriteError(f"Node '{name}' not found", "not_found")

        affected = self._count_channels(parent[name])
        del parent[name]

        self._commit()

        return {"success": True, "name": name, "affected_channels": affected}

    def get_expansion(
        self,
        level: str,
        selections: dict[str, str],
    ) -> dict:
        """Read the current expansion config for an instance-type level.

        Args:
            level: The instance-type level name.
            selections: Parent selections for navigation.

        Returns:
            Dict with ``expansion`` key containing the expansion config.
        """
        parent_selections = {}
        for lvl in self.hierarchy_levels:
            if lvl == level:
                break
            val = selections.get(lvl)
            if val is not None:
                parent_selections[lvl] = val

        parent = self._navigate_for_write(parent_selections)

        for key, child in parent.items():
            if key.startswith("_") or not isinstance(child, dict):
                continue
            if "_expansion" in child:
                return {"expansion": self._normalize_expansion(child["_expansion"])}

        # Instance-type levels without a container yet get a default
        if self._level_types.get(level) == "instances":
            return {"expansion": {"pattern": "", "range": [1, 1]}}

        raise DatabaseWriteError(
            f"No expansion container found at level '{level}'",
            "not_found",
        )

    def edit_expansion(
        self,
        level: str,
        selections: dict[str, str],
        pattern: str | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
    ) -> dict:
        """Edit the expansion config for an instance-type level.

        Args:
            level: The instance-type level name.
            selections: Parent selections.
            pattern: New pattern string (None leaves unchanged).
            range_start: New range start (None leaves unchanged).
            range_end: New range end (None leaves unchanged).

        Returns:
            Success dict with updated expansion info.
        """
        parent_selections = {}
        for lvl in self.hierarchy_levels:
            if lvl == level:
                break
            val = selections.get(lvl)
            if val is not None:
                parent_selections[lvl] = val

        parent = self._navigate_for_write(parent_selections)

        # Find the child that has _expansion
        expansion_container = None
        for key, child in parent.items():
            if key.startswith("_") or not isinstance(child, dict):
                continue
            if "_expansion" in child:
                expansion_container = child
                break

        if expansion_container is None:
            if self._level_types.get(level) != "instances":
                raise DatabaseWriteError(
                    f"No expansion container found at level '{level}'",
                    "not_found",
                )
            container_key = level.upper()
            parent[container_key] = {
                "_expansion": {"_type": "range", "_pattern": "", "_range": [1, 1]},
            }
            expansion_container = parent[container_key]

        expansion = expansion_container["_expansion"]

        if pattern is not None:
            expansion["_pattern"] = pattern
            expansion.pop("pattern", None)
        if range_start is not None and range_end is not None:
            expansion["_range"] = [range_start, range_end]
            expansion.pop("range", None)
        elif range_start is not None:
            r = expansion.get("_range") or expansion.get("range", [1, 1])
            expansion["_range"] = [range_start, r[1] if len(r) > 1 else range_start]
            expansion.pop("range", None)
        elif range_end is not None:
            r = expansion.get("_range") or expansion.get("range", [1, 1])
            expansion["_range"] = [r[0] if r else 1, range_end]
            expansion.pop("range", None)

        if "_type" not in expansion:
            expansion["_type"] = "range"

        self._commit()

        return {
            "success": True,
            "level": level,
            "expansion": self._normalize_expansion(expansion),
        }

    def count_descendants(
        self,
        level: str,
        selections: dict[str, str],
        name: str,
    ) -> dict[str, int]:
        """Compute the impact of deleting a node, broken down by hierarchy level.

        Returns:
            Dict mapping each descendant level name to its count, plus
            ``"channels"`` for the total expanded leaf channel count.
        """
        parent = self._navigate_for_write(selections)

        if name not in parent:
            raise DatabaseWriteError(f"Node '{name}' not found", "not_found")

        try:
            lvl_idx = self.hierarchy_levels.index(level)
        except ValueError:
            lvl_idx = -1
        remaining = self.hierarchy_levels[lvl_idx + 1 :] if lvl_idx >= 0 else self.hierarchy_levels

        impact = self._collect_impact(parent[name], remaining, self._level_types)
        impact["channels"] = self._count_channels(parent[name])
        return impact
