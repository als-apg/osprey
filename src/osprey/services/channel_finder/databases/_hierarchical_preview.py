"""Human-readable tree previews of the hierarchical database.

Renders compact, indented overviews of the database structure - either the full
tree or the subtree at a given navigation position - to help an LLM understand
where channels live before drilling in. Builds on the navigation provided by
:class:`_HierarchicalQueryMixin`.
"""

from typing import Any

from ._hierarchical_query import _HierarchicalQueryMixin


class _HierarchicalPreviewMixin(_HierarchicalQueryMixin):
    """Compact text previews of the tree and its subtrees."""

    def generate_tree_preview(self, max_depth: int = 3, max_children: int = 5) -> str:
        """
        Generate a compact text overview of the full database structure.

        Shows hierarchy levels, category names, and channel counts to help
        LLMs understand where channels live before navigating.

        Args:
            max_depth: Maximum tree depth to render (0-indexed from root)
            max_children: Maximum children to show per node before truncating

        Returns:
            Indented text preview of the database structure
        """
        if not hasattr(self, "_tree_preview_cache"):
            self._tree_preview_cache = {}

        cache_key = (max_depth, max_children)
        if cache_key in self._tree_preview_cache:
            return self._tree_preview_cache[cache_key]

        lines: list[str] = []
        lines.append(f"Database Structure ({len(self.channel_map)} total channels)")
        lines.append(f"Hierarchy: {' → '.join(self.hierarchy_levels)}")
        lines.append(f"Naming: {self.naming_pattern}")
        lines.append("")

        self._render_tree_node(
            node=self.tree,
            depth=0,
            max_depth=max_depth,
            max_children=max_children,
            lines=lines,
            level_idx=0,
        )

        preview = "\n".join(lines)
        self._tree_preview_cache[cache_key] = preview
        return preview

    def generate_subtree_preview(
        self,
        previous_selections: dict[str, Any],
        max_depth: int = 2,
        max_children: int = 5,
    ) -> str:
        """
        Generate a preview from the current navigation position.

        Args:
            previous_selections: Selections made so far at previous levels
            max_depth: Maximum depth to render below current position
            max_children: Maximum children to show per node

        Returns:
            Indented text preview of the subtree, or empty string if position invalid
        """
        # Determine which level we're at
        current_level_idx = 0
        for i, level in enumerate(self.hierarchy_levels):
            if level not in previous_selections:
                current_level_idx = i
                break
        else:
            current_level_idx = len(self.hierarchy_levels)

        # Navigate to position
        target_level = (
            self.hierarchy_levels[current_level_idx]
            if current_level_idx < len(self.hierarchy_levels)
            else None
        )
        if target_level is None:
            return ""

        node = self._navigate_to_node(target_level, previous_selections)
        if not node:
            return ""

        # Build path string
        path_parts = [f"{k}={v}" for k, v in previous_selections.items()]
        path_str = " → ".join(path_parts) if path_parts else "ROOT"

        lines: list[str] = []
        lines.append(f"Subtree at: {path_str}")
        channel_count = self._count_channels(node)
        lines.append(f"({channel_count} channels below this point)")
        lines.append("")

        self._render_tree_node(
            node=node,
            depth=0,
            max_depth=max_depth,
            max_children=max_children,
            lines=lines,
            level_idx=current_level_idx,
        )

        return "\n".join(lines)

    def _render_tree_node(
        self,
        node: dict,
        depth: int,
        max_depth: int,
        max_children: int,
        lines: list[str],
        level_idx: int,
    ) -> None:
        """
        Recursively render a tree node as indented text.

        Args:
            node: Current tree node dict
            depth: Current rendering depth (for indentation)
            max_depth: Maximum depth to render
            max_children: Maximum children per node
            lines: Output line buffer
            level_idx: Current hierarchy level index
        """
        if depth >= max_depth:
            return

        indent = "  " * depth

        # Determine level info
        level_name = (
            self.hierarchy_levels[level_idx] if level_idx < len(self.hierarchy_levels) else None
        )
        level_config = self.hierarchy_config["levels"].get(level_name, {}) if level_name else {}
        level_type = level_config.get("type", "tree")

        if level_type == "instances":
            # Find the instance container
            for key, value in node.items():
                if key.upper() == level_name.upper() and isinstance(value, dict):
                    if "_expansion" in value:
                        expanded = self._expand_instances(value["_expansion"])
                        count = len(expanded)
                        lines.append(
                            f"{indent}[{level_name}] {count} instances "
                            f"({expanded[0]['name']}..{expanded[-1]['name']})"
                        )
                        # Recurse into the container's children at next level
                        if depth + 1 < max_depth:
                            self._render_tree_node(
                                node=value,
                                depth=depth + 1,
                                max_depth=max_depth,
                                max_children=max_children,
                                lines=lines,
                                level_idx=level_idx + 1,
                            )
                    break
        else:
            # Tree level: show children
            children = self._child_keys(node)
            shown = children[:max_children]
            truncated_count = len(children) - len(shown)

            for child_key in shown:
                child_node = node[child_key]
                if not isinstance(child_node, dict):
                    continue

                channel_count = self._count_channels(child_node)
                desc = child_node.get("_description", "")
                desc_str = f" - {desc}" if desc else ""

                # Check for inline expansion
                if "_expansion" in child_node:
                    exp_count = self._expansion_instance_count(child_node["_expansion"])
                    lines.append(
                        f"{indent}{child_key} ({channel_count} ch, {exp_count} instances){desc_str}"
                    )
                else:
                    lines.append(f"{indent}{child_key} ({channel_count} ch){desc_str}")

                # Recurse
                if depth + 1 < max_depth:
                    self._render_tree_node(
                        node=child_node,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_children=max_children,
                        lines=lines,
                        level_idx=level_idx + 1,
                    )

            if truncated_count > 0:
                lines.append(f"{indent}... and {truncated_count} more")
