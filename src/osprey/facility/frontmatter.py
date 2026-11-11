"""Markdown with a YAML frontmatter block, split into its two parts.

The pages under ``data/facility/knowledge`` open with a frontmatter block
delimited by ``---`` lines, exactly like Jekyll / Hugo front matter. The
facility build reads them for their links; Knowledge reads them as OKF
documents on top of this split.
"""

from __future__ import annotations

from typing import Any

import yaml

__all__ = ["FrontmatterError", "split_frontmatter"]

_DELIM = "---"


class FrontmatterError(ValueError):
    """A frontmatter block that does not parse."""


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a page into its frontmatter mapping and its Markdown body.

    Text without a leading ``---`` line is body only, with empty frontmatter.
    The body drops one leading blank line after the closing delimiter.

    Args:
        text: Raw page text.

    Returns:
        The frontmatter mapping and the body.

    Raises:
        FrontmatterError: The block is unterminated, is not valid YAML, or is
            not a YAML mapping.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _DELIM:
        return {}, text

    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _DELIM:
            end_idx = i
            break
    if end_idx is None:
        raise FrontmatterError("Unterminated YAML frontmatter block")

    fm_text = "\n".join(lines[1:end_idx])
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"Invalid YAML in frontmatter: {exc}") from exc
    if not isinstance(fm, dict):
        raise FrontmatterError("Frontmatter must be a YAML mapping")

    body = "\n".join(lines[end_idx + 1 :])
    return fm, body.removeprefix("\n")
