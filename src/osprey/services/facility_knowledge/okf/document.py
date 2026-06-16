"""OKF document model: parse, serialize, and validate facility knowledge documents.

An OKF document is a Markdown file with a YAML frontmatter block, following the
OSPREY Knowledge Framework specification (OKF §9).  The frontmatter is delimited
by ``---`` lines, exactly like Jekyll / Hugo front matter.

Example document::

    ---
    type: facility_overview
    title: ALS Accelerator Complex
    description: High-level overview of the Advanced Light Source accelerator systems.
    ---

    # ALS Accelerator Complex

    The Advanced Light Source (ALS) is a ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

_FRONTMATTER_DELIM = "---"

# Validation levels supported by :py:meth:`OKFDocument.validate`.
_CONSUMING_REQUIRED: tuple[str, ...] = ("type",)
_AUTHORING_REQUIRED: tuple[str, ...] = ("type", "title", "description")

ValidationLevel = Literal["consuming", "authoring"]


class OKFDocumentError(ValueError):
    """Raised when an OKF document is malformed or fails validation.

    Args:
        message: Human-readable description of the problem.
    """


@dataclass
class OKFDocument:
    """An OKF knowledge document with parsed frontmatter and Markdown body.

    Args:
        frontmatter: Parsed YAML mapping from the document header.
        body: The Markdown body text that follows the frontmatter block.
    """

    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, text: str) -> OKFDocument:
        """Parse an OKF document string into an :class:`OKFDocument`.

        Documents without a leading ``---`` delimiter are treated as body-only
        (frontmatter is empty).  Documents that open a frontmatter block but
        never close it raise :class:`OKFDocumentError`.

        Args:
            text: Raw document text (UTF-8 string).

        Returns:
            Parsed :class:`OKFDocument` instance.

        Raises:
            OKFDocumentError: If the frontmatter block is unterminated or
                contains non-mapping YAML.
        """
        lines = text.splitlines()
        if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
            return cls(frontmatter={}, body=text)

        end_idx: int | None = None
        for i in range(1, len(lines)):
            if lines[i].strip() == _FRONTMATTER_DELIM:
                end_idx = i
                break
        if end_idx is None:
            raise OKFDocumentError("Unterminated YAML frontmatter block")

        fm_text = "\n".join(lines[1:end_idx])
        try:
            fm = yaml.safe_load(fm_text) or {}
        except yaml.YAMLError as exc:
            raise OKFDocumentError(f"Invalid YAML in frontmatter: {exc}") from exc
        if not isinstance(fm, dict):
            raise OKFDocumentError("Frontmatter must be a YAML mapping")

        body = "\n".join(lines[end_idx + 1 :])
        if body.startswith("\n"):
            body = body[1:]
        return cls(frontmatter=fm, body=body)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serialize this document back to a string with YAML frontmatter.

        Returns:
            Full document text, suitable for writing to a ``.md`` file.
        """
        fm_text = yaml.safe_dump(self.frontmatter, sort_keys=False, allow_unicode=True).rstrip()
        body = self.body if self.body.endswith("\n") else self.body + "\n"
        return f"{_FRONTMATTER_DELIM}\n{fm_text}\n{_FRONTMATTER_DELIM}\n\n{body}"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, level: ValidationLevel = "authoring") -> None:
        """Validate this document's frontmatter against the specified OKF level.

        Two named levels are supported (OKF §9):

        * ``"consuming"`` — tolerant reader check: only a non-empty ``type``
          field is required.  Use this when ingesting documents from external
          or untrusted sources.
        * ``"authoring"`` — strict author check: ``type``, ``title``, and
          ``description`` must all be present and non-empty.  ``timestamp`` is
          **not** required, so freshly-seeded stub documents remain valid.

        Args:
            level: Validation strictness level — ``"consuming"`` or
                ``"authoring"``.

        Raises:
            OKFDocumentError: If required frontmatter keys are absent or empty.
            ValueError: If *level* is not a recognised validation level.
        """
        if level == "consuming":
            required = _CONSUMING_REQUIRED
        elif level == "authoring":
            required = _AUTHORING_REQUIRED
        else:
            raise ValueError(
                f"Unknown validation level: {level!r}. Expected 'consuming' or 'authoring'."
            )

        missing = [k for k in required if not self.frontmatter.get(k)]
        if missing:
            raise OKFDocumentError(
                f"Missing required frontmatter keys for level={level!r}: {', '.join(missing)}"
            )
