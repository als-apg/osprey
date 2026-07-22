"""Policy constants and fail-closed validation for dispatch-worker input files.

A dispatch request may carry a batch of caller-supplied files (images, text,
JSON) alongside the prompt. This module is the single source of truth for what
that batch may contain: the mime allowlist, the per-file and total decoded-size
caps, the ingest-file count cap, and filename sanitisation. The request model
and its request-time validation live in ``dispatch_api``; the follow-on
ingestion step imports the same constants and helpers here so both surfaces
agree byte-for-byte on the policy.

The module is framework-free (no FastAPI import): validation raises
:class:`InputFilesError` carrying a machine-readable ``detail`` code, and the
HTTP layer maps that to a 400 response.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Sequence
from typing import Any

# Mime allowlist — the only content types a caller may push as input files.
ALLOWED_MIMES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "text/csv",
        "text/plain",
        "text/markdown",
        "application/json",
    }
)

# Per-file decoded-size caps, keyed on mime family.
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB per image/* file
MAX_TEXT_BYTES = 100 * 1024  # 100 KB per text/* or application/json file

# At most this many files may request ingestion (ingest=True).
MAX_INGEST_FILES = 5

# Ceiling on the combined decoded size of ALL carried files (ingest=True and
# ingest=False alike). The headroom above the ~10 MB working budget exists so a
# request can re-inject prior images (carried ingest=False) without tripping the
# total.
MAX_TOTAL_DECODED_BYTES = 18 * 1024 * 1024  # 18 MB

# Machine-readable HTTP 400 detail codes. A bridge maps these to a non-retryable
# failure class, so the exact strings are load-bearing — do not reword.
DETAIL_CAP_EXCEEDED = "input_files_cap_exceeded"
DETAIL_INVALID = "input_files_invalid"


class InputFilesError(Exception):
    """Raised when a batch of input files violates policy.

    ``detail`` is one of :data:`DETAIL_CAP_EXCEEDED` / :data:`DETAIL_INVALID`;
    the HTTP layer copies it verbatim into the 400 response body. ``message`` is
    a human-readable explanation for logs only.
    """

    def __init__(self, detail: str, message: str = "") -> None:
        super().__init__(message or detail)
        self.detail = detail


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_filename(name: str) -> str:
    """Reduce ``name`` to a safe basename — no directory or traversal parts.

    Strips any path components (either separator), drops NUL bytes, collapses
    every character outside ``[A-Za-z0-9._-]`` to ``_``, and removes leading
    dots so the result can never be ``""``, ``"."``, ``".."`` or a bare dotfile.
    Always returns a non-empty string (``"file"`` as a last resort).
    """
    base = name.replace("\\", "/").split("/")[-1].replace("\x00", "")
    base = _UNSAFE_CHARS.sub("_", base)
    base = base.lstrip(".")
    return base or "file"


def decoded_size(content_b64: str) -> int:
    """Return the decoded byte length of ``content_b64``.

    Raises :class:`InputFilesError` with :data:`DETAIL_INVALID` if the string is
    not valid (strict) base64.
    """
    try:
        raw = base64.b64decode(content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputFilesError(DETAIL_INVALID, f"undecodable base64: {exc}") from exc
    return len(raw)


def validate_input_files(files: Sequence[Any]) -> None:
    """Fail-closed validation of a dispatch request's input files.

    Each element is duck-typed as an input file: attributes ``mime`` (str),
    ``content_b64`` (str), and ``ingest`` (bool). Enforces, in order: the
    ingest-file count cap, then per file the mime allowlist and decoded-size
    cap, then the combined decoded-size ceiling across every file. Raises
    :class:`InputFilesError` on the first violation (the whole request is
    rejected); a valid or empty batch returns ``None``.
    """
    if not files:
        return

    ingest_count = sum(1 for f in files if f.ingest)
    if ingest_count > MAX_INGEST_FILES:
        raise InputFilesError(
            DETAIL_CAP_EXCEEDED,
            f"{ingest_count} ingest files exceeds cap of {MAX_INGEST_FILES}",
        )

    total = 0
    for f in files:
        if f.mime not in ALLOWED_MIMES:
            raise InputFilesError(DETAIL_INVALID, f"mime not allowed: {f.mime!r}")
        size = decoded_size(f.content_b64)
        if f.mime.startswith("image/"):
            if size > MAX_IMAGE_BYTES:
                raise InputFilesError(
                    DETAIL_CAP_EXCEEDED,
                    f"image {size} bytes exceeds per-file cap of {MAX_IMAGE_BYTES}",
                )
        elif size > MAX_TEXT_BYTES:
            raise InputFilesError(
                DETAIL_CAP_EXCEEDED,
                f"text/json {size} bytes exceeds per-file cap of {MAX_TEXT_BYTES}",
            )
        total += size

    if total > MAX_TOTAL_DECODED_BYTES:
        raise InputFilesError(
            DETAIL_CAP_EXCEEDED,
            f"total decoded {total} bytes exceeds cap of {MAX_TOTAL_DECODED_BYTES}",
        )
