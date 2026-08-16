"""Render an ALS logbook JSONL snapshot into a qmd-indexable markdown mirror.

This renderer is *probe-internal*: it exists to produce a corpus of realistic
shape and scale for the scale probe, and deliberately does not share code with
the production ``qmd_export`` enhancement module. The probe measures how qmd
behaves on ~10^5 micro-documents; it does not validate the production exporter.

The layout mirrors the production design so the measurements transfer:
sharded ``YYYY/MM/<percent-encoded entry_id>.md``, with author, timestamp and
source rendered into *body text* rather than frontmatter -- qmd does not index
frontmatter as searchable metadata.

Usage:
    python export_corpus.py --jsonl <path> --out <dir> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

# Matches the production exporter's cap. Recorded here so the probe report can
# state how many real entries actually hit it (at ALS scale: none).
BODY_CAP_BYTES = 256 * 1024
TRUNCATION_MARKER = "\n\n[truncated]\n"

# Control characters are stripped before indexing; newline and tab survive.
_CONTROL_CHARS = {c: None for c in range(0x20) if c not in (0x09, 0x0A, 0x0D)}
_CONTROL_CHARS[0x7F] = None


def sanitize(text: str) -> str:
    """Strip NUL/control characters and coerce to valid UTF-8.

    Args:
        text: Raw text from the snapshot, possibly containing control bytes.

    Returns:
        Text safe to write to a markdown file.
    """
    cleaned = text.encode("utf-8", errors="replace").decode("utf-8")
    return cleaned.translate(_CONTROL_CHARS)


def encode_entry_id(entry_id: str) -> str:
    """Percent-encode an entry id into a lossless, invertible filename.

    Every character outside the unreserved set is encoded, so ``a/b``, ``..``
    and unicode ids all round-trip through the filesystem safely.

    Args:
        entry_id: Raw entry identifier from the source system.

    Returns:
        A filename-safe encoding of ``entry_id``.
    """
    return quote(entry_id, safe="")


def render(entry: dict) -> str:
    """Render one logbook entry to markdown.

    Author, timestamp, category and source are emitted as body text because
    qmd indexes body content, not frontmatter.

    Args:
        entry: One decoded JSONL record.

    Returns:
        The markdown document body.
    """
    subject = sanitize(entry.get("subject") or "").strip()
    details = sanitize(entry.get("details") or "").strip()
    author = sanitize(entry.get("author") or "").strip()
    category = sanitize(entry.get("category") or "").strip()
    level = sanitize(entry.get("level") or "").strip()

    try:
        when = datetime.fromtimestamp(int(entry["timestamp"]), tz=UTC)
        stamp = when.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (KeyError, ValueError, OSError):
        stamp = "unknown"

    parts = [
        f"# {subject or '(no subject)'}",
        "",
        f"Logged by {author or 'unknown'} on {stamp}.",
        f"Category: {category or 'uncategorized'}. Level: {level or 'unknown'}. "
        f"Source: ALS logbook.",
        "",
        details,
    ]
    body = "\n".join(parts)

    encoded = body.encode("utf-8")
    if len(encoded) > BODY_CAP_BYTES:
        body = encoded[:BODY_CAP_BYTES].decode("utf-8", errors="ignore")
        body += TRUNCATION_MARKER
    return body


def shard_path(out: Path, entry: dict) -> Path:
    """Compute the sharded ``YYYY/MM/<encoded id>.md`` path for an entry.

    Args:
        out: Mirror root directory.
        entry: One decoded JSONL record.

    Returns:
        Absolute path the entry's markdown file should occupy.

    Raises:
        ValueError: If the resolved path escapes the mirror root.
    """
    try:
        when = datetime.fromtimestamp(int(entry["timestamp"]), tz=UTC)
        shard = when.strftime("%Y/%m")
    except (KeyError, ValueError, OSError):
        shard = "unknown"

    target = out / shard / f"{encode_entry_id(str(entry['id']))}.md"
    resolved = target.resolve()
    if not resolved.is_relative_to(out.resolve()):
        raise ValueError(f"path escapes mirror root: {entry['id']!r}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 = all entries")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    truncated = 0
    total_bytes = 0

    with args.jsonl.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if args.limit and written >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            try:
                path = shard_path(args.out, entry)
            except (ValueError, KeyError):
                skipped += 1
                continue

            body = render(entry)
            if TRUNCATION_MARKER in body:
                truncated += 1

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            written += 1
            total_bytes += len(body.encode("utf-8"))

            if written % 20000 == 0:
                print(f"  ... {written} entries", file=sys.stderr, flush=True)

    report = {
        "entries_written": written,
        "entries_skipped": skipped,
        "entries_truncated_at_cap": truncated,
        "body_cap_bytes": BODY_CAP_BYTES,
        "total_markdown_bytes": total_bytes,
        "mirror_root": str(args.out),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
