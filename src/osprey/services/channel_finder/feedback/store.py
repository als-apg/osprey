"""
Feedback Store for Hierarchical Channel Finder

Records successful and failed navigation paths for identical future queries.
File-backed JSON with cross-process locking via fcntl.flock.
"""

import fcntl
import hashlib
import json
import logging
import re
import string
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STORE_VERSION = 1
MAX_ENTRIES_PER_KEY = 10


class FeedbackStore:
    """
    Records navigation outcomes keyed by normalized (query, facility) pairs.

    Successful paths are injected as hints in future prompts so the LLM
    can resolve identical queries faster. Failed paths help avoid
    repeating dead ends.

    File format::

        {
            "version": 1,
            "entries": {
                "<sha256_key>": {
                    "successes": [
                        {
                            "selections": {"system": "MAG", ...},
                            "channel_count": 42,
                            "timestamp": "2026-02-23T..."
                        }
                    ],
                    "failures": [
                        {
                            "partial_selections": {"system": "MAG"},
                            "reason": "no options at family level",
                            "timestamp": "2026-02-23T..."
                        }
                    ]
                }
            }
        }
    """

    def __init__(self, store_path: str | Path) -> None:
        self._path = Path(store_path)
        self._lock_file = self._path.with_suffix(".lock")
        self._data: dict[str, Any] = {"version": STORE_VERSION, "entries": {}}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_success(
        self,
        query: str,
        facility: str,
        selections: dict[str, Any],
        channel_count: int,
    ) -> None:
        """Record a successful navigation path."""
        key = self._make_key(query, facility)

        with self._locked():
            bucket = self._data["entries"].setdefault(key, {"successes": [], "failures": []})

            # Deduplicate: skip if identical selections already recorded
            sel_repr = str(sorted(_flatten_selections(selections).items()))
            for existing in bucket["successes"]:
                if str(sorted(_flatten_selections(existing["selections"]).items())) == sel_repr:
                    return

            bucket["successes"].append({
                "selections": _flatten_selections(selections),
                "channel_count": channel_count,
                "timestamp": datetime.now(UTC).isoformat(),
            })

            # Cap at MAX_ENTRIES_PER_KEY
            if len(bucket["successes"]) > MAX_ENTRIES_PER_KEY:
                bucket["successes"] = bucket["successes"][-MAX_ENTRIES_PER_KEY:]

            self._save()

    def record_failure(
        self,
        query: str,
        facility: str,
        partial_selections: dict[str, Any],
        reason: str,
    ) -> None:
        """Record a failed navigation path."""
        key = self._make_key(query, facility)

        with self._locked():
            bucket = self._data["entries"].setdefault(key, {"successes": [], "failures": []})

            bucket["failures"].append({
                "partial_selections": _flatten_selections(partial_selections),
                "reason": reason,
                "timestamp": datetime.now(UTC).isoformat(),
            })

            # Cap at MAX_ENTRIES_PER_KEY
            if len(bucket["failures"]) > MAX_ENTRIES_PER_KEY:
                bucket["failures"] = bucket["failures"][-MAX_ENTRIES_PER_KEY:]

            self._save()

    def get_hints(
        self,
        query: str,
        facility: str,
        max_hints: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve prior successful navigation paths as hints.

        Returns:
            List of dicts with ``selections`` and ``channel_count`` keys,
            most recent first.
        """
        key = self._make_key(query, facility)
        self._load()
        bucket = self._data["entries"].get(key, {})
        successes = bucket.get("successes", [])

        # Return most recent first, capped
        return [
            {"selections": s["selections"], "channel_count": s["channel_count"]}
            for s in reversed(successes)
        ][:max_hints]

    def clear(self) -> None:
        """Wipe all stored feedback data."""
        with self._locked():
            self._data = {"version": STORE_VERSION, "entries": {}}
            self._save()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize a query for consistent matching."""
        q = query.lower().strip()
        q = q.translate(str.maketrans("", "", string.punctuation))
        q = re.sub(r"\s+", " ", q)
        return q

    @staticmethod
    def _make_key(query: str, facility: str) -> str:
        """Generate SHA-256 key from normalized facility::query."""
        normalized = FeedbackStore._normalize_query(query)
        raw = f"{facility.lower().strip()}::{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # File I/O with locking
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load data from file (no lock required for reads)."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load feedback store: {e}")
                self._data = {"version": STORE_VERSION, "entries": {}}
        else:
            self._data = {"version": STORE_VERSION, "entries": {}}

    def _save(self) -> None:
        """Save data to file (must be called within _locked context)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    @contextmanager
    def _locked(self):
        """Acquire exclusive file lock for writes."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(self._lock_file, "w")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
            self._load()  # Always reload under lock
            yield
        finally:
            fd.close()


def _flatten_selections(selections: dict[str, Any]) -> dict[str, Any]:
    """Flatten list selections to strings for consistent storage."""
    flat = {}
    for k, v in selections.items():
        if isinstance(v, list):
            flat[k] = v[0] if len(v) == 1 else v
        else:
            flat[k] = v
    return flat
