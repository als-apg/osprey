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

STORE_VERSION = 2
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
            bucket["_meta"] = {"query": query, "facility": facility}

            # Deduplicate: skip if identical selections already recorded
            sel_repr = json.dumps(_flatten_selections(selections), sort_keys=True)
            for existing in bucket["successes"]:
                if json.dumps(existing["selections"], sort_keys=True) == sel_repr:
                    return

            bucket["successes"].append(
                {
                    "selections": _flatten_selections(selections),
                    "channel_count": channel_count,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

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
            bucket["_meta"] = {"query": query, "facility": facility}

            # Deduplicate: skip if identical (selections, reason) already recorded
            sel_repr = json.dumps(_flatten_selections(partial_selections), sort_keys=True)
            for existing in bucket["failures"]:
                if (
                    json.dumps(existing["partial_selections"], sort_keys=True) == sel_repr
                    and existing["reason"] == reason
                ):
                    return

            bucket["failures"].append(
                {
                    "partial_selections": _flatten_selections(partial_selections),
                    "reason": reason,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

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
    # Read methods
    # ------------------------------------------------------------------

    def list_keys(self) -> list[dict]:
        """Return summary of all feedback entries.

        Returns:
            List of dicts with key, query, facility, success_count,
            failure_count, and last_activity fields.
        """
        self._load()
        result = []
        for key, bucket in self._data.get("entries", {}).items():
            meta = bucket.get("_meta", {})
            successes = bucket.get("successes", [])
            failures = bucket.get("failures", [])

            timestamps = [s.get("timestamp", "") for s in successes] + [
                f.get("timestamp", "") for f in failures
            ]
            last_activity = max(timestamps) if timestamps else ""

            result.append(
                {
                    "key": key,
                    "query": meta.get("query", "(unknown)"),
                    "facility": meta.get("facility", "(unknown)"),
                    "success_count": len(successes),
                    "failure_count": len(failures),
                    "last_activity": last_activity,
                }
            )
        return result

    def get_entry(self, key: str) -> dict | None:
        """Return full bucket for a key, or None if not found."""
        self._load()
        bucket = self._data.get("entries", {}).get(key)
        if bucket is None:
            return None
        return dict(bucket)

    def export_data(self) -> dict:
        """Return deep copy of entire store data for JSON export."""
        self._load()
        return json.loads(json.dumps(self._data))

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def delete_entry(self, key: str) -> bool:
        """Remove entire bucket for a key.

        Returns:
            True if entry was found and deleted, False otherwise.
        """
        with self._locked():
            if key in self._data.get("entries", {}):
                del self._data["entries"][key]
                self._save()
                return True
            return False

    def delete_record(
        self,
        key: str,
        record_type: str,
        index: int,
        expected_timestamp: str,
    ) -> bool:
        """Delete a single success/failure record by index.

        Args:
            key: Bucket key.
            record_type: "successes" or "failures".
            index: Record index within the list.
            expected_timestamp: Stale check — must match record's timestamp.

        Returns:
            True if record was deleted.

        Raises:
            ValueError: If the record's timestamp doesn't match expected_timestamp.
            KeyError: If key or record_type not found or index out of range.
        """
        with self._locked():
            bucket = self._data.get("entries", {}).get(key)
            if bucket is None:
                raise KeyError(f"Key not found: {key}")
            records = bucket.get(record_type)
            if records is None or index >= len(records) or index < 0:
                raise KeyError(f"Record not found: {record_type}[{index}]")

            if records[index].get("timestamp") != expected_timestamp:
                raise ValueError("Stale timestamp — record has been modified")

            records.pop(index)

            # Clean up empty bucket
            if not bucket.get("successes") and not bucket.get("failures"):
                del self._data["entries"][key]

            self._save()
            return True

    def update_record(
        self,
        key: str,
        record_type: str,
        index: int,
        expected_timestamp: str,
        **fields: Any,
    ) -> bool:
        """Update fields on a single success/failure record.

        For successes: selections (dict), channel_count (int).
        For failures: partial_selections (dict), reason (str).

        Args:
            key: Bucket key.
            record_type: "successes" or "failures".
            index: Record index within the list.
            expected_timestamp: Stale check — must match record's timestamp.
            **fields: Fields to update on the record.

        Returns:
            True if record was updated.

        Raises:
            ValueError: If the record's timestamp doesn't match expected_timestamp.
            KeyError: If key or record_type not found or index out of range.
        """
        with self._locked():
            bucket = self._data.get("entries", {}).get(key)
            if bucket is None:
                raise KeyError(f"Key not found: {key}")
            records = bucket.get(record_type)
            if records is None or index >= len(records) or index < 0:
                raise KeyError(f"Record not found: {record_type}[{index}]")

            if records[index].get("timestamp") != expected_timestamp:
                raise ValueError("Stale timestamp — record has been modified")

            for field_name, value in fields.items():
                if value is not None:
                    records[index][field_name] = value

            records[index]["timestamp"] = datetime.now(UTC).isoformat()
            self._save()
            return True

    def add_manual_entry(
        self,
        query: str,
        facility: str,
        entry_type: str = "success",
        selections: dict | None = None,
        channel_count: int = 0,
        reason: str = "",
    ) -> str:
        """Manually add a success or failure entry.

        Delegates to record_success() or record_failure() so existing
        validation (dedup, MAX_ENTRIES_PER_KEY cap) applies automatically.

        Returns:
            The bucket key for the entry.
        """
        key = self._make_key(query, facility)
        if entry_type == "success":
            self.record_success(
                query=query,
                facility=facility,
                selections=selections or {},
                channel_count=channel_count,
            )
        else:
            self.record_failure(
                query=query,
                facility=facility,
                partial_selections=selections or {},
                reason=reason,
            )
        return key

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
                self._migrate()
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load feedback store: {e}")
                self._data = {"version": STORE_VERSION, "entries": {}}
        else:
            self._data = {"version": STORE_VERSION, "entries": {}}

    def _migrate(self) -> None:
        """Migrate store data from older versions to current."""
        version = self._data.get("version", 1)
        if version < 2:
            for bucket in self._data.get("entries", {}).values():
                if "_meta" not in bucket:
                    bucket["_meta"] = {"query": "(unknown)", "facility": "(unknown)"}
            self._data["version"] = STORE_VERSION

    def _save(self) -> None:
        """Save data to file atomically (must be called within _locked context)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp_path.replace(self._path)

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
    """Flatten list selections for consistent storage.

    Single-element lists are unwrapped to scalars (e.g. ``["MAG"]`` → ``"MAG"``).
    Multi-element lists are preserved as-is (e.g. ``["MAG", "QUAD"]`` stays a list).
    Non-list values pass through unchanged.
    """
    flat = {}
    for k, v in selections.items():
        if isinstance(v, list):
            flat[k] = v[0] if len(v) == 1 else v
        else:
            flat[k] = v
    return flat
