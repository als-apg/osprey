"""The single authority for which plan-parameter fields name read-only devices.

Two independent consumers need the same answer to "is this parameter field a
list of devices that are *read* at every point, rather than *driven* through
them?", and until this module existed each carried its own private spelling of
the rule:

- :func:`~osprey.services.bluesky_bridge.plan_validation._collect_device_names`
  buckets sample-arg device names into settable and readable mocks for the
  dry run, and matched on the substring ``"detect"`` inline;
- :func:`~osprey.services.bluesky_bridge.document_plane.expected_points_from_params`
  decides whether a device list multiplies a scan's expected point count, and
  matched against a three-element frozenset of exact key names.

Those two rules disagreed at the edges, and each failed *open* in its own
direction — a read-only field the rule did not recognize became a driven
device, silently. Naming the rule once makes the disagreement impossible.

**The catalog is open**, which is why the rule is a predicate over spellings
rather than a fixed list. OSPREY's own shipped plans name their read side
``readbacks``, but facilities and the agent both contribute plans, and a
plan authored against upstream bluesky convention will call the same thing
``detectors``, ``dets``, or ``readables``. All of those must keep working;
recognizing them is not a compatibility shim but the point of an open catalog.

This module is deliberately a **leaf**: it imports nothing but the standard
library, and must stay that way. ``plan_validation`` imports nothing else from
this package, and ``document_plane`` reaches ``queue_backend`` (and through it
``bluesky_queueserver_api``) — so any import added here would drag a heavy,
optional-at-some-call-sites dependency into both.
"""

__all__ = ["is_read_only_device_field"]


def is_read_only_device_field(key: str | None) -> bool:
    """Whether *key* names a plan parameter holding read-only devices.

    Read-only devices are read once at every point of a scan; they are never
    stepped through, so their count never multiplies a point total and they
    are backed by the read-only mock during a dry run.

    Args:
        key: A plan parameter's field name, or ``None`` for a value with no
            enclosing field (an unlabeled string leaf in ``sample_args``).

    Returns:
        ``True`` for OSPREY's own ``readbacks`` and for the bluesky-conventional
        ``detectors`` / ``dets`` / ``readables`` a facility-authored plan may
        use; ``False`` for driven fields (``correctors``, ``setpoints``,
        ``axes``, …) and for ``None``.
    """
    if key is None:
        return False
    low = key.lower()
    return low == "dets" or "detect" in low or "readback" in low or "readable" in low
