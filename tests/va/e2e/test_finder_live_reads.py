"""SC6 acceptance: every address a channel-finder pipeline surfaces is live over CA.

The Control Assistant preset ships the SAME ~1,228-address namespace in three
interchangeable channel-finder paradigms (``docker/virtual-accelerator/manifest``'s
own build step cross-checks this: ``build_manifest()`` raises
``ParadigmMismatchError`` if the three tier-3 DBs ever disagree on their
expanded address set -- see ``manifest/build.py``). This suite proves that
guarantee holds against a *live* container by expanding each paradigm DB
through its own loader (the same loaders ``manifest/build.py`` uses, so this
exercises the real per-pipeline expansion path rather than re-deriving
addresses from one shared source) and reading a representative slice of
finder-surfaced addresses -- orbit, corrector, RF, vacuum -- back over real
Channel Access.

Reuses ``scripts/va/sweep_check.sweep()`` for the actual batched CA read (same
reachability-sweep logic ``test_full_sweep.py`` already validates at full
manifest scale), so this test is about *which* addresses each pipeline hands
back, not a new CA-read mechanism.
"""

from __future__ import annotations

import sys

import pytest

from tests.va.e2e.conftest import sweep_check

# Representative finder-query categories, keyed by the address-string tokens
# that identify them (``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD`` -- see
# manifest/classify.py for the same convention). Address-string matching
# works uniformly across all three loaders: two of them (in_context,
# middle_layer) return bare address sets with no hierarchy-path metadata, so
# there's no richer structure to filter on than the address itself.
QUERY_CATEGORIES: dict[str, tuple[str, ...]] = {
    "orbit": (":DIAG:BPM:",),
    "corrector": (":MAG:HCM:", ":MAG:VCM:"),
    "rf": (":RF:CAVITY:", ":RF:KLYSTRON:"),
    "vacuum": (":VAC:GAUGE:", ":VAC:ION-PUMP:", ":VAC:VALVE:"),
}

# Cap per category so the live read stays a fast representative slice, not a
# second full-manifest sweep (test_full_sweep.py already owns that scale).
ADDRESSES_PER_CATEGORY = 6

SWEEP_TIMEOUT_S = 30.0


def _representative_addresses(all_addresses: set[str]) -> dict[str, list[str]]:
    """Pick a small, deterministic, sorted slice of ``all_addresses`` per category."""
    picked: dict[str, list[str]] = {}
    for category, tokens in QUERY_CATEGORIES.items():
        matches = sorted(addr for addr in all_addresses if any(tok in addr for tok in tokens))
        assert matches, f"no addresses in this pipeline's expansion match category {category!r}"
        picked[category] = matches[:ADDRESSES_PER_CATEGORY]
    return picked


def _hierarchical_addresses() -> set[str]:
    from manifest import loaders

    return {c.address for c in loaders.load_hierarchical_channels()}


def _in_context_addresses() -> set[str]:
    from manifest import loaders

    return loaders.load_in_context_addresses()


def _middle_layer_addresses() -> set[str]:
    from manifest import loaders

    return loaders.load_middle_layer_addresses()


PIPELINES = {
    "hierarchical": _hierarchical_addresses,
    "in_context": _in_context_addresses,
    "middle_layer": _middle_layer_addresses,
}


class TestFinderLiveReads:
    @pytest.mark.parametrize("pipeline_name", sorted(PIPELINES))
    def test_pipeline_representative_addresses_are_live(self, va_container, pipeline_name):
        all_addresses = PIPELINES[pipeline_name]()
        assert len(all_addresses) > 1000, (
            f"sanity check: {pipeline_name} expansion looked too small "
            f"({len(all_addresses)} addresses)"
        )

        representative = _representative_addresses(all_addresses)
        addresses = sorted({addr for addrs in representative.values() for addr in addrs})

        result = sweep_check.sweep(addresses, timeout=SWEEP_TIMEOUT_S)

        assert not result.missing_connect, (
            f"[{pipeline_name}] {len(result.missing_connect)} finder-returned addresses "
            f"never connected: {result.missing_connect}"
        )
        assert not result.missing_value, (
            f"[{pipeline_name}] {len(result.missing_value)} finder-returned addresses "
            f"connected but returned no value: {result.missing_value}"
        )
        assert result.connected == result.total == len(addresses)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
