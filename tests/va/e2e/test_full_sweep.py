"""SC4 acceptance: the complete namespace-union manifest is live over CA.

Batched bulk reads (one shared connect deadline, never a per-channel serial
loop -- see ``scripts/va/sweep_check.py``) of every address the manifest
defines, against the real running container. Hard wall-clock bound: the full
sweep must finish in under 60s, and zero addresses may fail to connect or
return a value.
"""

from __future__ import annotations

import sys

import pytest

from tests.va.e2e.conftest import sweep_check

FULL_SWEEP_WALL_CLOCK_BUDGET_S = 60.0


class TestFullSweep:
    def test_full_manifest_is_live_over_ca(self, va_container):
        addresses = sweep_check.all_manifest_addresses()
        assert len(addresses) > 1000, (
            f"sanity check: expected the full ~1,290-channel manifest, got {len(addresses)}"
        )

        result = sweep_check.sweep(addresses, timeout=45.0)

        assert result.elapsed_s < FULL_SWEEP_WALL_CLOCK_BUDGET_S, (
            f"full sweep took {result.elapsed_s:.1f}s, over the "
            f"{FULL_SWEEP_WALL_CLOCK_BUDGET_S}s budget"
        )
        assert not result.missing_connect, (
            f"{len(result.missing_connect)} addresses never connected: "
            f"{result.missing_connect[:20]}"
            + (" ..." if len(result.missing_connect) > 20 else "")
        )
        assert not result.missing_value, (
            f"{len(result.missing_value)} addresses connected but returned no value: "
            f"{result.missing_value[:20]}"
            + (" ..." if len(result.missing_value) > 20 else "")
        )
        assert result.connected == result.total == len(addresses)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
