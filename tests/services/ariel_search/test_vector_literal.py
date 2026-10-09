"""Unit tests for the pgvector text literal."""

from __future__ import annotations

import math

import pytest

from osprey.services.ariel_search.database.vector_literal import vector_literal


class TestVectorLiteral:
    def test_brackets_and_commas_without_spaces(self) -> None:
        assert vector_literal([0.5, -1.0, 2.0]) == "[0.5,-1.0,2.0]"

    def test_matches_the_repository_idiom_for_floats(self) -> None:
        vec = [0.1, 0.2, -0.30000000000000004, 1e-07]
        assert vector_literal(vec) == "[" + ",".join(str(x) for x in vec) + "]"

    def test_ints_are_written_as_floats(self) -> None:
        assert vector_literal([1, 0, -2]) == "[1.0,0.0,-2.0]"

    def test_components_round_trip_exactly(self) -> None:
        vec = [1 / 3, math.pi, -2.5e-12]
        parsed = [float(p) for p in vector_literal(vec)[1:-1].split(",")]
        assert parsed == vec

    def test_accepts_any_iterable(self) -> None:
        assert vector_literal(x / 2 for x in range(3)) == "[0.0,0.5,1.0]"

    def test_numpy_scalars(self) -> None:
        np = pytest.importorskip("numpy")
        assert vector_literal(np.array([0.25, -0.5], dtype=np.float32)) == "[0.25,-0.5]"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_component_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="not finite"):
            vector_literal([0.0, bad])

    def test_empty_vector_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            vector_literal([])

    @pytest.mark.parametrize("bad", [3.5e38, -1e300])
    def test_component_beyond_single_precision_is_rejected(self, bad: float) -> None:
        """pgvector stores float32, so a larger finite float would fail in SQL."""
        with pytest.raises(ValueError, match="single-precision"):
            vector_literal([0.0, bad])

    def test_largest_single_precision_component_is_accepted(self) -> None:
        assert vector_literal([3.4028234663852886e38]) == "[3.4028234663852886e+38]"
