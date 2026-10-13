"""Tests for utils/serialization.py — DataFrame and object round-trips."""

from __future__ import annotations

import polars as pl
import pytest

from osprey.services.optimizer.utils.serialization import (
    deserialize_dataframe,
    deserialize_obj,
    serialize_dataframe,
    serialize_obj,
)


class TestDataFrameSerialization:
    @pytest.mark.parametrize(
        ("df", "expected_shape"),
        [
            pytest.param(
                pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.5, 3.5], "c": ["x", "y", "z"]}),
                (3, 3),
                id="basic",
            ),
            pytest.param(pl.DataFrame(), (0, 0), id="empty_no_columns"),
            pytest.param(
                pl.DataFrame(
                    {"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Int64)}
                ),
                (0, 2),
                id="empty_rows",
            ),
        ],
    )
    def test_round_trip(self, df: pl.DataFrame, expected_shape: tuple[int, int]) -> None:
        result = deserialize_dataframe(serialize_dataframe(df))
        assert result.shape == expected_shape
        assert result.schema == df.schema
        assert result.equals(df)


class TestObjectSerialization:
    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param({"key": "value", "nested": {"a": 1}}, id="dict"),
            pytest.param([1, "two", 3.0, None], id="list"),
            pytest.param(None, id="none"),
        ],
    )
    def test_round_trip(self, obj: object) -> None:
        data = serialize_obj(obj)
        assert deserialize_obj(data) == obj
