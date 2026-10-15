import pickle
from typing import Any, cast

import polars as pl


def serialize_dataframe(df: pl.DataFrame) -> bytes:
    """Serialize a Polars DataFrame to bytes."""
    return cast(bytes, df.write_ipc(None).getvalue())


def deserialize_dataframe(data: bytes) -> pl.DataFrame:
    """Deserialize a Polars DataFrame from bytes."""
    return pl.read_ipc(data)


def serialize_obj(obj: Any) -> bytes:
    """Serialize a Python object to bytes using pickle."""
    return pickle.dumps(obj)


def deserialize_obj(data: bytes) -> Any:
    """Deserialize a Python object from bytes using pickle."""
    return pickle.loads(data)
