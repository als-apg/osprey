"""Execution-metadata timestamps emitted by the wrapper carry a UTC offset.

The wrapper emits *source text* that runs inside the executor child, so the
timestamps it stamps are only observable by running that text. A naive
``isoformat()`` there is unreadable next to the offset-carrying times the
in-process producer writes, and unorderable against any other subsystem's.
"""

from datetime import datetime
from datetime import datetime as _datetime

import pytest

from osprey.services.python_executor.execution.wrapper import ExecutionWrapper


@pytest.mark.unit
def test_start_time_parses_with_tzinfo():
    """``start_time`` in the emitted metadata is offset-aware."""
    namespace: dict = {"_datetime": _datetime}
    exec(compile(ExecutionWrapper()._get_metadata_init(), "<wrapper>", "exec"), namespace)

    start = datetime.fromisoformat(namespace["execution_metadata"]["start_time"])
    assert start.tzinfo is not None
    assert start.utcoffset() is not None


@pytest.mark.unit
def test_no_naive_timestamp_left_in_the_emitted_wrapper():
    """Every ``end_time`` branch stamps an aware time too, not just ``start_time``."""
    source = ExecutionWrapper().create_wrapper("x = 1")

    assert "_datetime.now().isoformat()" not in source
    assert "_datetime.now().astimezone().isoformat()" in source
