"""The control-context record, as the posture and target route suites read and fake it.

:func:`only_alive` decides which pids the record's readers believe are running,
and :func:`read_record` reads the record past the reader's cache. The
throwaway agent-data root those suites write the record under, and the audit
ledger they count, are the ``agent_data_root`` and ``ledger`` fixtures of this
directory's conftest.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from osprey_connectors import control_context, posture_store


@contextmanager
def only_alive(*pids: int) -> Iterator[None]:
    """Report exactly *pids* as running processes.

    One patch reaches every reader in the request: ``control_context`` owns the
    predicate, and ``target_state.is_process_alive`` — which the route hands to
    the report filter and the marker sweep — delegates to it at call time.
    """
    wanted = {int(p) for p in pids}

    def alive(pid: Any) -> bool:
        try:
            return int(pid) in wanted
        except (TypeError, ValueError):
            return False

    with patch.object(control_context, "is_process_alive", side_effect=alive):
        yield


def read_record() -> Any:
    """The control-context record as it is on disk now, not as last cached.

    The posture store's memo is dropped too, so a verdict read after it is
    taken against the same file.
    """
    control_context.invalidate_cache()
    posture_store.invalidate_cache()
    return control_context.read_record()
