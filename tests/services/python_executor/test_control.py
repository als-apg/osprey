"""Unit tests for :mod:`osprey.services.python_executor.execution.control`.

The module now holds only :class:`ExecutionMode`. Its write-posture report (a
config class and the factory that built it) is gone: no executor gate read it. The cases that pinned it are covered where the answer is
now decided:

- the tier's ``allowed``/``never`` reading, target-blind, and the tier key
  winning over a retired leaf: ``test_tier_writes.py``
  (``test_literal_allowed_arms``, ``test_any_other_value_of_the_new_key_is_never``,
  ``test_new_key_decides_when_both_are_present``) and the ``tier-*`` rows of the
  shared write-verdict case table;
- an unnormalized render: ``test_tier_writes.py``
  (``test_an_unrebuilt_render_reads_never``) and the table's
  ``tier-absent-old-armed-*`` and ``backstop-legacy-armed`` rows;
- the unset-type fallback to the simulator: ``tests/connectors/
  test_factory_mock_default.py::TestFactoryTypeFallback``.
"""

from __future__ import annotations

from osprey.services.python_executor.execution.control import ExecutionMode


class TestExecutionModeEnum:
    def test_mode_values(self):
        assert ExecutionMode.READ_ONLY.value == "read_only"
        assert ExecutionMode.WRITE_ACCESS.value == "write_access"
