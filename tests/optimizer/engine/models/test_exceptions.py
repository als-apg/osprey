"""Exception taxonomy contract for the interface layer (issue #81, B1 boundary).

``WriteRefusedError`` expresses a *blocked* write: refused by control-system
policy before ever being attempted. The run loop aborts the phase loudly with
the operator-facing reason and never retries (policy does not become true by
retrying). Distinct from a *failed* write (attempted but unverified), which
keeps the existing verification-failure semantics.
"""

import pickle

import pytest

from osprey_optimizer.models.exceptions import InterfaceError, WriteRefusedError


class TestWriteRefusedError:
    def test_is_an_interface_error(self) -> None:
        """Additive subclass: every existing ``except InterfaceError`` keeps working."""
        assert issubclass(WriteRefusedError, InterfaceError)

    def test_carries_structured_refusal_reason(self) -> None:
        exc = WriteRefusedError(
            "Write to GTL:Q1 refused: writes disabled", reason="WRITES_DISABLED"
        )
        assert exc.reason == "WRITES_DISABLED"

    def test_survives_pickling_with_its_reason(self) -> None:
        """It crosses process boundaries (a CA target process to the API) intact."""
        exc = pickle.loads(
            pickle.dumps(WriteRefusedError("Write refused", reason="WRITES_DISABLED"))
        )
        assert type(exc) is WriteRefusedError
        assert (str(exc), exc.reason) == ("Write refused", "WRITES_DISABLED")

    def test_reason_is_required(self) -> None:
        """A refusal without a reason defeats the taxonomy — reject it at construction."""
        with pytest.raises(TypeError):
            WriteRefusedError("Write refused")  # type: ignore[call-arg]

    def test_message_reaches_str(self) -> None:
        """str(exc) is what the run loop surfaces to the operator (PhaseFailedData)."""
        exc = WriteRefusedError("Write to GTL:Q1 refused: outside limits", reason="LIMITS")
        assert str(exc) == "Write to GTL:Q1 refused: outside limits"
