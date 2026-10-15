"""Simulation engine plug-ins, discovered through the ``osprey.simulation.engines`` group.

Each plug-in registers a module under :data:`ENTRY_POINT_GROUP`. A build
reaches an engine only through its entry point, never by importing the module
by name, and :class:`EnginePlugin` states the contract the module implements.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lume.model import LUMEModel

__all__ = ["ENTRY_POINT_GROUP", "EnginePlugin"]

#: The entry-point group every engine plug-in registers under.
ENTRY_POINT_GROUP = "osprey.simulation.engines"


@runtime_checkable
class EnginePlugin(Protocol):
    """The functions an engine plug-in module states.

    ``osprey build`` calls the first four without building a model; the
    virtual accelerator calls ``build`` once per served model.
    ``isinstance(module, EnginePlugin)`` checks each is present, not its
    signature.

    A plug-in may state these names too, each looked up on the module by name:

    ======================================================  ==============================================  ==========================================
    name                                                    what it gives                                   a module without it
    ======================================================  ==============================================  ==========================================
    ``error_text(exc)``                                     the one line a failed build reports             the exception's own text
    ``fault_variables(wiring)``                             every fault a model built from the wiring       the model declares no faults
                                                            declares
    ``polynomial_kicks(deck, elements)``                    a deck copy carrying named kicks as dipole      the pyAML view stops ``engine-invalid``
                                                            polynomials; stand-in until pyAML reads a       when design correctors need it
                                                            dipole kick held in ``KickAngle``
    ``response_matrix(deck, wiring, settings,               the orbit response of the wired monitors:       ``model_response_matrix`` ends without a
    correctors, step)``                                     ``rows``, ``columns``, ``matrix`` and the       result; the response check stops
                                                            ``unsolved`` correctors with their texts
    ``SOLVE_ERRORS``                                        the exceptions meaning "this request has no     every exception is a failure, never a
                                                            solve"; stand-in until lume-base states         refusal
                                                            ``SolveError``
    ======================================================  ==============================================  ==========================================
    """

    def locate(self, deck: Any, element: str, *, model: str | None = None) -> tuple[float, float]:
        """An element's entrance and length along the deck."""
        ...

    def prepare(self, deck: Any, settings: Any, *, model: str | None = None) -> Any:
        """The model's settings, checked against the deck and normalised once."""
        ...

    def start_values(
        self,
        deck: Any,
        wiring: Any,
        settings: Any,
        *,
        readbacks: Mapping[str, str | None] | None = None,
        value_types: Mapping[str, str] | None = None,
        shapes: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Each wired channel's operating point, derived from the deck."""
        ...

    def describe(self, wiring_record: Any) -> Mapping[str, Any]:
        """What a wiring record is: its role, plane and refresh."""
        ...

    def build(
        self,
        model: str,
        wiring: Any,
        deck: Any,
        settings: Any,
        active: Mapping[str, Any] = ...,
    ) -> LUMEModel:
        """The LUME model that serves one model's wiring over its deck."""
        ...
