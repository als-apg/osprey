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
    ``condition(model)``                                    ``(text, names)`` while a built model serves    no model is ever degraded
                                                            degraded: one text of ``CONDITIONS`` and the
                                                            names whose values it voids; else ``None``
    ``error_text(exc)``                                     the one line a failed build reports             the exception's own text
    ``fault_variables(wiring)``                             every fault a model built from the wiring       the model declares no faults
                                                            declares
    ``polynomial_kicks(deck, elements)``                    a deck copy carrying named kicks as dipole      the pyAML view stops ``engine-invalid``
                                                            polynomials                                     when design correctors need it
    ``response_matrix(deck, wiring, settings,               ``(rows, matrix)``, the orbit response of the   ``model_response_matrix`` ends without a
    correctors, step, *, kind='orbit')``                    wired monitors; for ``kind='trajectory'`` row   result; the response check stops
                                                            names, column names, matrix, ``bpm_index``,
                                                            ``hcor_index`` and ``vcor_index``, the
                                                            first-turn response of the wired turn series
    ``scenario_stops(deck, faults)``                        one ``(fault name, detail, remedy)`` per        a scenario is checked against the fault
                                                            thing a scenario's faults on the model ask      names alone
                                                            that the deck cannot give
    ``CONDITIONS``                                        every text ``condition`` can answer, a closed   ``condition`` may answer any text
                                                            list
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
