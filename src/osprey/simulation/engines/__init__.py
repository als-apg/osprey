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

__all__ = ["ENTRY_POINT_GROUP", "OPTIONAL_NAMES", "TIME_CONTRACT", "EnginePlugin"]

#: The entry-point group every engine plug-in registers under.
ENTRY_POINT_GROUP = "osprey.simulation.engines"

#: The optional functions by which time and client writes reach a model.
TIME_CONTRACT = ("advance", "next_deadline", "emitted", "client_write")

#: Every name a plug-in may state beyond :class:`EnginePlugin`'s, each looked
#: up on the module by name; ``deckless_start_values`` and the ``*_ERRORS``
#: tuples are values, the rest are functions.
OPTIONAL_NAMES = (
    *TIME_CONTRACT,
    "deck_length",
    "deckless_start_values",
    "describe_input",
    "element_index",
    "error_text",
    "fault_variables",
    "polynomial_kicks",
    "readout",
    "response_matrix",
    "scenario_stops",
    "take_inputs",
    "REFUSAL_ERRORS",
    "SOLVE_ERRORS",
)


@runtime_checkable
class EnginePlugin(Protocol):
    """The functions an engine plug-in module states.

    ``osprey build`` calls the first four without building a model; the
    virtual accelerator calls ``build`` once per served model.
    ``isinstance(module, EnginePlugin)`` checks each is present, not its
    signature.

    A plug-in may state these names too, each looked up on the module by name:

    ======================================  ============================================  ==========================================
    name                                    what it gives                                 a module without it
    ======================================  ============================================  ==========================================
    ``advance(model, t_s)``                 applies every step due by epoch second        the model is never advanced
                                            ``t_s``; the first call after a build or a
                                            reset anchors the model's time
    ``client_write(model, values, t_s)``    tells the engine a client wrote these         a client write starts nothing
                                            setpoints, already forwarded; the one call
                                            that starts what a client write triggers
    ``deck_length(deck, *, model=None)``    the length of the deck's lattice in metres    the SDK's ``model_members`` raises
                                            (a ring's circumference)
    ``deckless_start_values``               ``True``: ``start_values`` may be called      ``start_values`` is not called without a
                                            with no deck                                  deck
    ``describe_input(input_record)``        checks one model input's ``{address,          every input's ``engine`` block is
                                            engine}`` record at facility build;           accepted as given
                                            raising ``ValueError`` refuses it and the
                                            build stops naming the input
    ``element_index(deck, element,``        the element's lattice index: ring order,      the SDK's ``model_members`` raises
    ``*, model=None)``                      which tells co-located elements apart
    ``emitted(model)``                      the addresses of ``event`` channels fired     the model emits nothing
                                            since the last call, cleared as returned
    ``error_text(exc)``                     the one line a failed build reports           the exception's own text
    ``fault_variables(wiring, settings)``   every fault a model built from the wiring     the model declares no faults
                                            and its settings declares, by address or by
                                            element, without a deck load. Raising
                                            ``ValueError`` refuses the model: the build
                                            stops ``engine-invalid`` with its text
    ``next_deadline(model)``                the epoch second of the model's next due      the model has no deadline
                                            step, or ``None``
    ``polynomial_kicks(deck, elements)``    a deck copy carrying named kicks as dipole    the pyAML view stops ``engine-invalid``
                                            polynomials; stand-in until pyAML reads a     when design correctors need it
                                            dipole kick held in ``KickAngle``
    ``readout(model, values, t_ms)``        what each monitor reads of ``values``, its    a monitor reads its beam truth as given
                                            beam truth at epoch millisecond ``t_ms``
    ``response_matrix(...)``                called with ``(deck, wiring, settings,        ``model_response_matrix`` ends without a
                                            correctors, step)`` and the keyword           result; the response check stops
                                            ``kind``, ``"orbit"`` or ``"trajectory"``:
                                            the response of the wired monitors,
                                            ``rows``, ``columns``, ``matrix`` and the
                                            ``unsolved`` correctors with their texts.
                                            One that takes no ``kind`` is called without
                                            it for ``"orbit"``, and any other kind is
                                            refused naming the engine. Raising one of
                                            ``REFUSAL_ERRORS`` refuses the request with
                                            its text; a ``FacilityBuildError`` refuses
                                            it as a facility-file defect; a return with
                                            no column is refused with the ``unsolved``
                                            map
    ``scenario_stops(wiring, faults)``      ``(fault, detail, remedy)`` per refusal of a  no scenario stops
                                            scenario's faults on the model, channel-
                                            keyed ones included, without a deck load;
                                            each stops the build ``engine-invalid``
                                            with the plug-in's remedy
    ``take_inputs(model, inputs, t_s,``     stores ``inputs``: one ``(input_record,       the build refuses a model that declares
    ``*, baseline)``                        value)`` pair per declared input, the         inputs
                                            record its ``{address, engine}`` and the
                                            value the producer's current plain value;
                                            ``baseline=True`` marks a starting point
    ``REFUSAL_ERRORS``                      the exceptions meaning "the request is not    no exception is a request refusal
                                            one this engine answers"; stand-in until
                                            lume-base states it
    ``SOLVE_ERRORS``                        the exceptions meaning "this request has no   every exception is a failure, never a
                                            solve"; stand-in until lume-base states       refusal
                                            ``SolveError``
    ======================================  ============================================  ==========================================

    The four names of :data:`TIME_CONTRACT` make an engine time-driven: a lazy
    state machine with no thread of its own. The same calls and clock readings
    always yield the same values. A model's ``set`` only stores values and
    never starts time-driven behaviour, since restore, rollback, journal replay
    and scenario writes all reach it; what a client write triggers lives behind
    ``client_write`` alone. ``build`` receives no clock, so the caller calls
    ``advance`` right after a build, a rebuild or a reset. The caller advances
    every built child on every pass, whether or not it has a step due. A model
    that declares inputs gets ``take_inputs`` immediately before its
    ``advance``, with the values its producers hold on that same pass.
    ``take_inputs`` only stores, like ``set``; what a new value starts happens
    in the ``advance`` that follows. ``baseline=True`` means the delivery is a
    starting point: compare nothing against what was held before. The
    ``engine`` block travelling with every delivery is how a model tells its
    inputs apart; ``build`` is unchanged. :data:`OPTIONAL_NAMES` lists every
    name of this table.
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
        """What a wiring record is: its engine_role, plane and refresh, and any marks."""
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
