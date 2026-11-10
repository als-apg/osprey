"""The ring view: a model's BPMs and correctors per plane, as the solver needs them.

Everything comes from the SDK (``closed_orbit_models``, ``model_members``); this
module adds no facility knowledge. Ring order and "downstream" go by lattice index:
elements sharing one ``s`` are ordered by their index, never by ``s``. A BPM's
ring position ``b`` is its place among its plane's BPMs in that order, a corrector's
``c`` its place among its plane's correctors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from osprey.apps.sdk import Member, ModelMembers, closed_orbit_models, model_members

#: The planes a member reads or steers, in the order the view lists them.
PLANES = ("x", "y")


@dataclass(frozen=True)
class PlaneView:
    """One plane of a model: its BPMs and correctors in ring order.

    Attributes:
        plane: ``x`` or ``y``.
        bpms: The plane's BPM members, by lattice index.
        correctors: The plane's corrector members, by lattice index.
        bpm_idx: Each BPM's lattice index, for ``split_rows``.
        cor_idx: Each corrector's lattice index, for ``split_rows``.
        next_cor: For each BPM, the ring position of the first corrector whose
            lattice index is greater, wrapping to 0 at the ring end; empty when
            the plane has no corrector.
        n_index: One past the highest lattice index of the model (both planes),
            the ring size ``split_rows`` wraps at.
    """

    plane: str
    bpms: tuple[Member, ...]
    correctors: tuple[Member, ...]
    bpm_idx: list[int]
    cor_idx: list[int]
    next_cor: list[int]
    n_index: int


@dataclass(frozen=True)
class UnsolvedCorrector:
    """A corrector placed on the ring by its setpoint address.

    Attributes:
        element: The corrector's lattice element.
        plane: ``x`` or ``y``.
        position: Its ring position among its plane's correctors.
        setpoint_addr: The setpoint address the ORM names it by.
        reason: The ORM's text for why its step has no solve; ``None`` when the
            corrector was only looked up.
    """

    element: str
    plane: str
    position: int
    setpoint_addr: str
    reason: str | None


@dataclass(frozen=True)
class RingView:
    """A model's ring: its length and one ``PlaneView`` per plane."""

    model: str
    length_m: float
    planes: Mapping[str, PlaneView]

    def plane(self, plane: str) -> PlaneView:
        """The view of ``plane``.

        Raises:
            ValueError: ``plane`` is not one of ``PLANES``.
        """
        try:
            return self.planes[plane]
        except KeyError:
            raise ValueError(f"unknown plane {plane!r}; the planes: {', '.join(PLANES)}") from None

    def corrector(self, setpoint_addr: str) -> UnsolvedCorrector | None:
        """The corrector whose setpoint is ``setpoint_addr``; ``None`` when none is."""
        for name in PLANES:
            for position, member in enumerate(self.planes[name].correctors):
                if member.setpoint_addr == setpoint_addr:
                    return UnsolvedCorrector(
                        element=member.element,
                        plane=name,
                        position=position,
                        setpoint_addr=setpoint_addr,
                        reason=None,
                    )
        return None

    def unsolved(self, unsolved: Mapping[str, str]) -> list[UnsolvedCorrector]:
        """The ORM's ``unsolved`` setpoint addresses placed on the ring.

        Ordered by plane, then ring position. An address that is not one of the
        model's correctors has no ring position and is left out.
        """
        placed = []
        for address, reason in unsolved.items():
            found = self.corrector(address)
            if found is not None:
                placed.append(
                    UnsolvedCorrector(
                        element=found.element,
                        plane=found.plane,
                        position=found.position,
                        setpoint_addr=address,
                        reason=reason,
                    )
                )
        return sorted(placed, key=lambda c: (PLANES.index(c.plane), c.position))

    def unsolved_positions(self, unsolved: Mapping[str, str], plane: str) -> set[int]:
        """The ring positions of ``plane``'s unsolved correctors, for ``select_correctors``."""
        self.plane(plane)
        return {c.position for c in self.unsolved(unsolved) if c.plane == plane}

    def unsolved_elements(self, unsolved: Mapping[str, str]) -> list[str]:
        """The unsolved correctors' elements, in ring order, for the state's ``orm.unsolved``."""
        return [c.element for c in self.unsolved(unsolved)]


def offered_models(render_root: Path | None = None) -> list[str]:
    """The models the panel offers: the render's closed-orbit models, facility order."""
    return closed_orbit_models(render_root)


def ring_view(model: str, render_root: Path | None = None) -> RingView:
    """The ring view of ``model``, read through the SDK.

    Raises:
        ValueError: The SDK refuses the model (see ``model_members``), or
            :func:`build_ring` does.
    """
    return build_ring(model_members(model, render_root=render_root))


def position_addr(bpm: Member) -> str:
    """A BPM's position readback; ``ValueError`` when it has none."""
    if bpm.position_addr is None:
        raise ValueError(f"the BPM {bpm.element} has no position readback")
    return bpm.position_addr


def setpoint_addr(corrector: Member) -> str:
    """A corrector's setpoint; ``ValueError`` when it has none."""
    if corrector.setpoint_addr is None:
        raise ValueError(f"the corrector {corrector.element} has no setpoint")
    return corrector.setpoint_addr


def build_ring(members: ModelMembers) -> RingView:
    """The ring view of already-read ``members``.

    Raises:
        ValueError: A BPM without a position readback or a corrector without a
            setpoint; the view holds none, so :func:`position_addr` and
            :func:`setpoint_addr` never raise on its members.
    """
    for bpm in members.bpms:
        position_addr(bpm)
    for corrector in members.correctors:
        setpoint_addr(corrector)
    everything = [*members.bpms, *members.correctors]
    n_index = max((m.index for m in everything), default=-1) + 1
    planes = {}
    for name in PLANES:
        bpms = sorted((m for m in members.bpms if m.plane == name), key=_index)
        correctors = sorted((m for m in members.correctors if m.plane == name), key=_index)
        cor_idx = [m.index for m in correctors]
        planes[name] = PlaneView(
            plane=name,
            bpms=tuple(bpms),
            correctors=tuple(correctors),
            bpm_idx=[m.index for m in bpms],
            cor_idx=cor_idx,
            next_cor=_next_cor([m.index for m in bpms], cor_idx),
            n_index=n_index,
        )
    return RingView(model=members.model, length_m=members.length_m, planes=planes)


def _index(member: Member) -> int:
    return member.index


def _next_cor(bpm_idx: list[int], cor_idx: list[int]) -> list[int]:
    """Per BPM, the first corrector with a greater lattice index, wrapping to 0."""
    if not cor_idx:
        return []
    return [next((c for c, idx in enumerate(cor_idx) if idx > b), 0) for b in bpm_idx]
