"""The runner that serves the virtual accelerator's whole namespace.

One process, one model, one Channel Access server. :class:`CohostRunner`
takes the serving package's runner -- which knows how to serve a
:class:`~lume.model.LUMEModel`'s own variables -- and co-hosts the
facility's entire channel manifest alongside them, so a client sees one
namespace of ~2,900 channels rather than the handful a model happens to
describe. Everything specific to that arrangement is here; what a write to
a setpoint *means* is in
:mod:`~osprey.services.virtual_accelerator.serving.write_path`, which
imports no server library and is therefore testable without one.

**This module imports the CA server extension.** It is reached lazily
(``serving.runner``, never an eager re-export) precisely so that importing
the serving package stays cheap; see the package docstring.

Four decisions shape the subclass:

*The Channel Access namespace is contributed whole.* Every served CA PV
comes from the manifest-derived database, including the addresses that also
happen to be model variables. The base class would otherwise build a second
CA PV per model variable, from a spec derived from the variable rather than
from the manifest's record-type table -- a duplicate name that the database
merge rejects outright, and, were it not rejected, a BPM served without the
display precision its metre-scale readings need and without the boot value
the physics bridge pushed into the manifest's own spec. So the CA half of
the base class's per-variable PV creation is suppressed and the model's
variables are served on PVA alone.

*No served value derives from a model read.* The run loop's output pass --
read every variable back after a cycle and publish it -- is suppressed
entirely. Readings reach their PVs from the physics bridge instead, which
applies each BPM's seeded readout error to the value it pushes. Publishing
``model.get`` alongside that would overwrite the reading with the truth it
is supposed to differ from, and would do it a fraction of a second later, so
the divergence would come and go rather than hold.

*Every model access happens on the run loop's thread.* The model is
explicitly not thread-safe: two threads writing through it interleave their
lattice mutations and its rollback restores whichever snapshot it happens to
hold. The server thread therefore never touches it -- a co-hosted setpoint
write is enqueued and completed later from the loop -- and the physics hook
is reached through the model wrapper, so it too runs there.

*The runner claims no name of its own.* Control PVs are off: nothing is
served that the facility's channel manifest does not describe.

One consequence is worth stating plainly. The model's variables are served
on **both** transports -- co-hosted on CA, natively on PVA -- and the two
are not synchronised, because synchronising them is what the suppressed
output pass used to do. A PVA client sees its own puts echoed and nothing
else; Channel Access is the authoritative view of the machine, and PVA is
served in-container for model metadata and for exercising the runner's
native write path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import pcaspy
from lume_pva_apg.runner import Runner

from osprey.services.virtual_accelerator.serving.write_path import (
    RUNNER_CONFIG_POLICY,
    CohostWritePath,
    SetpointRoutedModel,
    physics_setpoint_addresses,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel, Variable

    from osprey.services.virtual_accelerator.serving.pvdb import ServingRecords

#: Protocols served. PVA carries the model's own variables and the
#: ``model_info`` structure; CA carries the co-hosted namespace. CA must be
#: enabled for the co-hosted database to be contributed at all.
DEFAULT_PROTOCOLS = ("ca", "pva")

#: Alarm raised on a setpoint whose write the model refused. Put-completion
#: can only ever report success, so this is the only signal a Channel Access
#: client gets that its write did not land.
REFUSAL_ALARM = (pcaspy.Alarm.WRITE_ALARM, pcaspy.Severity.INVALID_ALARM)


class CohostDriver(Runner.CaDriver):
    """Serves the co-hosted namespace, which the stock driver knows nothing of.

    The stock driver resolves a write's ``reason`` through the model's
    variables and refuses whatever it cannot find -- which here is every
    served PV, since the co-hosted database is the whole Channel Access
    namespace. So this intercepts *before* delegating and never delegates:
    the stock path would enqueue a bare ``model.set``, bypassing the physics
    hook that a setpoint write is supposed to run through.
    """

    def write(self, reason: str, value: Any) -> bool:
        """Delegate the whole decision to the runner's write path."""
        accepted: bool = self.runner.write_path.write(self, reason, value)
        return accepted


class CohostRunner(Runner):
    """Serves a model's variables and the facility's whole channel manifest.

    See the module docstring for the four decisions this makes and their one
    visible consequence.
    """

    ca_driver_cls = CohostDriver

    # Declared for the type checker alone: the base class sets it in its own
    # constructor, from a module that carries no annotations. No value, so
    # nothing exists at runtime that could shadow it.
    supports_ca: bool

    def __init__(
        self,
        model: LUMEModel,
        records: ServingRecords,
        *,
        on_setpoint: Callable[[str, float], None] | None = None,
        drive_limits: Mapping[str, tuple[float, float]] | None = None,
        stuck_setpoints: frozenset[str] = frozenset(),
        prefix: str = "",
        protocol: tuple[str, ...] = DEFAULT_PROTOCOLS,
    ) -> None:
        """Assemble the server around an already-built serving database.

        The database must be built, and every value that should be on the
        wire at boot pushed into it, *before* this runs: the CA server copies
        each spec when it creates the PV, so a value written into a spec
        afterwards is never served. That is why the physics bridge is bound
        to its records by the caller rather than here -- its first push of
        BPM readings is the boot state, and it has to land in the specs this
        constructor hands to the server.

        Args:
            model: the physics model. Its variables are served on PVA; its
                writable ones are what ``on_setpoint`` ultimately writes to.
                A model with no variables (no lattice in this process) is
                valid and serves the co-hosted namespace alone.
            records: the built serving database, from
                ``build_serving_pvdb``. Its records are pointed at the live
                driver here, once that driver exists.
            on_setpoint: the physics hook, called as
                ``on_setpoint(address, value)`` on the run loop's thread for
                each pyat-coupled setpoint write. ``None`` serves those
                setpoints as plain latches: they record what was written and
                propagate nothing, there being no physics to propagate into.
            drive_limits: ``{address: (low, high)}``; each written value is
                clamped into its band before anything else happens to it.
            stuck_setpoints: apply-fault addresses whose readbacks must never
                move again.
            prefix: prepended to every served name. Empty for a facility
                whose manifest addresses are already absolute.
            protocol: the protocols to serve.

        Raises:
            ValueError: Channel Access is not among the protocols. The
                co-hosted database is contributed on a hook that fires only
                for CA, so without it the facility's namespace is silently
                not served at all.
            RuntimeError: the serving database is empty, so no Channel Access
                server was created and there is nothing to serve.
        """
        if "ca" not in protocol:
            raise ValueError(
                "the co-hosted channel namespace is served over Channel Access: "
                f"'ca' must be among the protocols, got {list(protocol)}"
            )

        self._records = records
        physics_setpoints = physics_setpoint_addresses(records)

        # Bound before the base constructor runs, because the base
        # constructor is what builds the driver, and the driver's first act
        # may be to consult this. `self._enqueue` is bound here and called
        # only later, once the base constructor has created the queue behind
        # it.
        self.write_path = CohostWritePath(
            records,
            enqueue=self._enqueue if on_setpoint is not None else None,
            physics_setpoints=physics_setpoints,
            stuck_setpoints=stuck_setpoints,
            drive_limits=drive_limits,
            refusal_alarm=REFUSAL_ALARM,
        )

        if on_setpoint is not None:
            model = SetpointRoutedModel(model, on_setpoint=on_setpoint, routed=physics_setpoints)

        config = Runner.generate_config(model, prefix=prefix)
        config["protocol"] = list(protocol)
        config.update(RUNNER_CONFIG_POLICY)

        super().__init__(model=model, config=config)

        if self.ca_driver is None:
            raise RuntimeError(
                "no Channel Access server was created: the serving database is empty, "
                "so there is nothing for this runner to serve"
            )
        # Last, and only now: until the driver exists a record writes into
        # its spec (which is how the boot values above got there), and after
        # the server has created the PVs a spec write reaches nobody.
        records.attach_driver(self.ca_driver)

    def _extend_pvdb(self) -> dict[str, dict[str, Any]]:
        """Contribute the whole co-hosted database.

        Called by the base constructor before the server is created, which is
        the only point at which the database can still be added to.
        """
        return dict(self._records.pvdb)

    def _add_pv(self, pv: str, var: Variable, ro: bool, prefix: str, handler: Any) -> None:
        """Serve one model variable on PVA only.

        The base implementation builds a PVA provider and a CA database entry
        from the same call, choosing each half by protocol. The CA half is
        suppressed here because the manifest already describes these
        addresses and contributes them itself -- see the module docstring.
        The suppression is done by scoping the flag the base implementation
        consults rather than by reimplementing the PVA half, so the PVA
        provider stays the base class's own code instead of a copy of it that
        would quietly drift.
        """
        supports_ca: bool = self.supports_ca
        self.supports_ca = False
        try:
            super()._add_pv(pv, var, ro=ro, prefix=prefix, handler=handler)
        finally:
            self.supports_ca = supports_ca

    def _post_outputs(self, out_values: dict[str, Any], ts: float) -> None:
        """Publish nothing.

        The run loop reads every variable back at the end of a cycle and
        offers them here to be published. Nothing is: a served reading comes
        from the physics bridge, which applies each BPM's seeded readout
        error on the way out, and publishing the model's own view alongside
        it would overwrite that reading with the truth it exists to differ
        from. Setpoints are published by the write path from the value the
        client wrote and the model accepted, which is likewise not a value
        read back out of the model.
        """

    def _reset_to_cached_state(self) -> None:
        """Roll nothing back.

        The run loop snapshots the model's settable variables before each
        cycle and writes them all back if the cycle fails. Here that would be
        actively harmful: the model already rolls a failed write back
        atomically and completely, and re-writing hundreds of retained
        currents through the physics hook would re-run each one's calibration
        and re-solve the lattice -- turning a refused write, which is
        supposed to be a complete no-op, into a machine-wide one.
        """


__all__ = [
    "DEFAULT_PROTOCOLS",
    "REFUSAL_ALARM",
    "CohostDriver",
    "CohostRunner",
]
