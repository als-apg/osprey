"""The model a boot with no physics behind it serves.

``VA_LATTICE=none`` describes a facility whose channels a manifest lists but
whose physics this process does not have: no lattice, no PyAT, nothing to
solve. The serving layer still needs *a*
:class:`~lume.model.LUMEModel`, because the runner is built around one -- it
derives its PVA namespace, its ``model_info`` structure and the shape of its
run loop from the model's variables.

:class:`NullModel` is that model at zero size. It describes no variables, so
the runner creates no PVA variable PV and no value is ever read out of it;
what a client sees is the co-hosted Channel Access namespace alone, which
the runner contributes whole from the manifest and which is therefore
*identical* to the one a lattice-backed boot serves -- same addresses, same
count, same boot values. The difference is confined to what happens after a
write.

The run loop still turns. Its startup cycle, and every cycle a queued write
would drive, applies an empty batch, and that is deliberately not an error:
an empty cycle is exactly what a process with no physics has to do with one,
and raising instead would take the loop down on its first tick.

Setpoint writes are unaffected by the absence. A pyat-coupled setpoint with
no physics hook behind it latches the value written to it (``MODE_LATCH`` in
:mod:`~osprey.services.virtual_accelerator.serving.write_path`) -- it
records what was commanded and propagates nothing -- which is what those
channels did under the previous server when no hook was installed.

Nothing here imports a server library or an accelerator-physics one, so a
lattice-free boot pays for this module only what importing ``lume`` costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lume.model import LUMEModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import Variable


class NullModel(LUMEModel):
    """A model that describes nothing and holds nothing.

    Every method is total rather than raising, because the run loop calls
    all of them unconditionally: it snapshots the settable variables before
    a cycle, sets a batch, reads every variable back, and may be asked to
    reset. With no variables, each of those is an operation on an empty
    collection, and answering them emptily is what keeps the loop running.
    """

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """No variables at all.

        A fresh dict per access: the base class hands this straight to
        callers, and a shared one would let any of them mutate the model's
        declared namespace.
        """
        return {}

    def _get(self, names: list[str]) -> dict[str, Any]:
        """Return nothing.

        Only ever reached with an empty ``names``: the public ``get`` checks
        every requested name against :attr:`supported_variables` first and
        raises on anything it does not find, and nothing is ever found here.
        """
        return {}

    def _set(self, values: dict[str, Any]) -> None:
        """Accept nothing, and do nothing with it.

        Likewise only reached with an empty batch -- the public ``set``
        rejects unknown names before delegating -- which is the run loop's
        every cycle in this configuration.
        """

    def reset(self) -> None:
        """Reset nothing: there is no state to restore."""


__all__ = ["NullModel"]
