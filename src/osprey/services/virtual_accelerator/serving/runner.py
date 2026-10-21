"""The runner that serves a composite on Channel Access and PVAccess.

:class:`ModelRunner` serves a composite: every channel the composite
declares is one of its variables, so the base class serves them all on both
transports from one configuration, and the model RPC is answered over the
composite's simulator view
(:meth:`~osprey.services.virtual_accelerator.serving.model_surface.ModelSurface.for_view`).

**This module imports the CA server extension**, through the serving
package (``lume_pva_apg``) it builds on. It is reached lazily
(``serving.runner``, never an eager re-export) precisely so that importing
the serving package stays cheap; see the package docstring.

Every model access happens on the run loop's thread. The composite is not
thread-safe, so a model RPC call taken on a p4p worker thread is answered by
a job the run loop runs, never on the thread that received it; that rule is
kept by the RPC handler,
:mod:`~osprey.services.virtual_accelerator.serving.model_rpc_handler`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lume_pva_apg.runner import Runner
from p4p.server.thread import SharedPV

from osprey.services.virtual_accelerator.serving.health import write_document
from osprey.services.virtual_accelerator.serving.model_rpc import REPLY_TYPE, RPC_PV
from osprey.services.virtual_accelerator.serving.model_rpc_handler import (
    RpcFrontDoor,
    instance_name,
    pva_endpoint,
)
from osprey.services.virtual_accelerator.serving.model_surface import ModelSurface
from osprey.services.virtual_accelerator.serving.runner_config import (
    HEALTH_KEYS,
    apply_safety,
    periodic_addresses,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel

    from osprey_connectors.simulation.view import SimulatorView


class ModelRunner(Runner):
    """Serves a composite's channels on both transports, and its model RPC on PVA.

    The configuration is ``Runner.generate_config`` over the composite with
    the view's write safety applied
    (:func:`~osprey.services.virtual_accelerator.serving.runner_config.apply_safety`).
    A write pass reads back every served channel except those the view says
    refresh ``periodic``, which the next periodic pass publishes; a pass with
    no input values reads them all.
    A failed pass rolls the composite back to the state cached before it only
    when ``model.set`` had already succeeded: a refused ``set`` leaves the
    composite as it was. The model RPC's write verbs reply only after a
    publishing pass has run. Every publishing pass's outcome, success or
    failure, is recorded in the surface's health record, which ``status``
    reports and which is rewritten to the health file after every pass when
    one is named.
    """

    def __init__(
        self,
        composite: LUMEModel,
        view: SimulatorView,
        *,
        model_write_token: str | None,
        tick_interval_s: float | None = None,
        instance: str | None = None,
        health_file: Path | None = None,
        failed_pass_tolerance: int | None = None,
    ) -> None:
        """Serve ``composite``, built from the simulator view ``view``.

        Args:
            composite: the composite over the view.
            view: the simulator view.
            model_write_token: the secret a model RPC write must present, or
                ``None`` to refuse every model write. Never logged.
            tick_interval_s: the period of the runner's own passes, or
                ``None`` for none.
            instance: the instance name the model RPC's ``status`` reports,
                or ``None`` for the host this server answers as.
            health_file: where the health record is rewritten, atomically,
                after every publishing pass, or ``None`` to keep it in
                process only.
            failed_pass_tolerance: how many consecutive failed publishing
                passes the health record still counts as ``degraded``, or
                ``None`` for the default of
                :data:`~osprey.services.virtual_accelerator.serving.runner_config.HEALTH_KEYS`.
        """
        self._view = view
        self._instance = instance
        self._health_file = health_file
        self._model_write_token = model_write_token
        self._periodic = periodic_addresses(view)
        self._write_pass = False
        self._set_landed = False
        self._pass_started = False
        self._passes_run = 0
        self._first_pass_error: str | None = None
        config = apply_safety(Runner.generate_config(composite, prefix=""), view)
        if tick_interval_s is not None:
            config["tick_interval_s"] = tick_interval_s
        config["failed_pass_tolerance"] = (
            failed_pass_tolerance
            if failed_pass_tolerance is not None
            else HEALTH_KEYS["failed_pass_tolerance"]
        )
        super().__init__(model=composite, config=config)

    def first_pass(self) -> str | None:
        """Run the start-up publishing pass; the error it failed on, or ``None``.

        Call it before :meth:`run`, on the thread that will call :meth:`run`,
        so every model access stays on one thread. It runs queued items until
        one has run a publishing pass. The start-up item is always queued, and
        a jobs-only item ahead of it runs no pass, so it returns once that item
        has run.
        """
        while self._passes_run < 1:
            self._run_cycle(self.queue.get())
        return self._first_pass_error

    def _run_cycle(self, item: dict[str, Any]) -> None:
        """Note whether this pass carries input values, then run it and record its outcome."""
        self._write_pass = bool(item["values"])
        self._set_landed = False
        self._pass_started = False
        item["done"].append(self._record_pass)
        super()._run_cycle(item)

    def _set_cached_state(self, state: dict[str, Any]) -> None:
        """Note that this cycle runs a publishing pass, then cache ``state``."""
        self._pass_started = True
        super()._set_cached_state(state)

    def _record_pass(self, error: str | None) -> None:
        """Record a publishing pass's outcome; a cycle that ran no pass records nothing."""
        if not self._pass_started:
            return
        if self._passes_run == 0:
            self._first_pass_error = error
        self._passes_run += 1
        document = self._surface.record_pass(error)
        if self._health_file is not None:
            write_document(self._health_file, document)

    def _cycle_output_names(self) -> list[str]:
        """The roster read after ``model.set``; a write pass leaves the periodic addresses out."""
        self._set_landed = True
        roster: list[str] = super()._cycle_output_names()
        if not self._write_pass:
            return roster
        return [name for name in roster if name not in self._periodic]

    def _reset_to_cached_state(self) -> None:
        """Restore the cached state only when the pass failed after ``model.set`` succeeded."""
        if self._set_landed:
            super()._reset_to_cached_state()

    def _create_model_info(self) -> None:
        """Serve the model RPC channel beside the base class's model info."""
        super()._create_model_info()
        self._surface = ModelSurface.for_view(
            self.model,
            self._view,
            instance=self._instance if self._instance is not None else instance_name(),
            endpoint=pva_endpoint(),
            model_write_token=self._model_write_token,
            failed_pass_tolerance=self.config["failed_pass_tolerance"],
        )
        channel = SharedPV(initial=REPLY_TYPE.wrap(""))
        channel.rpc(
            RpcFrontDoor(
                self._surface,
                enqueue=self._enqueue,
                driver=lambda: self.ca_driver,
                queue_depth=self.queue.qsize,
            )
        )
        self.providers[f"{self.config['prefix']}{RPC_PV}"] = channel


__all__ = ["ModelRunner"]
