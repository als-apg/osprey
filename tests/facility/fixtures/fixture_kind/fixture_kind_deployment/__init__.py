"""A deployment package that registers a measurement kind and an engine, as a test fixture.

The package a deployment ships beside its facility tree (``data/facility/`` in
the deployment repository, not a mounted machine repository): it registers the
measurement kind ``fixture_kind`` under ``osprey.facility.kinds`` and the engine
``fixture_ring`` (:mod:`fixture_kind_deployment.engine`) under
``osprey.simulation.engines``, and it imports OSPREY only through
:mod:`osprey.facility.kinds`.

The kind is shaped from what a turn-by-turn measurement needs, under its own
names:

* members: one turn-series monitor group of its own (``series_monitors``,
  wired), the two corrector groups it steps (``hcor`` and ``vcor``, wired), and
  five instruments
  judged ``exists`` or ``optional``: an injection request, an injection busy
  flag, a bunch-clean enable, the beam current and an RF status;
* a response matrix, measured on a single-pass model too, with no ``by``;
* its own sub-block ``kind_blocks.fixture_kind``, validated by
  :class:`FixtureBlock`: the trigger sample, the ordered operator settings (one
  scoped to a device and named by its signal, one global and named by its
  address) and the ordered chain of steps, each of which may fire a command and
  wait for its done reading;
* a view file, ``fixture_chain.json``, holding the chain's step ids;
* a build stop refusing a command step on a channel the facility does not mark
  ``command``;
* the facts it writes into each model's kind block;
* a settings transform asking the engine for enough turns past the trigger
  sample.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from osprey.facility.kinds import KindDeclaration, KindStop, KindTool, ViewContext, ViewSection

#: The kind's name, as a measurement file lists it.
KIND = "fixture_kind"

#: The engine the package registers beside the kind.
ENGINE = "fixture_ring"

#: The mark the build puts on a channel a program fires as a command.
COMMAND = "command"

#: The view file the kind adds to a model's pyAML view.
CHAIN_FILE = "fixture_chain.json"

#: Turns the engine tracks past the trigger sample.
WINDOW = 64


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Done(_Strict):
    """The reading a command's step waits for."""

    address: str
    value: Any


class Command(_Strict):
    """What a step fires, and how long it waits."""

    address: str
    value: Any
    timeout_s: float = Field(gt=0)
    done: Done | None = None


class Step(_Strict):
    """One step of the chain, in order."""

    id: str
    label: str
    description: str
    default: bool
    command: Command | None = None


class Setting(_Strict):
    """One operator setting: per device by its signal, or global by its address."""

    id: str
    label: str
    description: str
    scope: Literal["device", "global"]
    signal: str | None = None
    address: str | None = None
    timeout_s: float = Field(gt=0)


class FixtureBlock(_Strict):
    """The kind's own measurement sub-block."""

    trigger_sample: int = Field(ge=0)
    settings: list[Setting] = []
    steps: list[Step] = []


def _commands(document: Mapping[str, Any]) -> set[str]:
    """The addresses the facility file marks ``command``."""
    return {
        str(channel.get("id"))
        for channel in document.get("channels") or []
        if channel.get(COMMAND) is True
    }


def build_stops(document: Mapping[str, Any], block: Any) -> Iterator[KindStop]:
    """Refuse a command step on a channel the facility does not mark ``command``."""
    commands = _commands(document)
    for step in (block or {}).get("steps") or []:
        address = (step.get("command") or {}).get("address")
        if address is not None and str(address) not in commands:
            yield KindStop(
                f"step {step.get('id')} fires {address}, which the facility does not mark "
                f"{COMMAND}",
                f"wire {address} where its engine marks it {COMMAND}, or drop the step's command",
            )


def facts_writer(document: Mapping[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
    """The facts a model's block gives the kind: its chain, trigger sample and settings."""
    del document
    return {
        "chain": [step["id"] for step in block.get("steps") or []],
        "commands": [
            step["command"]["address"] for step in block.get("steps") or [] if step.get("command")
        ],
        "trigger_sample": block.get("trigger_sample"),
        "settings": [setting["id"] for setting in block.get("settings") or []],
    }


def settings_from_measurement(
    model_entry: Mapping[str, Any], measurement_record: Mapping[str, Any]
) -> dict[str, Any]:
    """The turns the engine must reach past the trigger sample, as a settings patch.

    A model whose file states no block, or a model of another engine, is
    patched nothing.
    """
    if model_entry.get("engine") != ENGINE:
        return {}
    block = (measurement_record.get("kind_blocks") or {}).get(KIND) or {}
    if "trigger_sample" not in block:
        return {}
    return {ENGINE: {"turns": int(block["trigger_sample"]) + WINDOW}}


def _section(context: ViewContext) -> ViewSection:
    arrays = context.arrays
    return ViewSection({"hcorr_array_name": arrays["hcor"], "vcorr_array_name": arrays["vcor"]})


def _chain(context: ViewContext) -> str | None:
    block = (context.measurement.get("kind_blocks") or {}).get(KIND)
    if not block:
        return None
    return json.dumps([step["id"] for step in block.get("steps") or []]) + "\n"


_SETTLE = {
    "sleep_between_step": "$settle",
    "n_avg_meas": "$n_avg_meas",
    "sleep_between_meas": "$sleep_between_meas",
}

FIXTURE_KIND = KindDeclaration(
    name=KIND,
    tool=KindTool("pyaml.tuning_tools.orbit_response_matrix", "FIXTURE_KIND_RESPONSE"),
    members={
        "series_monitors": "wired",
        "hcor": "wired",
        "vcor": "wired",
        "inject_request": "exists",
        "inject_busy": "exists",
        "clean_enable": "optional",
        "beam_current": "exists",
        "rf_status": "optional",
    },
    member_sorts={
        "series_monitors": "monitors",
        "inject_request": "instrument",
        "inject_busy": "instrument",
        "clean_enable": "instrument",
        "beam_current": "instrument",
        "rf_status": "instrument",
    },
    stepped_roles=("hcor", "vcor"),
    tool_keys=("corrector_delta",),
    single_pass=True,
    matrix=True,
    takes_by=False,
    corrector_roles=("hcor", "vcor"),
    view_files={CHAIN_FILE: _chain},
    measure_plan={
        "tool": {"attribute": KIND},
        "fit": "magnet-group",
        "magnets": {"members": ["hcorrectors", "vcorrectors"]},
        "named": True,
        "delta": "corrector_delta",
        "bipolar": {"set_delta": True, "kwargs": {"corrector_names": "$names", **_SETTLE}},
        "unipolar": {
            "per_member": True,
            "factory": "orm",
            "kwargs": {
                "correctors": "$names",
                "deltas": "$deltas",
                "set_wait_time": "$settle",
                "n_avg_meas": "$n_avg_meas",
            },
        },
        "result": "matrix",
        "missing": "kind",
    },
    description="corrector response of the turn-series monitors",
    measurement_schema=FixtureBlock,
    view_section=_section,
    facts_writer=facts_writer,
    build_stops=build_stops,
    settings_from_measurement=settings_from_measurement,
)
