"""A measurement kind a deployment package registers: the registry's test fixture.

Shaped from what a turn-by-turn measurement needs of a kind, under its own
names: a monitor group and two corrector groups stepped as a response matrix,
instruments judged ``exists`` and ``optional``, a sub-block of its own (a
trigger sample, operator settings and an ordered chain of steps, each of which
may write a command), a build-stop hook that refuses a command step whose
address the facility does not mark ``command``, a facts writer and a settings
transform. It imports OSPREY only through :mod:`osprey.facility.kinds`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from osprey.facility.kinds import KindDeclaration, KindStop, KindTool, ViewContext, ViewSection

#: The mark the build puts on a channel a program fires as a command.
COMMAND = "command"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Done(_Strict):
    address: str
    value: Any


class Command(_Strict):
    address: str
    value: Any
    timeout_s: float = Field(gt=0)
    done: Done | None = None


class Step(_Strict):
    id: str
    label: str
    description: str
    default: bool
    command: Command | None = None


class Setting(_Strict):
    id: str
    label: str
    description: str
    scope: Literal["device", "global"]
    signal: str | None = None
    address: str | None = None
    timeout_s: float = Field(gt=0)


class RegistryKindBlock(_Strict):
    """The kind's own measurement sub-block."""

    trigger_sample: int = Field(ge=0)
    settings: list[Setting] = []
    steps: list[Step] = []


def _commands(document: Mapping[str, Any]) -> set[str]:
    """The addresses the rendered facility definition marks ``command``."""
    return {
        str(channel.get("id"))
        for channel in document.get("channels") or []
        if channel.get(COMMAND) is True
    }


def build_stops(document: Mapping[str, Any], block: Any) -> Iterator[KindStop]:
    """Refuse a command step whose address the facility does not mark ``command``."""
    commands = _commands(document)
    for step in (block or {}).get("steps") or []:
        command = step.get("command") or {}
        address = command.get("address")
        if address is not None and str(address) not in commands:
            yield KindStop(
                f"step {step.get('id')} writes {address}, which the facility does not mark "
                f"{COMMAND}",
                f"mark {address} {COMMAND} in the facility description, or drop the step's command",
            )


def facts_writer(document: Mapping[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
    """The kind's own facts: the chain of steps, the trigger sample and the settings."""
    del document
    return {
        "chain": [step["id"] for step in block.get("steps") or []],
        "trigger_sample": block.get("trigger_sample"),
        "settings": [setting["id"] for setting in block.get("settings") or []],
    }


#: Turns the engine tracks past the trigger sample.
WINDOW = 64


def settings_from_measurement(settings: Mapping[str, Any], block: Mapping[str, Any]) -> dict:
    """The engine's settings with its tracked turns reaching past the trigger sample."""
    return {**settings, "turns": int(block["trigger_sample"]) + WINDOW}


def _section(context: ViewContext) -> ViewSection:
    arrays = context.arrays
    return ViewSection(
        {
            "bpm_array_name": arrays["bpm"],
            "hcorr_array_name": arrays["hcor"],
            "vcorr_array_name": arrays["vcor"],
        }
    )


_SETTLE = {
    "sleep_between_step": "$settle",
    "n_avg_meas": "$n_avg_meas",
    "sleep_between_meas": "$sleep_between_meas",
}

REGISTRY_KIND = KindDeclaration(
    name="registry_kind",
    tool=KindTool("pyaml.tuning_tools.orbit_response_matrix", "REGISTRY_KIND_RESPONSE"),
    members={
        "bpm": "wired",
        "hcor": "wired",
        "vcor": "wired",
        "tune": "exists",
        "rf": "optional",
    },
    stepped_roles=("hcor", "vcor"),
    tool_keys=("corrector_delta",),
    single_pass=True,
    matrix=True,
    takes_by=False,
    corrector_roles=("hcor", "vcor"),
    view_files={},
    measure_plan={
        "tool": {"attribute": "registry_kind"},
        "fit": "magnet-group",
        "magnets": {"members": ["hcorrectors", "vcorrectors"]},
        "named": True,
        "delta": "corrector_delta",
        "bpm_array": "bpm_array_name",
        "bipolar": {"set_delta": True, "kwargs": {"corrector_names": "$names", **_SETTLE}},
        "unipolar": {
            "per_member": True,
            "factory": "orm",
            "kwargs": {
                "bpm_array": "@bpm_array_name",
                "correctors": "$names",
                "deltas": "$deltas",
                "set_wait_time": "$settle",
                "n_avg_meas": "$n_avg_meas",
            },
        },
        "result": "matrix",
        "missing": "kind",
    },
    description="corrector-stepped response of the turn-series monitors",
    measurement_schema=RegistryKindBlock,
    view_section=_section,
    facts_writer=facts_writer,
    build_stops=build_stops,
    settings_from_measurement=settings_from_measurement,
)
