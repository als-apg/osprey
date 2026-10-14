"""MCP tool: pyaml_measure — run one pyAML measurement on the control target.

The tool is registered on the ``python`` server when the render records a
pyAML view, through :func:`register`; importing this module registers nothing.

One call:

1. refuses when the session's posture refuses writes, before anything is read;
2. resolves the arguments against the render
   (:func:`~._pyaml_measure_args.resolve_measure_args`) and each corrector
   address to its pyAML magnet name through the view's one name mapping;
3. renders the measurement script (:func:`~._pyaml_measure_script.build_measure_script`)
   and runs it through :func:`~._execution_gates.run_gated_execution` in
   ``readwrite`` mode, so it meets every gate ``execute`` meets, carrying the
   approved target and view digest the approval hook set;
4. notes who the run wrote as and which channels it attempted on the call's
   audit record, then parses what the script printed, names every row, column
   and step by its address and device id, and copies a completed run's saved
   measurement, read from the run's own execution folder under its fixed
   name, into the artifact store.

The response is a summary: names, shape, steps used and the view's digest,
never the matrix itself, which ``data_file`` and ``artifact_id`` point at.
Every result lists the setpoints the call moved with their values before the
call (``moved``); a stopped run also returns its ``remainder``. A run that
ends without a result line, killed at its timeout above all, fails naming the
channels it attempted (``written_channels``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, NoReturn

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async
from osprey.mcp_server.python_executor.tools._execution_gates import (
    enforce_posture_clamp,
    run_gated_execution,
)
from osprey.mcp_server.python_executor.tools._pyaml_measure_args import (
    TIMEOUT_KEY,
    MeasureArgs,
    resolve_measure_args,
)
from osprey.mcp_server.python_executor.tools._pyaml_measure_result import (
    MeasureResultError,
    label_result,
    measure_labels,
    parse_measure_result,
)
from osprey.mcp_server.python_executor.tools._pyaml_measure_script import (
    MeasureScript,
    build_measure_script,
)

logger = logging.getLogger("osprey.mcp_server.tools.pyaml_measure")

__all__ = ["TOOL_NAME", "pyaml_measure", "register"]

#: The tool's name as the agent knows it (``mcp__python__pyaml_measure``).
TOOL_NAME = "pyaml_measure"

#: A measurement steps setpoints, so it always runs in write mode.
_EXECUTION_MODE = "readwrite"

#: The rendered config's file name in the render root.
_CONFIG_FILE = "config.yml"

#: The measurement-file keys the script waits and averages by, with the value
#: an unstated key takes.
_SETTLE_KEYS: dict[str, float | int] = {
    "sleep_between_step": 0.0,
    "n_avg_meas": 1,
    "sleep_between_meas": 0.0,
}


def register(server: Any) -> None:
    """Add ``pyaml_measure`` to ``server``.

    Args:
        server: The FastMCP server to register on; nothing else is touched.
    """
    server.tool(pyaml_measure, name="pyaml_measure")


def _refuse(message: str, suggestion: str) -> NoReturn:
    make_error("validation_error", message, [suggestion])


def _timeout(config: dict[str, Any]) -> object:
    """The configured sandbox timeout, defaulted the way the executor defaults it.

    An explicit null stays null, so the resolver refuses it by name; the
    executor refuses the same value too.
    """
    from osprey_connectors.config import DEFAULT_EXECUTION_TIMEOUT_SECONDS

    section, key = TIMEOUT_KEY.split(".")
    block = config.get(section) or {}
    return block.get(key, DEFAULT_EXECUTION_TIMEOUT_SECONDS)


def _facility(render_root: Path) -> dict[str, Any]:
    from osprey.facility import FACILITY_FILE

    document: dict[str, Any] = json.loads((render_root / FACILITY_FILE).read_text(encoding="utf-8"))
    return document


def _script(args: MeasureArgs, approved_view_sha256: str | None) -> MeasureScript:
    """The script inputs: the request plus the names and waits the render states."""
    from osprey.facility.views.pyaml import view_names
    from pyaml_cs_osprey.names import UnmappedName

    document = _facility(args.render_root)
    record: dict[str, Any] = next(
        (entry for entry in document.get("models") or [] if entry.get("name") == args.model), {}
    )
    measurement = record.get("measurement") or {}
    names = None
    if args.correctors is not None:
        mapping = view_names(document, args.model)
        try:
            names = [mapping.magnet_name(address) for address in args.correctors]
        except UnmappedName as exc:
            _refuse(str(exc), f"Use the corrector setpoints of {args.model}'s pyAML view.")
    waits = {key: measurement.get(key, default) for key, default in _SETTLE_KEYS.items()}
    return MeasureScript(
        kind=args.kind,
        model=args.model,
        modulation=args.modulation,
        by=args.by,
        corrector_names=names,
        render_root=str(args.render_root),
        view_path=args.view_path,
        approved_view_sha256=approved_view_sha256,
        settle=float(waits["sleep_between_step"]),
        n_avg_meas=int(waits["n_avg_meas"]),
        sleep_between_meas=float(waits["sleep_between_meas"]),
    )


def _description(args: MeasureArgs) -> str:
    text = f"pyaml_measure {args.kind} on {args.model}, {args.modulation}"
    if args.by is not None:
        text += f", by {args.by}"
    if args.correctors:
        text += f", {len(args.correctors)} correctors"
    return text


def _raise_unparsed(
    exec_result: Any, error: MeasureResultError, kind: str, folder: Path | None
) -> NoReturn:
    """Raise the envelope for a run that left no usable result line.

    The script saves its measurement only after the run completed, so a saved
    file in the run's folder means the setpoints moved and the data is kept:
    the envelope names that file and never reads as a refusal. A failed run
    could not list its setpoints, so its envelope names the channels it
    attempted (``written_channels``), left where the run stopped.
    """
    from osprey.mcp_server.python_executor.tools._response_builder import (
        SUBSYSTEM,
        _raise_failure,
    )

    payload: dict[str, Any] = {"tool": TOOL_NAME, "parse_error": str(error)}
    saved = _measurement_file(kind, folder)
    message = f"The measurement script ended without a usable result: {error}"
    if saved is not None and saved.is_file():
        payload["data_file"] = str(saved)
        message = f"{message.rstrip('.')}. The measurement completed; its data is saved at {saved}."
    if exec_result.success:
        make_error(
            "execution_error",
            message,
            ["Report this; the measurement script and its parser disagree."],
            details={**payload, "subsystem": SUBSYSTEM, "kind": "script_error"},
        )
    written = list(exec_result.written_channels)
    payload["written_channels"] = written
    stderr = exec_result.stderr or ""
    _raise_failure(exec_result, f"{_attempted(written)}\n{stderr}".rstrip(), payload)
    raise AssertionError("unreachable: _raise_failure always raises")


#: How many attempted channels a failure names before it gives only their count.
_MAX_NAMED_CHANNELS = 20


def _attempted(channels: list[str]) -> str:
    """The sentence a failed run's envelope says about the channels it attempted."""
    if not channels:
        return "The run ended before it could list its setpoints; it attempted no writes."
    named = (
        ", ".join(channels) if len(channels) <= _MAX_NAMED_CHANNELS else f"{len(channels)} channels"
    )
    return (
        "The run ended before it could list its setpoints; it attempted writes to "
        f"{named}, left where the run stopped."
    )


def _measurement_file(kind: str, folder: Path | None) -> Path | None:
    """Where the script of a ``kind`` run in ``folder`` saves its measurement."""
    if folder is None:
        return None
    return Path(folder).resolve() / f"pyaml_measure_{kind}.json"


def _save_measurement(result: dict[str, Any], args: MeasureArgs, folder: Path | None) -> None:
    """Copy a completed run's saved measurement into the artifact store.

    The file read is the one the run's own execution folder holds under the
    script's fixed name, never a path the script printed: a printed
    ``data_file`` naming any other path is refused, nothing is read or saved,
    and the result says so under ``data_file_refused``.
    """
    printed = result.get("data_file")
    if not printed:
        return
    path = _measurement_file(args.kind, folder)
    if path is None or str(printed) != str(path):
        result["data_file"] = None
        result["data_file_refused"] = (
            f"measurement file not saved: the run named {printed}, "
            f"not pyaml_measure_{args.kind}.json in its own execution folder"
        )
        return
    try:
        from osprey.stores.artifact_manifest import read_collected_file
        from osprey.stores.artifact_store import get_artifact_store

        entry = get_artifact_store().save_file(
            file_content=read_collected_file(path),
            filename=path.name,
            artifact_type="json",
            title=f"pyAML {args.kind} measurement of {args.model}",
            description=_description(args),
            mime_type="application/json",
            tool_source=TOOL_NAME,
            metadata={
                "kind": args.kind,
                "model": args.model,
                "modulation": args.modulation,
                "by": args.by,
                "sha256": result.get("sha256"),
            },
        )
    except Exception:
        logger.warning("Could not save %s to the artifact store", path, exc_info=True)
        return
    result["artifact_id"] = entry.id


async def pyaml_measure(
    kind: Literal["orm", "dispersion", "trm", "crm", "chromaticity_monitor"],
    model: str | None = None,
    correctors: list[str] | None = None,
    modulation: Literal["bipolar", "unipolar"] | None = None,
    by: Literal["magnet", "group"] | None = None,
    approved_target: str | None = None,
    approved_view_sha256: str | None = None,
) -> str:
    """Measure a response matrix or the chromaticity with pyAML on the control target.

    Steps setpoints on the machine inside the control target's run lock, so
    every call is approval-gated and runs in read-write mode. A run that
    stops, fails or is interrupted leaves each setpoint where it stopped;
    every result lists the setpoints the call moved with their values before
    the call (``moved``). The model's pyAML
    view is ``data/pyaml/<model>/configuration.yaml``; each step is sized so
    no setpoint's largest single write exceeds half its limits.yaml max_step.

    Kinds:
      * ``orm`` — orbit response of the corrector setpoints;
      * ``dispersion`` — orbit response to the RF frequency;
      * ``trm`` — tune response of the quadrupoles;
      * ``crm`` — chromaticity response of the sextupoles (steps the RF);
      * ``chromaticity_monitor`` — the chromaticity [xi_x, xi_y] (steps the RF).

    Args:
        kind: The measurement.
        model: The model measured; omit when exactly one model has a pyAML view.
        correctors: Corrector setpoint addresses to step, ``orm`` only; omit for
            every corrector.
        modulation: ``bipolar`` steps each setting to both sides of its present
            value, ``unipolar`` to one; omit for bipolar, or for unipolar with
            ``by`` ``group``.
        by: ``magnet`` steps one magnet per column, ``group`` steps every magnet
            of the group by the same integrated strength as one column
            (``trm`` and ``crm`` only); group stepping is unipolar and refuses
            ``bipolar``; omit for magnet.
        approved_target: Set by the approval hook; a value you pass is
            overwritten.
        approved_view_sha256: Set by the approval hook; a value you pass is
            overwritten.

    Returns:
        JSON summary: ``kind``, ``model``, ``completed``, ``sha256`` of the view,
        ``modulation`` (and ``by``), the steps used by address or group, and for
        a completed run ``rows`` and ``columns`` named by address and device id,
        ``shape``, ``data_file`` and ``artifact_id``; a ``data_file`` that is not
        the run's own measurement file is answered as ``null`` with
        ``data_file_refused``; every result has ``moved``, each entry
        ``{address, device, before}``. A stopped run returns its
        ``remainder`` and, when an error stopped it, ``error``.
    """
    from osprey.mcp_server.python_executor.tools._response_builder import _note_writes
    from osprey.utils.config_paths import resolve_config_dir
    from osprey.utils.workspace import load_osprey_config

    enforce_posture_clamp(_EXECUTION_MODE, tool=TOOL_NAME)

    config_dir = resolve_config_dir()
    if config_dir is None:
        _refuse("no rendered config.yml to read the pyAML view from", "Run `osprey build`.")
    config = load_osprey_config()
    args = resolve_measure_args(
        kind,
        model,
        correctors,
        modulation,
        by,
        config_path=Path(config_dir) / _CONFIG_FILE,
        timeout=_timeout(config),
    )
    script = build_measure_script(_script(args, approved_view_sha256))

    exec_result, _patterns = await run_gated_execution(
        tool=TOOL_NAME,
        code=script,
        description=_description(args),
        execution_mode=_EXECUTION_MODE,
        notify=notify_agent_activity_async,
        approved_target=approved_target,
        approved_view_sha256=approved_view_sha256,
    )
    _note_writes(exec_result)

    try:
        result = parse_measure_result(exec_result.stdout, args.kind)
    except MeasureResultError as error:
        _raise_unparsed(exec_result, error, args.kind, exec_result.execution_folder)

    if "refused" in result:
        make_error(
            "validation_error",
            result["refused"],
            ["Nothing was stepped; the measurement refused before any setpoint moved."],
            details={"kind": args.kind, "model": args.model},
        )

    labelled = label_result(result, measure_labels(args.render_root, args.model, args.config_path))
    _save_measurement(labelled, args, exec_result.execution_folder)
    return json.dumps(labelled)
