"""
Execution Wrapper System

Wraps agent-generated Python code for execution in a host subprocess.
"""

import textwrap
from collections.abc import Iterable
from pathlib import Path

from osprey.services.python_executor.execution.fs_guard import (
    DEFAULT_DENYLIST_PREFIX,
    EXECUTOR_PATCH_TARGETS,
    render_fs_guard,
)
from osprey.services.python_executor.execution.net_guard import render_net_guard

# The write surface the readonly guard installs. Imported rather than spelled
# here so that the guard and the readonly import denylist
# (``analysis.safety_checks``) are produced by one table and cannot describe
# different libraries. Re-exported by this module because every existing
# consumer — the guard tests included — reads it from here. The framework rows
# are named separately because they are the rows the guard waits for: they are
# patched when the script imports their module, not before it runs.
from osprey.services.python_executor.write_surface import (
    _ARMED_BLOCKED,
    _ARMED_CA_PROVIDER,
    _ARMED_CHECKED,
    _ARMED_RPC,
    _FRAMEWORK_WRITE_TARGETS,
    _READONLY_WRITE_TARGETS,
)
from osprey.utils.logger import get_logger
from osprey_connectors.errors import RAW_CLIENT_WRITE_MARKER

logger = get_logger("execution_wrapper")


#: Message raised by every refused write in a readonly run. Tests match on
#: it, so keep it stable. The MCP tool layer also matches on it to recognise a
#: runtime refusal in the subprocess's stderr, so that a write blocked *during*
#: execution reaches the operator alert and the audit log the same way one
#: blocked before execution does.
READONLY_REFUSAL = (
    "readonly execution mode: control-system writes are refused — "
    "resubmit with execution_mode='readwrite' (human approval required) "
    "if the write is intended, and write through "
    "osprey.runtime.write_channel(address, value)"
)

#: The substring every readonly refusal message carries, whichever layer
#: raised it. The wrapper guard raises :data:`READONLY_REFUSAL`; the connector
#: reference monitor raises its own, channel-named message
#: (``osprey_connectors.control_system.base._writes_disabled_result``). Both
#: reach the tool layer only as a traceback on the subprocess's stderr, so the
#: tool matches on what they share rather than on either full message — which
#: is what lets a write refused through the *approved* ``write_channel`` path
#: be alerted and audited exactly like one refused by the guard.
#:
#: A test pins the connector's message against this constant, so rewording
#: either side without the other fails rather than silently stopping the alert.
READONLY_REFUSAL_MARKER = "readonly execution mode"

#: File in the execution folder where a readwrite run records every channel
#: ``osprey.runtime`` attempted to write, one ``{"channel": <address>}`` JSON
#: object per line. Appended per attempt, so a killed run leaves the list of
#: the writes it reached.
WRITES_LEDGER_FILENAME = "writes.jsonl"

#: Refusal prefix the filesystem guard carries in a readonly run. It embeds
#: :data:`READONLY_REFUSAL_MARKER` so that a write refused into the render zone
#: or the profile sources, and an open refused on a secret file, reaches the
#: operator alert and the audit ledger by the same path a refused control-system
#: write does — ``report_runtime_refusal`` scans the subprocess's stderr for that
#: marker and nothing else.
READONLY_FS_REFUSAL_PREFIX = f"Refused ({READONLY_REFUSAL_MARKER}):"

#: The same refusal in a readwrite run. It names the protected path and says
#: nothing about the mode: the run *is* readwrite, and telling the agent to
#: "resubmit with execution_mode='readwrite'" would be advice it has already
#: taken. Deliberately NOT carrying the readonly marker — see
#: :meth:`ExecutionWrapper._get_filesystem_guard` for what that costs and why it
#: is still the right trade.
READWRITE_FS_REFUSAL_PREFIX = DEFAULT_DENYLIST_PREFIX

#: Source of the raw-put block the readonly guard embeds. Read by path rather
#: than imported, so that emitting the guard never imports ``osprey.runtime``.
_RAW_PUT_BLOCK_SOURCE = Path(__file__).resolve().parents[3] / "runtime" / "raw_put_block.py"

#: Filename the embedded block is compiled under, so a traceback through it
#: names the module it came from.
_RAW_PUT_BLOCK_FILENAME = "osprey/runtime/raw_put_block.py"

#: Refusal a PVAccess rpc — p4p's ``Context.rpc`` or pvaPy's ``RpcClient.invoke``
#: — raises in a limits-checked readwrite run. An rpc carries an arbitrary
#: payload and no channel value, so no connector write can stand in for it and
#: no limit can bound it.
PVA_RPC_REFUSAL = "rpc is not mediated and cannot be approved — use the supervised write path"

#: Refusal a Tango command raises in a limits-checked readwrite run: a command
#: is an action on the device, not a channel value a limit could bound.
TANGO_COMMAND_REFUSAL = (
    "Tango command refused in a limits-checked run: a command carries no value to bound"
)

#: The rpc refusal texts the armed block receives, keyed by dotted prefix. The
#: block picks the text keyed by the longest prefix of ``dotted.attr``, so these
#: four keys cover every row of :data:`_ARMED_RPC`; a row they did not cover
#: would make the install raise rather than refuse with nothing to say.
ARMED_RPC_REFUSALS: dict[str, str] = {
    "p4p": PVA_RPC_REFUSAL,
    "pvaccess": PVA_RPC_REFUSAL,
    "tango": TANGO_COMMAND_REFUSAL,
    "PyTango": TANGO_COMMAND_REFUSAL,
}


def _armed_rows(table: dict[tuple[str, str], str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Group a ``(dotted, attr)``-keyed write-surface table into install rows.

    Rows keep the table's order: first by the first appearance of each dotted
    owner, then each owner's attributes as the table lists them.
    """
    rows: dict[str, list[str]] = {}
    for dotted, attr in table:
        rows.setdefault(dotted, []).append(attr)
    return tuple((dotted, tuple(attrs)) for dotted, attrs in rows.items())


def armed_contract(*, refuse_rpc: bool) -> dict:
    """The keyword arguments of ``raw_put_block.install("armed", ...)``.

    One place computes them, so every process that arms the block — the
    executor's generated script and the notebook kernel alike — refuses the
    same client entry points with the same marker.
    """
    return {
        "blocked_targets": _armed_rows(_ARMED_BLOCKED),
        "rpc_targets": _armed_rows(_ARMED_RPC),
        "refuse_rpc": bool(refuse_rpc),
        "marker": RAW_CLIENT_WRITE_MARKER,
        "rpc_refusals": dict(ARMED_RPC_REFUSALS),
        "ca_provider_targets": _armed_rows(
            {row: reason for row, reason in _ARMED_CHECKED.items() if row[0] in _ARMED_CA_PROVIDER}
        ),
    }


def _embedded_raw_put_block(comment: Iterable[str], mode: str, contract: dict) -> str:
    """Script lines that run ``raw_put_block.install(mode, **contract)`` in a private namespace.

    The block's *source* is embedded and executed rather than imported, so the
    script holds in an interpreter where ``osprey`` cannot be imported and no
    name the block defines leaks into the user code. *contract* is emitted as
    literals, one keyword per line.
    """
    source = _RAW_PUT_BLOCK_SOURCE.read_text(encoding="utf-8")
    return "\n".join(
        (
            *comment,
            "_ns = {}",
            f'exec(compile({source!r}, {_RAW_PUT_BLOCK_FILENAME!r}, "exec"), _ns)',
            '_ns["install"](',
            f'    "{mode}",',
            *(f"    {name}={value!r}," for name, value in contract.items()),
            ")",
            "del _ns",
        )
    )


class ExecutionWrapper:
    """
    Wrapper system for subprocess Python execution.

    Creates wrapped Python scripts with:
    - Standard imports and setup
    - Context loading
    - Output capture
    - Results export
    - Error handling
    """

    def __init__(
        self,
        limits_validator=None,
        execution_mode: str = "readonly",
        protected_roots: Iterable[str | Path] = (),
        permitted_roots: Iterable[str | Path] = (),
        perimeter_denied_ports: Iterable[int] = (),
        secret_roots: Iterable[str | Path] = (),
    ):
        """
        Initialize the wrapper.

        Args:
            limits_validator: Optional LimitsValidator instance for channel checking
            execution_mode: The mode the script was submitted under. A
                ``"readonly"`` run gets the readonly guard (every direct
                control-system write entry point refuses at runtime); the
                default is readonly so a wrapper built without a mode fails
                closed. It does **not** decide whether the filesystem guard is
                installed — that one is unconditional — only how its refusals
                are worded.
            protected_roots: Absolute, already-resolved paths executed code may
                not write into, in either mode: the render zone and the profile
                source set. This is the self-change boundary, not a
                control-system write gate, which is why the mode does not enter
                into it. Resolved by the parent
                (:func:`osprey.mcp_server.python_executor.executor.resolve_protected_roots`)
                and baked into the emitted guard as literals — a child that
                re-derived them could be pointed at different ones by the very
                code the guard exists to contain. Empty leaves the guard
                installed and refusing nothing, which is what a caller that
                knows no project layout (a unit test, a bare ``ExecutionWrapper()``)
                should get.
            secret_roots: Absolute, already-resolved directories where the env
                chain lives. A ``.env`` or ``.env.*`` file at any depth under
                one of them may not be opened by executed code, for read or
                write, in any mode. Resolved by the parent
                (:func:`osprey.mcp_server.python_executor.executor.resolve_secret_roots`)
                and baked in as literals, like ``protected_roots``. Empty still
                refuses ``/proc/<...>/environ`` and ``/proc/<...>/cmdline``.
            permitted_roots: Absolute, already-resolved paths carved back out of
                the protected set — the agent's own data zone. The execution
                folder is added to this in :meth:`_get_filesystem_guard`, since
                that is the one root the wrapper knows and the parent does not
                until the folder exists.
            perimeter_denied_ports: Host ports on this machine that executed code
                may not connect to — the web ports of a deployment whose
                perimeter authenticates on the caller's behalf, where a request
                made from inside a terminal container would arrive already
                credentialed as whoever owns the port. Resolved by the parent
                (:func:`osprey.mcp_server.python_executor.executor._perimeter_denied_ports`,
                which reads the deployment's stamp) and passed as literals for
                the same reason ``protected_roots`` is: a child that re-derived
                the set could equally derive an empty one. Empty — the default,
                and what every non-open posture yields — means no ports are
                denied and no network guard is emitted at all
                (:meth:`_get_net_guard`); a non-empty set puts the guard in
                front of user code in **every** execution mode, because the
                perimeter is orthogonal to the write posture.
        """
        self.limits_validator = limits_validator
        self.execution_mode = execution_mode
        self.protected_roots = tuple(str(root) for root in protected_roots)
        self.permitted_roots = tuple(str(root) for root in permitted_roots)
        self.secret_roots = tuple(str(root) for root in secret_roots)
        self.perimeter_denied_ports = tuple(perimeter_denied_ports)

    def create_wrapper(self, user_code: str, execution_folder: Path | None = None) -> str:
        """
        Create complete wrapped Python script.

        Args:
            user_code: Clean user code to execute
            execution_folder: Optional execution directory

        Returns:
            Complete wrapped Python script
        """

        # Build wrapper components
        imports = self._get_imports()
        environment_setup = self._get_environment_setup(execution_folder)
        write_ledger_observer = self._get_write_ledger_observer(execution_folder)
        limits_checking = self._get_limits_checking_monkeypatch()
        pva_limits_guard = self._get_pva_limits_guard()
        readonly_guard = self._get_readonly_guard()
        armed_block = self._get_armed_block()
        filesystem_guard = self._get_filesystem_guard(execution_folder)
        net_guard = self._get_net_guard()
        metadata_init = self._get_metadata_init()
        save_artifact_injection = self._get_save_artifact_injection()
        output_capture_start = self._get_output_capture_start()
        user_code_section = self._wrap_user_code(user_code)
        cleanup_and_export = self._get_cleanup_and_export()

        # Assemble complete wrapper
        wrapped_code = "\n".join(
            [
                imports,
                environment_setup,
                write_ledger_observer,
                limits_checking,
                pva_limits_guard,
                readonly_guard,
                armed_block,
                filesystem_guard,
                net_guard,
                metadata_init,
                save_artifact_injection,
                output_capture_start,
                user_code_section,
                cleanup_and_export,
            ]
        )

        return wrapped_code

    def _get_imports(self) -> str:
        """Get standard imports."""
        imports = """
# Standard imports for agent execution
import sys
import json
import os
import time
import traceback
from pathlib import Path
from io import StringIO
from datetime import datetime as _datetime, timedelta
import pickle


# Scientific libraries
try:
    import numpy as np
except ImportError:
    print("NumPy not available")

try:
    import pandas as pd
except ImportError:
    print("Pandas not available")

try:
    import matplotlib.pyplot as plt
    # Configure matplotlib for non-interactive use
    plt.switch_backend('Agg')
except ImportError:
    print("Matplotlib not available")
"""

        return textwrap.dedent(imports).strip()

    def _get_environment_setup(self, execution_folder: Path | None) -> str:
        """Get subprocess environment setup code (sys.path, registry init)."""

        setup = """
# Local execution environment setup
import sys
import os
from pathlib import Path

# Add framework src directory to Python path
current_path = Path.cwd()
project_root = None

# Find project root by looking for src/osprey
for parent in [current_path] + list(current_path.parents):
    src_dir = parent / "src"
    if src_dir.exists() and (src_dir / "osprey").exists():
        project_root = parent
        break

if project_root:
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✅ Added framework path to sys.path: {src_path}")
else:
    print("⚠️ Could not locate framework src directory")

# IMPORTANT: Also add the application's src directory to Python path
# This is needed for the registry to import a deployment's own modules --
# whatever packages live under the `src/` tree beside its config.yml.
# Note: config_file is reused below for registry initialization
config_file = os.environ.get('CONFIG_FILE')
if config_file:
    config_dir = Path(config_file).parent
    app_src_dir = config_dir / "src"
    if app_src_dir.exists():
        app_src_path = str(app_src_dir)
        if app_src_path not in sys.path:
            sys.path.insert(0, app_src_path)
            print(f"✅ Added application src path to sys.path: {app_src_path}")
    else:
        print(f"⚠️ Application src directory not found at: {app_src_dir}")

# Initialize registry for context loading
# Uses CONFIG_FILE environment variable for proper path resolution in subprocesses
try:
    from osprey.registry import initialize_registry
    initialize_registry(auto_export=False, config_path=config_file)
    print("✅ Registry initialized successfully")
except Exception as e:
    print(f"Registry initialization failed: {e}", file=sys.stderr)
    print("Context loading may not work properly", file=sys.stderr)
"""

        # Set execution directory variable (but do NOT chdir — user code
        # needs cwd to be the project root so relative workspace paths work)
        if execution_folder:
            setup += f"""
# Execution directory for wrapper outputs (results, figures, artifacts).
# User code cwd stays at the project root so relative paths like
# "_agent_data/data/002_archiver_read.json" resolve correctly.
_execution_dir = Path(r"{execution_folder}")
if not _execution_dir.exists():
    print(f"Warning: Execution directory {{_execution_dir}} does not exist")
"""

        return textwrap.dedent(setup).strip()

    def _get_write_ledger_observer(self, execution_folder: Path | None) -> str:
        """Generate the write-ledger observer registration; empty for a readonly run.

        A readwrite run registers an ``osprey.runtime`` write observer that
        appends ``{"channel": <address>}`` to :data:`WRITES_LEDGER_FILENAME` in
        the execution folder on every ``attempt`` phase. Each line is written
        and closed on its own, so the file holds every attempt made before the
        child was killed. A readonly run cannot write through the runtime, and a
        run without an execution folder has nowhere for the parent to read the
        ledger from, so neither gets the section. Without an importable
        ``osprey.runtime`` there is no runtime write to record, and the section
        does nothing.
        """
        if self.execution_mode == "readonly" or execution_folder is None:
            return ""
        ledger_path = Path(execution_folder) / WRITES_LEDGER_FILENAME
        return textwrap.dedent(
            f"""
            # Write ledger: every channel osprey.runtime attempts to write
            try:
                import osprey.runtime as _write_ledger_runtime

                _write_ledger_path = r"{ledger_path}"

                def _write_ledger_observer(address, phase):
                    if phase != "attempt":
                        return
                    with open(_write_ledger_path, "a", encoding="utf-8") as _ledger:
                        _ledger.write(json.dumps({{"channel": str(address)}}) + "\\n")

                _write_ledger_runtime._register_write_observer(_write_ledger_observer)
            except ImportError:
                pass
            """
        ).strip()

    def _get_limits_checking_monkeypatch(self) -> str:
        """Rebuild the run's limits validator and hand it to ``osprey.runtime``.

        The sandbox holds no config, so the limits database and the policy
        travel in the generated source as literals. The rebuilt validator is
        injected into ``osprey.runtime``, so ``write_channel`` checks the same
        limits the parent resolved for this run's control target. Direct client
        puts are not wrapped here: in a readwrite run the armed raw-put block
        (:meth:`_get_armed_block`) refuses them before they reach the network,
        except the PVAccess puts :meth:`_get_pva_limits_guard` checks.
        """
        if self.limits_validator is None:
            return ""  # No limits checking

        import json

        # Serialize limits database to JSON
        limits_db_serialized = {}
        for channel_name, config in self.limits_validator.limits.items():
            limits_db_serialized[channel_name] = {
                "min_value": config.min_value,
                "max_value": config.max_value,
                "max_step": config.max_step,  # IMPORTANT: Include max_step for serialization
                "writable": config.writable,
            }

        db_json = json.dumps(limits_db_serialized)
        policy_json = json.dumps(self.limits_validator.policy)

        return textwrap.dedent(
            f"""
            # Runtime Channel Limits Checking (Embedded Config)
            try:
                import json
                from osprey.connectors.control_system.limits_validator import (
                    LimitsValidator, ChannelLimitsConfig
                )

                # Deserialize embedded config
                _limits_db_raw = json.loads('''{db_json}''')
                _policy = json.loads('''{policy_json}''')

                # Reconstruct limits database
                _limits_db = {{}}
                for channel_name, config_dict in _limits_db_raw.items():
                    _limits_db[channel_name] = ChannelLimitsConfig(
                        channel_address=channel_name,
                        min_value=config_dict.get('min_value'),
                        max_value=config_dict.get('max_value'),
                        max_step=config_dict.get('max_step'),  # Include max_step from serialized config
                        writable=config_dict.get('writable', True)
                    )

                # Create validator with embedded config
                _limits_validator = LimitsValidator(_limits_db, _policy)
                print("🛡️  Runtime channel limits checking ENABLED")

                # IMPORTANT: Also inject validator into osprey.runtime module
                # This ensures write_channel() uses the same embedded validator
                try:
                    import osprey.runtime as _runtime_module
                    _runtime_module._limits_validator = _limits_validator
                    print("✅ Injected limits validator into osprey.runtime")
                except ImportError:
                    print("ℹ️  osprey.runtime not available for limits injection")
            except Exception as e:
                print(f"⚠️  Limits checking setup failed: {{e}}")
                import traceback
                traceback.print_exc()
        """
        ).strip()

    def _get_pva_limits_guard(self) -> str:
        """Limits-check raw PVAccess puts in a readwrite run; empty otherwise.

        The connector reads PVAccess but does not write it yet, so a raw
        ``p4p`` or ``pvaccess`` put is the one PVAccess write route a readwrite
        run has, and the armed raw-put block lets it through
        (:data:`_ARMED_CHECKED`) instead of refusing it — on a PVAccess channel.
        A pvaPy channel opened on Channel Access is refused by that block before
        this check is reached. It keeps the check it
        had before that block existed: every put is validated against the
        run's limits before it reaches the network. With no limits validator
        there is nothing to check against, and a readonly run refuses these
        puts outright, so both emit nothing.

        Emitted after :meth:`_get_limits_checking_monkeypatch`, whose
        ``_limits_validator`` it reads; if that setup failed the guard installs
        nothing and says so. The p4p flavours and pvaPy are each patched in
        their own ``try`` so one missing client does not skip the rest.
        """
        if self.limits_validator is None or self.execution_mode != "readwrite":
            return ""
        return textwrap.dedent(
            """
            # PVAccess puts: limits-checked, not refused, until the connector
            # writes PVAccess (write_surface._ARMED_CHECKED).
            if "_limits_validator" not in globals():
                print("⚠️  PVAccess limits guard not installed: limits checking setup failed")
            else:
                import inspect as _inspect
                import json as _json

                # p4p / PVAccess clients: p4p ships parallel Context classes per
                # concurrency flavor, so each one is imported and patched in its
                # OWN try/except - patching only the thread client would leave an
                # approved `from p4p.client.asyncio import Context` put unvalidated.
                def _p4p_current_value(_context):
                    '''A reader for the max_step check, bound to the putting context.

                    The step is measured over the same PVA context the put
                    goes through, so it follows that client's own addressing.
                    NTScalar and friends carry the number in a ``value``
                    field; a bare scalar answers itself.

                    p4p's asyncio flavour answers get() with a coroutine, and
                    the validator is synchronous — there is no read to make
                    here, so this answers None and a max_step channel on that
                    flavour fails closed. Calling get() anyway would hand the
                    validator an un-awaited coroutine and refuse the write with
                    a TypeError about it.
                    '''
                    _get = getattr(_context, 'get', None)
                    if _get is None or _inspect.iscoroutinefunction(_get):
                        return None

                    def _read(_address):
                        _current = _get(_address)
                        return getattr(_current, 'value', _current)

                    return _read

                def _p4p_reduce(_value):
                    '''Reduce a p4p put payload to the number the limits are about.

                    p4p takes the value FOUR ways: a bare scalar, a ``Value``
                    carrying it in a ``value`` field, a plain dict of fields,
                    and a JSON STRING of those fields. The limits database
                    holds numbers and the validator SKIPS every check it
                    cannot read a number for, so handing it a structure
                    unreduced would let an out-of-range value dressed as a
                    structure through unchecked.

                    The JSON string is the spelling that has to be read out of
                    p4p's own put() rather than its docstring: the flavour
                    client does ``json.loads`` on a payload whose first
                    character is '{' INSIDE put(), after this guard has
                    already run. Left as a str it fails the validator's
                    ``float()`` and skips every check, and p4p then decodes it
                    into the very number that was never checked. So it is
                    decoded HERE, on p4p's own rule, and checked as the dict it
                    becomes. Bytes are decoded too, which is one spelling
                    stricter than p4p (whose test never matches a bytes
                    payload) and errs closed.

                    A structure carrying no ``value`` field fails CLOSED with a
                    ValueError - dict, decoded JSON and ``Value`` alike, the
                    last recognised by p4p's own ``has()`` field protocol. That
                    is the same trade the batch guard makes for a payload it
                    cannot pair up; a payload whose ``has()`` is something else
                    entirely raises out of here, which refuses the write.

                    p4p also takes a BUILDER CALLABLE, which p4p invokes with a
                    blank Value only once the operation is under way. There is
                    no value here to check and no way to know the one the
                    callable will write, so the write is refused outright, in
                    range or not - the posture the pvaPy guard takes for
                    ``parsePut``.
                    '''
                    if callable(_value):
                        raise ValueError(
                            "p4p put with a builder callable cannot be "
                            "limits-checked; pass the value itself"
                        )
                    if isinstance(_value, (str, bytes)) and _value[:1] in ('{', b'{'):
                        try:
                            _value = _json.loads(_value)
                        except ValueError as _json_error:
                            raise ValueError(
                                "p4p put requires a value to limits-check: the "
                                "payload reads as a JSON structure but does "
                                "not decode"
                            ) from _json_error
                        if not isinstance(_value, dict):
                            raise ValueError(
                                "p4p put requires a value to limits-check: the "
                                "JSON payload is not a structure of fields"
                            )
                    if isinstance(_value, dict):
                        if 'value' not in _value:
                            raise ValueError(
                                "p4p put requires a value to limits-check: "
                                "the structure written carries no 'value' field"
                            )
                        return _value['value']
                    _has_field = getattr(_value, 'has', None)
                    if callable(_has_field) and not _has_field('value'):
                        raise ValueError(
                            "p4p put requires a value to limits-check: "
                            "the structure written carries no 'value' field"
                        )
                    return getattr(_value, 'value', _value)

                def _p4p_validate_put(_name, _values, _read_current):
                    '''Validate a p4p put payload BEFORE any network operation.

                    Discrimination mirrors p4p's OWN rule - a str name is the
                    scalar form, ANY other value is the batch form. Keying on
                    list/tuple instead would let an exotic iterable of names
                    (numpy array, generator) take the scalar path here while
                    p4p executed a batch, so per-channel bounds could be
                    skipped. A length mismatch fails closed via
                    zip(..., strict=True) rather than silently validating only
                    the shorter prefix, and a shape p4p would accept but we
                    cannot pair up fails closed via ValueError.

                    Every payload is reduced by ``_p4p_reduce`` before it
                    is checked, so a structure is checked on the number it
                    carries and a builder callable is refused.

                    ``_read_current`` is the putting context's own reader,
                    used only by channels that configure max_step.

                    Returns the name to forward to the original put(); a
                    one-shot iterable is materialized so validation does not
                    consume the caller's names.
                    '''
                    if isinstance(_name, str):
                        _limits_validator.validate(
                            _name, _p4p_reduce(_values), read_current=_read_current
                        )
                        return _name

                    if not isinstance(_values, (list, tuple)):
                        raise ValueError(
                            "p4p batch put requires a sequence of values "
                            "matching the sequence of channel names"
                        )

                    if isinstance(_name, (list, tuple)):
                        _names_seq = _name
                    else:
                        try:
                            _names_seq = tuple(_name)
                        except TypeError as _shape_error:
                            raise ValueError(
                                "p4p put requires a channel name string or a "
                                "sequence of channel names"
                            ) from _shape_error

                    for _pair_name, _pair_value in zip(_names_seq, _values, strict=True):
                        _limits_validator.validate(
                            _pair_name, _p4p_reduce(_pair_value), read_current=_read_current
                        )
                    return _names_seq

                def _p4p_install_guard(_context_cls):
                    '''Limits-check put() on one p4p Context class.

                    rpc() is not touched here: the armed raw-put block refuses
                    it in a limits-checked run, beside the Tango commands.
                    '''
                    if hasattr(_context_cls, 'put'):
                        _original_p4p_put = _context_cls.put

                        def _p4p_checked_put(self, name, values, *args, **kwargs):
                            '''Limits-checked wrapper for p4p Context.put()'''
                            name = _p4p_validate_put(
                                name, values, _p4p_current_value(self)
                            )  # Raises if invalid
                            return _original_p4p_put(self, name, values, *args, **kwargs)

                        _context_cls.put = _p4p_checked_put

                try:
                    from p4p.client.thread import Context as _P4PThreadContext

                    _p4p_install_guard(_P4PThreadContext)
                    print("✅ Monkeypatched p4p.client.thread Context.put()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.thread not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    # One flavor failing must NOT skip the remaining flavors,
                    # so this stops short of the outer swallow-all handler.
                    print(f"⚠️  p4p.client.thread guard failed: {_p4p_error}")

                try:
                    from p4p.client.asyncio import Context as _P4PAsyncioContext

                    _p4p_install_guard(_P4PAsyncioContext)
                    print("✅ Monkeypatched p4p.client.asyncio Context.put()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.asyncio not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    print(f"⚠️  p4p.client.asyncio guard failed: {_p4p_error}")

                try:
                    from p4p.client.cothread import Context as _P4PCothreadContext

                    _p4p_install_guard(_P4PCothreadContext)
                    print("✅ Monkeypatched p4p.client.cothread Context.put()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.cothread not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    print(f"⚠️  p4p.client.cothread guard failed: {_p4p_error}")

                # --- p4p's raw Context: the base every flavour above
                # subclasses, and an object a script can drive on its own. A
                # flavour put re-enters here through ``super().put`` once it
                # has already been validated, so the gate below checks only a
                # put whose receiver IS the raw Context - validating the
                # re-entry too would check each pair twice and buy a second
                # max_step read for the same write.
                try:
                    from p4p.client.raw import Context as _P4PRawContext

                    if hasattr(_P4PRawContext, 'put'):
                        _original_raw_put = _P4PRawContext.put

                        def _raw_gated_put(self, name, handler, builder=None, *args, **kwargs):
                            '''Limits-checked wrapper for p4p raw Context.put().

                            A raw put names ONE channel and spells its value
                            ``builder`` - a value, a structure, or a callable
                            p4p invokes later, which ``_p4p_reduce`` refuses
                            because there is nothing to check yet. The batch
                            form belongs to the flavour clients, which reach
                            this method one pair at a time.

                            A receiver of a SUBCLASS type is a flavour put
                            re-entering through ``super().put``, already
                            validated by the flavour wrapper, so it is
                            forwarded untouched. A subclass a script writes
                            itself is forwarded unchecked for the same reason,
                            which is why the write surface records this row as
                            guarded for the raw Context only.

                            The reader is the raw context's own, and raw
                            ``get`` answers through a handler rather than
                            returning a value - so a max_step channel written
                            straight through raw fails CLOSED rather than
                            being measured.
                            '''
                            if type(self) is not _P4PRawContext:
                                return _original_raw_put(
                                    self, name, handler, builder, *args, **kwargs
                                )

                            _limits_validator.validate(
                                name,
                                _p4p_reduce(builder),
                                read_current=_p4p_current_value(self),
                            )  # Raises if invalid
                            return _original_raw_put(
                                self, name, handler, builder, *args, **kwargs
                            )

                        _P4PRawContext.put = _raw_gated_put

                    print("✅ Monkeypatched p4p.client.raw Context.put()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.raw not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    print(f"⚠️  p4p.client.raw guard failed: {_p4p_error}")

                # --- pvaPy. A PVAccess client of its own, not a p4p spelling:
                # a ``pvaccess.Channel`` put never passes through a p4p
                # Context, so the flavour guards above leave it unchecked. The
                # binding spells one typed setter per scalar and array kind
                # (putDouble, putScalarArray, ...), so the family is swept by
                # PREFIX, the way the readonly guard sweeps it - enumerating
                # the names would go stale against the binding, and every name
                # it missed would be an unchecked write.
                try:
                    import pvaccess as _pvaccess

                    def _pva_reduce(_payload):
                        '''Reduce a pvaPy value to the number the limits are about.

                        pvaPy takes the value three ways: a ``PvObject``
                        structure, a plain dict of fields, and a bare scalar.
                        The limits database holds numbers, and the validator
                        SKIPS every check it cannot read a number for - so
                        handing it the structure unreduced would let an
                        out-of-range value dressed as a structure through
                        unchecked.

                        ``PvObject.getPyObject()`` answers the value under the
                        structure's ``value`` field and raises pvaPy's own
                        InvalidRequest when there is none, so a valueless
                        structure is refused THERE, before the dict branch is
                        reached. That branch is for a plain dict handed in
                        directly; one with no ``value`` key fails CLOSED with a
                        ValueError, the same trade the p4p batch guard makes
                        for a payload it cannot pair up.
                        '''
                        _get_py = getattr(_payload, 'getPyObject', None)
                        if _get_py is not None:
                            _payload = _get_py()
                        if isinstance(_payload, dict):
                            if 'value' not in _payload:
                                raise ValueError(
                                    "pvaccess put requires a value to limits-check: "
                                    "the structure written carries no 'value' field"
                                )
                            _payload = _payload['value']
                        if callable(_payload):
                            raise ValueError(
                                "pvaccess put requires a value to limits-check, "
                                "not a callable"
                            )
                        return _payload

                    _pva_channel_cls = getattr(_pvaccess, 'Channel', None)
                    _pva_original_get = getattr(_pva_channel_cls, 'get', None)

                    def _pva_current_value(_channel):
                        '''A reader for the max_step check, bound to the writing Channel.

                        The read goes through the ORIGINAL ``Channel.get`` and
                        the very channel the put is going through, so the step
                        is measured over that channel under its own addressing.
                        pvaPy answers a get with a structure, so the answer is
                        reduced exactly as the payload is - left unreduced it
                        is non-numeric, and the validator would SKIP the step
                        check rather than enforce it. A read that fails answers
                        None, which fails the step check closed.
                        '''
                        if _pva_original_get is None:
                            return None

                        def _read(_address):
                            try:
                                return _pva_reduce(_pva_original_get(_channel))
                            except Exception:
                                return None

                        return _read

                    def _pva_refuse_unreducible(_pva_name):
                        '''Refuse a write whose payload cannot be reduced.

                        ``parsePut``/``parsePutGet`` take a LIST OF JSON
                        strings, parsed against the channel's introspected
                        structure - not a value object. There is nothing to
                        reduce, and nothing that says which string carries the
                        field the limits are about; guessing would be a check
                        that silently means nothing. So the write is refused
                        outright, in range or not, the same posture the p4p
                        guard takes for a callable payload.
                        '''
                        def _refuse(self, *args, **kwargs):
                            raise ValueError(
                                "pvaccess " + _pva_name + " cannot be "
                                "limits-checked; use put"
                            )

                        return _refuse

                    def _pva_checked_put(_original_put):
                        '''Wrap ONE member of the put family.

                        The original is bound per attribute by this factory
                        because closing over the sweep's loop variable would
                        leave every wrapper calling whichever put was seen
                        last - one setter's writes going out under another
                        setter's typing.
                        '''
                        def _checked(self, value, *args, **kwargs):
                            _limits_validator.validate(
                                self.getName(),
                                _pva_reduce(value),
                                read_current=_pva_current_value(self),
                            )  # Raises if invalid
                            return _original_put(self, value, *args, **kwargs)

                        return _checked

                    # pvaPy spells three more writes OUTSIDE the put prefix:
                    # asyncPut (a put with a completion callback, PvObject
                    # first, so the ordinary wrapper covers it) and
                    # parsePut/parsePutGet (JSON strings, refused above).
                    # Each is the same value reaching the machine, so each is
                    # swept with the family.
                    _pva_swept = []
                    if _pva_channel_cls is not None:
                        for _pva_attr in dir(_pva_channel_cls):
                            if not _pva_attr.startswith(
                                ('put', 'asyncPut', 'parsePut')
                            ):
                                continue
                            _pva_original_put = getattr(_pva_channel_cls, _pva_attr, None)
                            if not callable(_pva_original_put):
                                continue
                            if _pva_attr.startswith('parsePut'):
                                _pva_wrapper = _pva_refuse_unreducible(_pva_attr)
                            else:
                                _pva_wrapper = _pva_checked_put(_pva_original_put)
                            setattr(_pva_channel_cls, _pva_attr, _pva_wrapper)
                            _pva_swept.append(_pva_attr)

                    # The success line is the operator's only evidence the
                    # guard is on. A pvaccess without a Channel, or a Channel
                    # carrying no put, wrapped nothing and must not report
                    # itself guarded.
                    if _pva_swept:
                        print(
                            "✅ Monkeypatched pvaccess Channel "
                            "put*()/asyncPut()/parsePut*()"
                        )
                    elif _pva_channel_cls is None:
                        print("⚠️  pvaccess guard failed: no Channel class")
                    else:
                        print("⚠️  pvaccess guard failed: Channel has no put method")
                except ImportError:
                    print("ℹ️  pvaccess not available - pvaPy limits checking disabled")
                except Exception as _pva_error:
                    print(f"⚠️  pvaccess guard failed: {_pva_error}")

            """
        ).strip()

    def _get_readonly_guard(self) -> str:
        """Generate the readonly-run guard; empty for a readwrite run.

        The pre-execution regex sees only the standard spellings of a write.
        ``from epics import caput as _w`` evades it, so a readonly run is
        enforced here, at runtime: every entry point in
        :data:`_READONLY_WRITE_TARGETS` is replaced with a function that
        refuses. The guard is emitted *before* user code and *after* the limits
        monkeypatch, so an alias bound in the user code late-binds to the
        refusing function, and the refusal — not a limits check — is the first
        thing a write hits. It is deliberately independent of the limits
        validator: a deployment with limits checking off is exactly the one
        with nothing else standing behind the regex.

        The guard itself is :mod:`osprey.runtime.raw_put_block`. Its *source*
        is read by path and embedded, then executed in a private namespace, so
        the script stays self-contained — it holds in an interpreter where
        ``osprey`` cannot be imported — no guard name leaks into the user code,
        and this process never imports ``osprey.runtime``, whose package import
        registers an exit handler. The tables are emitted as literals read from
        this module's globals at emission time; the framework rows are the
        deferred half, patched when the script imports their module.

        The connector side of the same contract lives in
        ``osprey_connectors.control_system.base`` (refuses ``write_channel``
        when ``OSPREY_EXECUTION_MODE`` says readonly) and in the EPICS
        connector's gateway selection (stays on the read_only gateway).
        """
        if self.execution_mode != "readonly":
            return ""
        deferred = _FRAMEWORK_WRITE_TARGETS
        eager = tuple(row for row in _READONLY_WRITE_TARGETS if row not in deferred)
        return _embedded_raw_put_block(
            (
                "# Readonly run: refuse every control-system write entry point, and",
                "# every route out of Python that could reach one. Installed before",
                "# user code, so an alias bound later resolves here.",
            ),
            "readonly",
            {"eager_targets": eager, "deferred_targets": deferred, "refusal": READONLY_REFUSAL},
        )

    def _get_armed_block(self) -> str:
        """Generate the readwrite-run raw-put block; empty for a readonly run.

        A readwrite run may write, but only through the connector, which checks
        limits and records the write. A raw client put — ``epics.caput``, a caproto
        ``PV.write``, a Tango ``write_attribute`` — is a route around that,
        so every entry point in :data:`_ARMED_BLOCKED` is replaced with one that
        calls through only while a connector holds the write door open and
        otherwise refuses with :data:`RAW_CLIENT_WRITE_MARKER`. A pvaPy channel
        opened on Channel Access is refused the same way, since its put is a raw
        Channel Access write (:data:`_ARMED_CA_PROVIDER`); one opened on
        PVAccess is left to :meth:`_get_pva_limits_guard`. With a limits
        validator the rpc rows of :data:`_ARMED_RPC` refuse as well: an rpc or a
        Tango command carries no value a limit could bound.

        It occupies the readonly guard's slot — exactly one of the two is
        emitted — and is independent of the limits validator, which only
        decides ``refuse_rpc``. The source is embedded and executed in a private
        namespace exactly as :meth:`_get_readonly_guard` does it. The install is
        not wrapped in ``try``: a block that cannot install aborts the script
        before the user code runs, rather than letting it run unguarded.
        """
        if self.execution_mode != "readwrite":
            return ""
        return _embedded_raw_put_block(
            (
                "# Readwrite run: a raw client put that no connector let through is",
                "# refused, so every write goes through the connector's checks.",
                "# Installed before user code, so an alias bound later resolves here.",
            ),
            "armed",
            armed_contract(refuse_rpc=self.limits_validator is not None),
        )

    def _get_filesystem_guard(self, execution_folder: Path | None) -> str:
        """Generate the runtime filesystem guard. Emitted in EVERY mode.

        This is a different boundary from the readonly guard above, and the two
        are deliberately not folded together. The readonly guard answers "may
        this run touch the control system"; this one answers "may this run
        rewrite the thing that builds it". A readwrite run has human approval to
        move a magnet — it has no approval to overwrite ``profile.yml`` or the
        render zone, and there is no mode in which it does. So the guard is
        installed unconditionally and the mode selects only the wording:

        * readonly → :data:`READONLY_FS_REFUSAL_PREFIX`, which carries
          :data:`READONLY_REFUSAL_MARKER`. That marker is what
          ``report_runtime_refusal`` scans the subprocess's stderr for, so a
          refused write into the render zone is alerted and written to the
          audit ledger exactly like a refused control-system write.
        * readwrite → :data:`READWRITE_FS_REFUSAL_PREFIX`, which names the
          protected path and makes no claim about the mode.

        **Readwrite refusals are not audited**, and that is a decision rather
        than an oversight. The audit path is reached only by the readonly
        marker, and the marker is a factual claim about the run: carrying it in
        a readwrite run would put "readonly execution mode" in front of an
        operator whose run was approved for writes, and would record the
        refusal under a layer that names the wrong gate ("BLOCKED a
        control-system write in readwrite mode"). The refusal itself still
        holds and still reaches the agent and the operator as the traceback in
        the run's stderr. Auditing a protected-path refusal in its own right
        wants its own layer and its own marker on both ends — the ledger's
        writer and the tool that matches it — which is a change to files this
        does not own.

        The same guard refuses any open of a secret file (a ``.env`` file under
        :attr:`secret_roots`, a ``/proc/<...>/environ`` or ``cmdline``), read or
        write, in both modes and with the mode's prefix. A readonly refusal
        therefore carries the marker and reaches ``report_runtime_refusal``; a
        readwrite one is refused and not audited, the same split writes have.

        The roots are resolved in the parent and interpolated as literals; the
        child never re-derives them. ``permitted_roots`` is checked before
        ``protected_roots`` by the renderer, which is what lets the execution
        folder and the agent-data zone keep taking writes while sitting under a
        project root whose render zone is refused.

        **This is defense in depth, not a security boundary.** The guard is
        emitted *into* the child and installs itself in the same module
        namespace as the user code, restore handle and all, so code that knows
        it is there disarms it in two lines::

            _restore_patched_targets()
            open('bui' + 'ld/x', 'w')     # unguarded

        Nothing rendered into the child can be hidden from the child, so that
        is a property of the approach rather than a bug in it, and a
        characterization test pins it
        (``tests/services/python_executor/test_fs_guard.py::TestTamperLimit``)
        so the limit stays stated rather than assumed. What this guard closes
        is the gap the static pre-execution walker cannot see — the
        concatenated or computed path, and the ordinary accident of writing one
        directory too high. What contains code that is deliberately attacking
        the boundary is the OS: the container's privilege split, where the
        render zone and the profile sources belong to a different user than the
        one executing agent code.

        Args:
            execution_folder: The run's own output directory, permitted so the
                wrapper's ``save_artifact`` and figure writes keep working. It
                is resolved here — it is the one root this method derives
                rather than receives.

        Returns:
            The guard source, ready to splice in ahead of the user code.
        """
        permitted = list(self.permitted_roots)
        if execution_folder is not None:
            permitted.append(str(Path(execution_folder).resolve()))

        prefix = (
            READONLY_FS_REFUSAL_PREFIX
            if self.execution_mode == "readonly"
            else READWRITE_FS_REFUSAL_PREFIX
        )

        return render_fs_guard(
            default_deny=False,
            permitted_roots=permitted,
            protected_roots=self.protected_roots,
            read_roots=(),
            patch_targets=EXECUTOR_PATCH_TARGETS,
            refusal_prefix=prefix,
            secret_roots=self.secret_roots,
        ).strip()

    def _get_net_guard(self) -> str:
        """Generate the perimeter network guard; empty when no ports are denied.

        Emitted in **every** execution mode, exactly like the filesystem guard
        and deliberately unlike the readonly guard: the mode answers "may this
        run touch the control system", while the perimeter answers "may this
        run talk to the deployment's own web edge" — a readwrite run has human
        approval to move a magnet, not to originate requests that nginx would
        credential on the caller's behalf. What gates emission is solely
        whether the parent handed this wrapper a non-empty deny-list, which
        only an open-perimeter deployment does.

        Splice position (kept deterministic by :meth:`create_wrapper`'s
        assembly list): directly **after** the filesystem guard and before the
        wrapper's own metadata/artifact machinery, so both guards form one
        contiguous block installed ahead of any code that could open a
        connection. Order between the two guards carries no dependency — they
        patch disjoint entry points — but a fixed position keeps the emitted
        script diffable. The readonly ``_READONLY_WRITE_TARGETS`` spawn/ctypes
        table is untouched by this guard: multiprocessing and h5py-style
        compute keep working in every mode, which the net-guard test suite
        pins with a real ``Pool``.

        Returns:
            The guard source ready to splice, or ``""`` when
            ``perimeter_denied_ports`` is empty — the renderer refuses an
            empty set rather than emitting an inert guard, so skipping here is
            the one honest spelling of "no perimeter is open".
        """
        if not self.perimeter_denied_ports:
            return ""
        return render_net_guard(denied_ports=self.perimeter_denied_ports).strip()

    def _get_metadata_init(self) -> str:
        """Initialize execution metadata tracking."""
        return textwrap.dedent(
            """
            # Execution metadata
            execution_metadata = {
                "start_time": _datetime.now().astimezone().isoformat(),
                "success": True,
                "error": None,
                "traceback": None,
                "stdout": "",
                "stderr": "",
                "error_type": None,
                "results_saved": False,
                "results_captured": False,  # Runtime validation flag
                "results_missing": False,   # Set to True if results not found
                "figures_saved": [],
                "figure_count": 0
            }
        """
        ).strip()

    def _get_save_artifact_injection(self) -> str:
        """The ``save_artifact()`` the subprocess exposes to user code.

        Shared with the visualization sandbox via
        :data:`osprey.stores.artifact_manifest.SAVE_ARTIFACT_SOURCE`; the
        executor collects what it wrote post-execution, mirroring the figure
        collection pattern.
        """
        from osprey.stores.artifact_manifest import SAVE_ARTIFACT_SOURCE

        return SAVE_ARTIFACT_SOURCE.strip()

    def _get_output_capture_start(self) -> str:
        """Start output capture for both environments."""
        return textwrap.dedent(
            """
            # Capture stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            try:
                # Redirect output streams
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
        """
        ).strip()

    def _wrap_user_code(self, user_code: str) -> str:
        """Execute user code directly (synchronous).

        User code is expected to be synchronous - osprey.runtime utilities
        handle async internally so generated code can be simple and straightforward.
        """
        # Indent user code (8 spaces = 2 levels, inside try block)
        indented_code = "\n".join(
            "        " + line if line.strip() else line for line in user_code.split("\n")
        )

        return f"""
    # Execute user code
    try:
{indented_code}

        # Mark successful execution
        execution_metadata["success"] = True
        execution_metadata["error_type"] = None
        execution_metadata["end_time"] = _datetime.now().astimezone().isoformat()

    except Exception as user_code_error:
        # Capture user code errors
        execution_metadata["success"] = False
        execution_metadata["error_type"] = type(user_code_error).__name__
        execution_metadata["error_message"] = str(user_code_error)
        execution_metadata["end_time"] = _datetime.now().astimezone().isoformat()
        raise
"""

    def _get_cleanup_and_export(self) -> str:
        """Generate the tail of the script: cleanup, persistence, and the exit.

        The ``except``/``finally`` of the output-capture block come first —
        the failure record, the restored streams, the captured output echoed
        to the real pipes, and the guards taken off. The persistence section
        then runs at module level: ``results.json``, the figures, and last of
        all ``execution_metadata.json``, the record the executor reads the
        run's outcome from.

        The process then leaves with :func:`os._exit`, the way
        ``osprey_connectors.ipc.host`` does, rather than through interpreter
        shutdown. A control-system client holds native state whose shutdown
        hooks can block or crash the process — pyepics' ``finalize_libca``
        wedges once Channel Access was used from a worker thread, which the
        EPICS connector always does — and a child that will not exit is
        reported by the executor as a timeout long after its script finished.
        Everything the executor reads is on disk or already flushed to the
        pipes by then, so the abrupt exit costs nothing; the exit code stays 0
        because the outcome is read from the record, not from the status.
        """

        # Output captured content so the host process can see it
        host_output_section = textwrap.dedent(
            """
            # Output captured content so host process can see it
            captured_stdout = stdout_capture.getvalue()
            captured_stderr = stderr_capture.getvalue()

            if captured_stdout:
                print(captured_stdout, end='')
            if captured_stderr:
                print(captured_stderr, file=sys.stderr, end='')
        """
        ).strip()

        # Be forgiving about metadata save failures: log, don't raise
        metadata_error_handling = textwrap.dedent(
            """
                print(f"ERROR: Failed to save execution metadata: {e}", file=sys.stderr)
                # Don't raise - just log the error
        """
        ).strip()

        # Build the complete code block properly
        base_cleanup = textwrap.dedent(
            """
            except Exception as e:
                execution_metadata["success"] = False
                execution_metadata["error"] = str(e)
                execution_metadata["traceback"] = traceback.format_exc()

                # Print detailed error information to console for immediate debugging
                print(f"\\n{'='*60}", file=sys.stderr)
                print(f"PYTHON EXECUTION ERROR", file=sys.stderr)
                print(f"{'='*60}", file=sys.stderr)
                print(f"Error Type: {type(e).__name__}", file=sys.stderr)
                print(f"Error Message: {str(e)}", file=sys.stderr)
                print(f"\\nFull Traceback:", file=sys.stderr)
                print(f"{traceback.format_exc()}", file=sys.stderr)
                print(f"{'='*60}\\n", file=sys.stderr)

            finally:
                # Restore stdout/stderr and capture output
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                execution_metadata["stdout"] = stdout_capture.getvalue()
                execution_metadata["stderr"] = stderr_capture.getvalue()
                execution_metadata["end_time"] = _datetime.now().astimezone().isoformat()

                # Switch to execution directory for file persistence (results,
                # figures, metadata).  User code ran with cwd=project_root;
                # cleanup outputs go to the execution sandbox.
                _exec_dir = globals().get('_execution_dir')
                if _exec_dir:
                    os.chdir(_exec_dir)
        """
        ).strip()

        file_persistence_section = textwrap.dedent(
            """
                # Import robust serialization function
                from osprey.services.python_executor.services import serialize_results_to_file

                # Runtime validation: Check if 'results' exists in globals
                if 'results' in globals():
                    execution_metadata["results_captured"] = True

                    if results is not None:
                        # Use robust serialization function
                        serialization_metadata = serialize_results_to_file(results, 'results.json')
                        execution_metadata["results_saved"] = serialization_metadata["success"]

                        if not serialization_metadata["success"]:
                            # Serialization failed, capture detailed error info
                            execution_metadata["results_save_error"] = serialization_metadata["error"]
                            if "fallback_saved" in serialization_metadata:
                                execution_metadata["fallback_results_saved"] = serialization_metadata["fallback_saved"]
                    else:
                        # results exists but is None
                        execution_metadata["results_captured"] = True
                        execution_metadata["results_is_none"] = True
                        print("⚠️  Warning: 'results' variable exists but is set to None", file=sys.stderr)
                else:
                    # results variable was never created
                    execution_metadata["results_captured"] = False
                    execution_metadata["results_missing"] = True
                    print("⚠️  Warning: Code did not create required 'results' variable", file=sys.stderr)
                    print("    Downstream code may expect a 'results' dictionary to be present", file=sys.stderr)

                # Save matplotlib figures
                try:
                    figure_nums = plt.get_fignums()
                    if figure_nums:
                        figures_dir = Path('figures')
                        figures_dir.mkdir(exist_ok=True)

                        for i, fig_num in enumerate(figure_nums):
                            try:
                                fig = plt.figure(fig_num)
                                figure_path = figures_dir / f'figure_{i+1:02d}.png'
                                fig.savefig(figure_path, dpi=100, bbox_inches='tight', facecolor='white')
                                execution_metadata["figures_saved"].append(str(figure_path))
                            except Exception as fig_error:
                                if "figure_errors" not in execution_metadata:
                                    execution_metadata["figure_errors"] = []
                                execution_metadata["figure_errors"].append(f"Figure {{i+1}}: {{str(fig_error)}}")

                        execution_metadata["figure_count"] = len(execution_metadata["figures_saved"])
                except Exception as e:
                    execution_metadata["figure_save_error"] = str(e)

                # Save execution metadata for debugging
                try:
                    # Use serializer for execution metadata
                    from osprey.services.python_executor.services import make_json_serializable
                    serializable_metadata = make_json_serializable(execution_metadata)

                    with open('execution_metadata.json', 'w', encoding='utf-8') as f:
                        json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
                except Exception as e:
        """
        ).strip()

        # The filesystem guard comes off at the END of the finally block, so it
        # covers the user code and the output-capture teardown and nothing
        # after: the persistence section below runs at module level on unpatched
        # entry points, which is what keeps a misjudged protected root from
        # costing the operator the execution record. Everything the guard *does*
        # cover writes into the execution folder, which is permitted anyway.
        guard_restore_section = textwrap.dedent(
            """
            # Filesystem guard: put every patched entry point back.
            _restore_patched_targets()

            # Network guard, same point for the same reason — looked up via
            # globals() because it is only emitted when the deployment's
            # perimeter denies ports, and this cleanup tail is shared by every
            # wrapper. The persistence section below touches only the local
            # filesystem, so nothing here depends on the restore; it exists so
            # both guards come off together at a single documented point.
            _osprey_net_restore = globals().get('_restore_net_patched_targets')
            if _osprey_net_restore is not None:
                _osprey_net_restore()
        """
        ).strip()

        # The exit wraps the whole persistence section so that every path
        # through the tail ends here: a record written, a record that could
        # not be written, or a persistence step that raised before the record
        # was reached. The last case is the one the interpreter would
        # otherwise report through a non-zero status, so it keeps that status
        # — with the traceback printed first, since ``os._exit`` prints
        # nothing.
        exit_section = textwrap.dedent(
            """
            except BaseException:
                traceback.print_exc()
                _osprey_exit_status = 1
            finally:
                # Leave without interpreter shutdown: a control-system client's
                # shutdown hooks can block or crash the process, and everything
                # the executor reads is persisted or flushed by now.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(_osprey_exit_status)
        """
        ).strip()

        def indent(block: str, spaces: int) -> str:
            pad = " " * spaces
            return "\n".join(pad + line if line.strip() else line for line in block.split("\n"))

        return "\n".join(
            [
                base_cleanup,
                # 4-space indent to sit inside the finally block
                indent(host_output_section, 4),
                indent(guard_restore_section, 4),
                "_osprey_exit_status = 0",
                "try:",
                indent(file_persistence_section, 4),
                indent(metadata_error_handling, 8),
                exit_section,
            ]
        )
