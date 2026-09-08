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
# consumer — the guard tests included — reads it from here.
from osprey.services.python_executor.write_surface import (
    _READONLY_WRITE_TARGETS,
)
from osprey.utils.logger import get_logger

logger = get_logger("execution_wrapper")


#: Message raised by every refused write in a readonly run. Tests match on
#: it, so keep it stable. The MCP tool layer also matches on it to recognise a
#: runtime refusal in the subprocess's stderr, so that a write blocked *during*
#: execution reaches the operator alert and the audit log the same way one
#: blocked before execution does.
READONLY_REFUSAL = (
    "readonly execution mode: control-system writes are refused — "
    "resubmit with execution_mode='readwrite' (human approval required) "
    "if the write is intended"
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

#: Refusal prefix the filesystem guard carries in a readonly run. It embeds
#: :data:`READONLY_REFUSAL_MARKER` so that a write refused into the render zone
#: or the profile sources reaches the operator alert and the audit ledger by the
#: same path a refused control-system write does — ``report_runtime_refusal``
#: scans the subprocess's stderr for that marker and nothing else.
READONLY_FS_REFUSAL_PREFIX = f"Refused ({READONLY_REFUSAL_MARKER}):"

#: The same refusal in a readwrite run. It names the protected path and says
#: nothing about the mode: the run *is* readwrite, and telling the agent to
#: "resubmit with execution_mode='readwrite'" would be advice it has already
#: taken. Deliberately NOT carrying the readonly marker — see
#: :meth:`ExecutionWrapper._get_filesystem_guard` for what that costs and why it
#: is still the right trade.
READWRITE_FS_REFUSAL_PREFIX = DEFAULT_DENYLIST_PREFIX


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
        limits_checking = self._get_limits_checking_monkeypatch()
        readonly_guard = self._get_readonly_guard()
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
                limits_checking,
                readonly_guard,
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
# This is needed for the registry to import application-specific modules
# (e.g., its_control_assistant.context_classes, als_assistant.capabilities, etc.)
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

    def _get_limits_checking_monkeypatch(self) -> str:
        """Generate monkeypatch code with embedded validator config."""
        if self.limits_validator is None:
            return ""  # No limits checking

        import json

        # Serialize limits database to JSON
        limits_db_serialized = {}
        for pv_name, config in self.limits_validator.limits.items():
            limits_db_serialized[pv_name] = {
                "min_value": config.min_value,
                "max_value": config.max_value,
                "max_step": config.max_step,  # IMPORTANT: Include max_step for serialization
                "writable": config.writable,
            }

        db_json = json.dumps(limits_db_serialized)
        policy_json = json.dumps(self.limits_validator.policy)

        return textwrap.dedent(
            f"""
            # Runtime Channel Limits Checking (Monkeypatch with Embedded Config)
            try:
                import json
                from osprey.connectors.control_system.limits_validator import (
                    LimitsValidator, ChannelLimitsConfig, STEP_READ_TIMEOUT_SECONDS
                )
                from osprey.errors import ChannelLimitsViolationError

                # Deserialize embedded config
                _limits_db_raw = json.loads('''{db_json}''')
                _policy = json.loads('''{policy_json}''')

                # Reconstruct limits database
                _limits_db = {{}}
                for pv_name, config_dict in _limits_db_raw.items():
                    _limits_db[pv_name] = ChannelLimitsConfig(
                        channel_address=pv_name,
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

                import inspect as _inspect
                import json as _json

                # pyepics spells a write three ways - caput(), PV.put() and
                # ca.put() - and the first two reach the network through the
                # third. Guarding the choke point alone validates every
                # spelling exactly once and pays for one max_step read, where
                # a wrapper per spelling validated the same write at each layer
                # it passed and still left a direct ca.put() unchecked.
                try:
                    from epics import ca as _epics_ca

                    _original_ca_put = _epics_ca.put
                    _original_ca_get = _epics_ca.get
                    _original_ca_name = _epics_ca.name

                    def _ca_current_value(_chid):
                        '''Read a channel's present value for the max_step check.

                        The script's OWN Channel Access client, captured before
                        the guard is installed - the validator holds no client
                        and must not reach for one. The read goes through the
                        very channel id the put is going through, so the step
                        is measured over the channel being written. A client
                        that cannot read answers None, which fails the step
                        check closed.
                        '''
                        try:
                            return _original_ca_get(_chid, timeout=STEP_READ_TIMEOUT_SECONDS)
                        except Exception:
                            return None

                    def _checked_ca_put(chid, value, *args, **kwargs):
                        '''Limits-checked wrapper for epics.ca.put()

                        A chid is opaque, so the channel the limits database is
                        keyed by has to be asked for by name - validating the
                        chid itself would look up a channel no database has
                        heard of.

                        A refusal here leaves pyepics' own pre-put state as
                        it found it: ``PV.put`` coerces an enum string to its
                        index and sets ``_put_complete`` before it reaches
                        ca.put, and neither is rolled back.
                        '''
                        _limits_validator.validate(
                            _original_ca_name(chid),
                            value,
                            read_current=lambda _address: _ca_current_value(chid),
                        )  # Raises if invalid
                        return _original_ca_put(chid, value, *args, **kwargs)

                    _epics_ca.put = _checked_ca_put
                    print("✅ Monkeypatched epics.ca.put()")

                except ImportError:
                    print("ℹ️  pyepics not available - EPICS limits checking disabled")

                # --- aioca. A second Channel Access client, and a separate
                # library: an `await aioca.caput(...)` never passes through
                # epics.ca.put, so the pyepics choke point above leaves it
                # unchecked. aioca's package namespace re-exports what
                # `aioca._catools` defines, so both spellings are rebound to
                # the same wrapper - patching only the re-export would leave
                # `from aioca._catools import caput` unguarded and would also
                # break the array form, whose per-pair re-entry goes through
                # the submodule global.
                try:
                    import aioca as _aioca
                    from aioca import _catools as _aioca_catools

                    _orig_aioca_caput = _aioca_catools.caput
                    _orig_aioca_caget = _aioca_catools.caget

                    async def _checked_aioca_caput(pv, value, *args, **kwargs):
                        '''Limits-checked wrapper for aioca.caput().

                        A non-str ``pv`` is aioca's array form, which aioca
                        dispatches to ``caput_array``. That function writes
                        each pair by calling the module-global ``caput`` -
                        this wrapper, since it is bound there - so forwarding
                        the sequence unchanged is what gets every pair
                        validated, once, under its own channel's limits.
                        Validating the sequence here instead would check a
                        list of values against one channel's limits and refuse
                        the write for the wrong reason.

                        The guard is a coroutine, so unlike the synchronous
                        clients it can await the max_step read itself and hand
                        the validator a plain value. That read is bought only
                        by a channel that configures max_step, and a read that
                        fails answers None, which fails the step check closed.
                        '''
                        if not isinstance(pv, str):
                            return await _orig_aioca_caput(pv, value, *args, **kwargs)

                        _cfg = _limits_validator.get_limits_config(pv)
                        _current = None
                        if _cfg and _cfg['max_step'] is not None:
                            try:
                                _current = await _orig_aioca_caget(
                                    pv, timeout=STEP_READ_TIMEOUT_SECONDS
                                )
                            except Exception:
                                _current = None

                        _limits_validator.validate(
                            pv, value, read_current=lambda _address: _current
                        )  # Raises if invalid
                        return await _orig_aioca_caput(pv, value, *args, **kwargs)

                    _aioca.caput = _checked_aioca_caput
                    _aioca_catools.caput = _checked_aioca_caput
                    print("✅ Monkeypatched aioca.caput()")
                except ImportError:
                    print("ℹ️  aioca not available - aioca limits checking disabled")
                except Exception as _aioca_error:
                    print(f"⚠️  aioca guard failed: {{_aioca_error}}")

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
                    character is '{{' INSIDE put(), after this guard has
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
                    if isinstance(_value, (str, bytes)) and _value[:1] in ('{{', b'{{'):
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

                def _p4p_blocked_rpc(self, *args, **kwargs):
                    '''Unconditional refusal for p4p Context.rpc().

                    Limits semantics cannot apply to an arbitrary rpc payload,
                    so refusal is the only honest parity. It is defined once
                    here rather than per class because the raw base refuses
                    with the same function the flavours do.
                    '''
                    raise RuntimeError(
                        "rpc is not mediated and cannot be approved — "
                        "use the supervised write path"
                    )

                def _p4p_install_guard(_context_cls):
                    '''Limits-check put() and refuse rpc() on one p4p Context class.'''
                    if hasattr(_context_cls, 'put'):
                        _original_p4p_put = _context_cls.put

                        def _p4p_checked_put(self, name, values, *args, **kwargs):
                            '''Limits-checked wrapper for p4p Context.put()'''
                            name = _p4p_validate_put(
                                name, values, _p4p_current_value(self)
                            )  # Raises if invalid
                            return _original_p4p_put(self, name, values, *args, **kwargs)

                        _context_cls.put = _p4p_checked_put

                    if hasattr(_context_cls, 'rpc'):
                        _context_cls.rpc = _p4p_blocked_rpc

                try:
                    from p4p.client.thread import Context as _P4PThreadContext

                    _p4p_install_guard(_P4PThreadContext)
                    print("✅ Monkeypatched p4p.client.thread Context.put()/.rpc()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.thread not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    # One flavor failing must NOT skip the remaining flavors,
                    # so this stops short of the outer swallow-all handler.
                    print(f"⚠️  p4p.client.thread guard failed: {{_p4p_error}}")

                try:
                    from p4p.client.asyncio import Context as _P4PAsyncioContext

                    _p4p_install_guard(_P4PAsyncioContext)
                    print("✅ Monkeypatched p4p.client.asyncio Context.put()/.rpc()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.asyncio not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    print(f"⚠️  p4p.client.asyncio guard failed: {{_p4p_error}}")

                try:
                    from p4p.client.cothread import Context as _P4PCothreadContext

                    _p4p_install_guard(_P4PCothreadContext)
                    print("✅ Monkeypatched p4p.client.cothread Context.put()/.rpc()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.cothread not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    print(f"⚠️  p4p.client.cothread guard failed: {{_p4p_error}}")

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

                    if hasattr(_P4PRawContext, 'rpc'):
                        _P4PRawContext.rpc = _p4p_blocked_rpc

                    print("✅ Monkeypatched p4p.client.raw Context.put()/.rpc()")
                except ImportError:
                    print(
                        "ℹ️  p4p.client.raw not available - "
                        "PVA limits checking disabled"
                    )
                except Exception as _p4p_error:
                    print(f"⚠️  p4p.client.raw guard failed: {{_p4p_error}}")

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
                    print(f"⚠️  pvaccess guard failed: {{_pva_error}}")

                # --- Tango. Its writes carry the attribute name only; the
                # channel a limits database is keyed by is the full
                # ``device/attribute`` address, so it is rebuilt from the
                # proxy's own device name. Each client below gets its OWN
                # try/except for the same reason the p4p flavours do: one
                # client absent or broken must not skip the ones after it.
                try:
                    import tango as _tango

                    def _tango_channel(_proxy, _attr):
                        '''The device/attribute address a write is bounded under.

                        PyTango takes the attribute as a name or as an object
                        carrying one -- an ``AttributeInfo``, a
                        ``DeviceAttribute``. Interpolating the object builds an
                        address the limits database has never heard of, which
                        on a deployment that allows unlisted channels is a
                        write that passes with no bound applied, so the
                        argument is reduced to its name first.
                        '''
                        _name = getattr(_attr, 'name', _attr)
                        return f"{{_proxy.dev_name()}}/{{_name}}"

                    def _tango_current_value(_proxy):
                        '''A reader for the max_step check, bound to the writing proxy.

                        The step is measured over the same DeviceProxy the
                        write goes through. The address the validator holds is
                        the full device/attribute form built above, so the
                        attribute name is its last segment.
                        '''
                        if not hasattr(_proxy, 'read_attribute'):
                            return None

                        def _read(_address):
                            _attr = _proxy.read_attribute(_address.rsplit('/', 1)[-1])
                            return getattr(_attr, 'value', _attr)

                        return _read

                    def _tango_pairs(_name_val):
                        '''Normalise write_attributes' argument to (name, value) pairs.

                        A shape that cannot be paired up fails CLOSED — the
                        write raises rather than reaching the device
                        unvalidated, which is the same trade the p4p batch
                        guard makes.
                        '''
                        _pairs = []
                        for _item in _name_val:
                            try:
                                _attr, _value = _item
                            except (TypeError, ValueError) as _shape_error:
                                raise ValueError(
                                    "tango write_attributes requires (attribute, value) "
                                    "pairs so each write can be limits-checked"
                                ) from _shape_error
                            _pairs.append((_attr, _value))
                        return _pairs

                    def _tango_checked_write(_original):
                        '''Wrap a DeviceProxy spelling that writes ONE attribute.

                        Every ``(attribute, value)`` spelling shares this
                        shape: the value is bounded against the full
                        device/attribute address, and whatever the original
                        answers -- nothing, a request id, a read-back
                        attribute -- is passed back untouched.
                        '''

                        def _checked(self, attr_name, value, *args, **kwargs):
                            _limits_validator.validate(
                                _tango_channel(self, attr_name),
                                value,
                                read_current=_tango_current_value(self),
                            )
                            return _original(self, attr_name, value, *args, **kwargs)

                        return _checked

                    def _tango_checked_write_many(_original):
                        '''Wrap a DeviceProxy spelling that writes MANY attributes.

                        Every pair is bounded before any of them is written,
                        and the answer is passed back untouched. PyTango names
                        the pairs ``name_val`` on some spellings and
                        ``attr_values`` on others, so they are taken
                        positionally or under either name and forwarded the way
                        they arrived; a guard that accepted one spelling would
                        break the other with a TypeError naming its own
                        internals.
                        '''

                        def _checked(self, *args, **kwargs):
                            _keyword = None
                            if args:
                                _name_val = args[0]
                            else:
                                for _spelling in ('name_val', 'attr_values'):
                                    if _spelling in kwargs:
                                        _keyword = _spelling
                                        break
                                if _keyword is None:
                                    raise ValueError(
                                        "tango write_attributes requires (attribute, "
                                        "value) pairs so each write can be "
                                        "limits-checked"
                                    )
                                _name_val = kwargs[_keyword]
                            _pairs = _tango_pairs(_name_val)
                            _read_current = _tango_current_value(self)
                            for _attr, _value in _pairs:
                                _limits_validator.validate(
                                    _tango_channel(self, _attr),
                                    _value,
                                    read_current=_read_current,
                                )
                            # The materialised pairs, not the argument: a
                            # generator was consumed by the check above, and
                            # forwarding it would write nothing at all.
                            if _keyword is None:
                                return _original(self, _pairs, *args[1:], **kwargs)
                            _forwarded = dict(kwargs)
                            _forwarded[_keyword] = _pairs
                            return _original(self, **_forwarded)

                        return _checked

                    # An ``AttributeProxy`` addresses one attribute for its
                    # whole life, so its writes carry a value alone. The
                    # channel address is rebuilt from the device proxy behind
                    # it and the attribute's own name.
                    def _tango_attribute_channel(_proxy):
                        return f"{{_proxy.get_device_proxy().dev_name()}}/{{_proxy.name()}}"

                    def _tango_attribute_current_value(_proxy):
                        '''A reader for the max_step check, bound to the writing proxy.

                        The proxy already points at the attribute being
                        written, so the address the validator passes names
                        that same channel and nothing is looked up from it.
                        '''
                        if not hasattr(_proxy, 'read'):
                            return None

                        def _read(_address):
                            _value = _proxy.read()
                            return getattr(_value, 'value', _value)

                        return _read

                    def _tango_checked_attribute_write(_original):
                        '''Wrap an AttributeProxy spelling that writes its attribute.'''

                        def _checked(self, value, *args, **kwargs):
                            _limits_validator.validate(
                                _tango_attribute_channel(self),
                                value,
                                read_current=_tango_attribute_current_value(self),
                            )
                            return _original(self, value, *args, **kwargs)

                        return _checked

                    def _tango_wrap_on(_class_name, _spellings):
                        '''Install a limits-checked wrapper per spelling the class carries.

                        Answers the names installed, so the success line can
                        say what this binding actually had; a tango without
                        the class answers an empty list.
                        '''
                        _cls = getattr(_tango, _class_name, None)
                        _installed = []
                        if _cls is not None:
                            for _name, _wrap in _spellings:
                                if hasattr(_cls, _name):
                                    setattr(_cls, _name, _wrap(getattr(_cls, _name)))
                                    _installed.append(_name)
                        return _installed

                    # A Tango COMMAND is not a channel write. It names an
                    # operation on the device, and its argument is whatever
                    # that command's own signature says - a mode string, a
                    # struct, nothing at all. There is no channel address to
                    # look the limits up under and no number to bound, so a
                    # limits-checked run cannot approve one; a check that
                    # let it through would mean nothing. Both spellings
                    # refuse outright, argument or not, which is the posture
                    # the p4p guard takes for rpc().
                    def _tango_refuse_command(self, *args, **kwargs):
                        '''Unconditional refusal for a Tango command call.'''
                        raise RuntimeError(
                            "Tango command refused in a limits-checked run: "
                            "a command carries no value to bound"
                        )

                    # A group write fans one value out to every device the
                    # group matched - there is no single channel address to
                    # look limits up under and no one device to bound the
                    # step against, so it refuses in range or not.
                    def _tango_refuse_group_write(self, *args, **kwargs):
                        '''Unconditional refusal for a Group attribute write.'''
                        raise RuntimeError(
                            "Tango group write refused in a limits-checked "
                            "run: a group write fans one value out to many "
                            "devices and is not limits-checked"
                        )

                    def _tango_refuse_on(_class_name, _spellings):
                        '''Install a refusal per spelling the named class carries.

                        Answers the names installed, so the success line can
                        say what this binding actually had; a tango without
                        the class answers an empty list.
                        '''
                        _cls = getattr(_tango, _class_name, None)
                        _installed = []
                        if _cls is not None:
                            for _name, _refusal in _spellings:
                                if hasattr(_cls, _name):
                                    setattr(_cls, _name, _refusal)
                                    _installed.append(_name)
                        return _installed

                    _tango_commands = (
                        ('command_inout', _tango_refuse_command),
                        ('command_inout_asynch', _tango_refuse_command),
                    )
                    # Every attribute-write spelling DeviceProxy carries.
                    # The asynchronous and write-read spellings drive the same
                    # device with the same value as the plain write, so each
                    # one is checked; a spelling left out is a live unchecked
                    # write under a guard reporting itself installed.
                    _tango_device_writes = (
                        ('write_attribute', _tango_checked_write),
                        ('write_attribute_asynch', _tango_checked_write),
                        ('write_read_attribute', _tango_checked_write),
                        ('write_attributes', _tango_checked_write_many),
                        ('write_attributes_asynch', _tango_checked_write_many),
                        ('write_read_attributes', _tango_checked_write_many),
                    )
                    _tango_attribute_writes = (
                        ('write', _tango_checked_attribute_write),
                        ('write_asynch', _tango_checked_attribute_write),
                        ('write_read', _tango_checked_attribute_write),
                    )
                    _tango_wrapped = []
                    _tango_attribute_wrapped = []
                    _tango_refused = []
                    _tango_refused_base = []

                    if hasattr(_tango, "DeviceProxy"):
                        _tango_wrapped = _tango_wrap_on('DeviceProxy', _tango_device_writes)

                        _tango_refused = _tango_refuse_on('DeviceProxy', _tango_commands)

                        # PyTango DEFINES both spellings on the base class
                        # ``Connection``; ``DeviceProxy`` only inherits them.
                        # Patching the subclass installs a shadow in
                        # ``DeviceProxy.__dict__`` and leaves the original
                        # reachable on the definer, so an unbound
                        # ``tango.Connection.command_inout(proxy, 'On')`` would
                        # still drive the device with the refusal installed.
                        # Refusing on the definer closes that spelling, and
                        # every other Connection subclass with it.
                        _tango_refused_base = _tango_refuse_on('Connection', _tango_commands)

                        if not _tango_wrapped and not _tango_refused:
                            print(
                                "⚠️  tango guard failed: DeviceProxy has no write "
                                "or command method"
                            )
                    else:
                        # The success line is the operator's only evidence the
                        # guard is on. A tango without a DeviceProxy wrapped
                        # nothing on it, and must not report it guarded.
                        print("⚠️  tango guard failed: no DeviceProxy class")

                    # ``tango.AttributeProxy`` is its own class, bound to one
                    # attribute rather than to a device, so neither install
                    # above reaches it. It is patched on its own, outside the
                    # DeviceProxy branch: an AttributeProxy is a write surface
                    # whether or not a DeviceProxy sits beside it. Its writes
                    # forward to the DeviceProxy methods wrapped above, so a
                    # value is bounded twice and a max_step channel is read
                    # twice; the outer check is kept deliberately, because it
                    # is the only one left if the DeviceProxy install fails.
                    _tango_attribute_wrapped = _tango_wrap_on(
                        'AttributeProxy', _tango_attribute_writes
                    )

                    # ``tango.Group`` is not a Connection subclass. It is a
                    # separate pure-Python class carrying its own commands AND
                    # its own attribute writes, so neither install above
                    # reaches it. It is patched on its own, outside the
                    # DeviceProxy branch: a Group is a write surface whether
                    # or not a DeviceProxy sits beside it.
                    _tango_refused_group = _tango_refuse_on(
                        'Group',
                        _tango_commands
                        + (
                            ('write_attribute', _tango_refuse_group_write),
                            ('write_attribute_asynch', _tango_refuse_group_write),
                        ),
                    )

                    # The success line is the operator's only evidence the
                    # guard is on, so it names what this binding actually
                    # carried, class by class, rather than what the block can
                    # install.
                    _tango_report = [
                        _tango_label + ", ".join(_n + "()" for _n in _tango_names)
                        for _tango_label, _tango_names in (
                            ("wrapped on DeviceProxy: ", _tango_wrapped),
                            ("wrapped on AttributeProxy: ", _tango_attribute_wrapped),
                            ("refused on DeviceProxy: ", _tango_refused),
                            ("refused on Connection: ", _tango_refused_base),
                            ("refused on Group: ", _tango_refused_group),
                        )
                        if _tango_names
                    ]
                    if _tango_report:
                        print("✅ Monkeypatched tango: " + "; ".join(_tango_report))
                except ImportError:
                    print("ℹ️  tango not available - Tango limits checking disabled")
                except Exception as _tango_error:
                    print(f"⚠️  tango guard failed: {{_tango_error}}")

                # --- DOOCS. ``doocs4py.set`` is the call the shipped DOOCS
                # connector writes through, so a readwrite script naming it
                # reaches the same hardware the mediated path does.
                try:
                    import doocs4py as _doocs4py

                    def _doocs_current_value(_address):
                        '''A reader for the max_step check, over doocs4py itself.

                        ``get()`` answers an EqData, which carries the number
                        in ``get_data()`` — the same unwrapping the shipped
                        DOOCS connector does.
                        '''
                        _current = _doocs4py.get(_address)
                        _get_data = getattr(_current, 'get_data', None)
                        return _get_data() if _get_data is not None else _current

                    _doocs_reader = (
                        _doocs_current_value if hasattr(_doocs4py, "get") else None
                    )

                    if hasattr(_doocs4py, "set"):
                        _orig_doocs_set = _doocs4py.set

                        def _checked_doocs_set(address, value, *args, **kwargs):
                            '''Limits-checked wrapper for doocs4py.set().'''
                            _limits_validator.validate(
                                address, value, read_current=_doocs_reader
                            )
                            return _orig_doocs_set(address, value, *args, **kwargs)

                        _doocs4py.set = _checked_doocs_set

                    print("✅ Monkeypatched doocs4py.set()")
                except ImportError:
                    print("ℹ️  doocs4py not available - DOOCS limits checking disabled")
                except Exception as _doocs_error:
                    print(f"⚠️  doocs4py guard failed: {{_doocs_error}}")

                # --- caproto. Write entry points in three modules, one per
                # concurrency flavor: the sync client's module-level functions,
                # the threading client's PV and Batch classes, and the asyncio
                # client's own PV. Each module is imported and patched in its
                # OWN try/except - a flavor missing from this environment must
                # not take the guards for the others with it.
                def _caproto_scalar(_response):
                    '''The number in a caproto read response.

                    caproto answers a read with a response object whose
                    ``data`` is an array, even for a scalar channel; a stub or
                    a bare value answers itself.
                    '''
                    _data = getattr(_response, 'data', _response)
                    try:
                        return _data[0]
                    except (TypeError, IndexError, KeyError):
                        return _data

                try:
                    import caproto.sync.client as _caproto_sync

                    def _caproto_sync_current_value(_address):
                        '''A reader for the max_step check, over caproto's own client.

                        The step-read ceiling is passed explicitly rather than
                        left to caproto's own default, so a channel that never
                        answers fails the check closed quickly instead of
                        holding the write open.
                        '''
                        return _caproto_scalar(
                            _caproto_sync.read(_address, timeout=STEP_READ_TIMEOUT_SECONDS)
                        )

                    _caproto_sync_reader = (
                        _caproto_sync_current_value
                        if hasattr(_caproto_sync, "read")
                        else None
                    )

                    # Both module-level spellings lead with the channel name
                    # and the value, and ``read_write_read`` differs only in
                    # reading the channel back afterwards - it drives the
                    # channel with the same value, so it is bounded the same
                    # way. The pair is taken positionally or under the names
                    # caproto gives it, and the call is forwarded exactly as
                    # it arrived.
                    def _caproto_sync_checked(_original):
                        '''Wrap a sync-client spelling that writes a channel.'''

                        def _checked(*args, **kwargs):
                            _pv_name = args[0] if args else kwargs.get('pv_name')
                            _data = args[1] if len(args) > 1 else kwargs.get('data')
                            if _pv_name is None:
                                raise ValueError(
                                    "caproto write requires a channel name so "
                                    "the write can be limits-checked"
                                )
                            _limits_validator.validate(
                                _pv_name, _data, read_current=_caproto_sync_reader
                            )
                            return _original(*args, **kwargs)

                        return _checked

                    _caproto_sync_wrapped = []
                    for _caproto_name in ('write', 'read_write_read'):
                        if hasattr(_caproto_sync, _caproto_name):
                            setattr(
                                _caproto_sync,
                                _caproto_name,
                                _caproto_sync_checked(
                                    getattr(_caproto_sync, _caproto_name)
                                ),
                            )
                            _caproto_sync_wrapped.append(_caproto_name)

                    # The success line is the operator's only evidence the
                    # guard is on, so it names what this binding actually
                    # carried rather than what the block can install.
                    if _caproto_sync_wrapped:
                        print(
                            "✅ Monkeypatched caproto.sync.client: "
                            + ", ".join(_n + "()" for _n in _caproto_sync_wrapped)
                        )
                    else:
                        print(
                            "⚠️  caproto.sync.client guard failed: "
                            "no write function"
                        )
                except ImportError:
                    print(
                        "ℹ️  caproto.sync.client not available - "
                        "caproto limits checking disabled"
                    )
                except Exception as _caproto_error:
                    print(f"⚠️  caproto.sync.client guard failed: {{_caproto_error}}")

                try:
                    import caproto.threading.client as _caproto_threading

                    def _caproto_pv_current_value(_pv):
                        '''A reader for the max_step check, bound to the writing PV.

                        The step-read ceiling is passed explicitly rather than
                        left to the client context's own timeout, which a
                        script is free to set to None.
                        '''
                        if not hasattr(_pv, 'read'):
                            return None

                        def _read(_address):
                            return _caproto_scalar(
                                _pv.read(timeout=STEP_READ_TIMEOUT_SECONDS)
                            )

                        return _read

                    _caproto_threading_wrapped = []
                    _CaprotoPV = getattr(_caproto_threading, 'PV', None)
                    _CaprotoBatch = getattr(_caproto_threading, 'Batch', None)

                    if _CaprotoPV is not None and hasattr(_CaprotoPV, "write"):
                        _orig_caproto_pv_write = _CaprotoPV.write

                        def _checked_caproto_pv_write(self, data, *args, **kwargs):
                            '''Limits-checked wrapper for caproto threading PV.write().'''
                            _limits_validator.validate(
                                self.name, data, read_current=_caproto_pv_current_value(self)
                            )
                            return _orig_caproto_pv_write(self, data, *args, **kwargs)

                        _CaprotoPV.write = _checked_caproto_pv_write
                        _caproto_threading_wrapped.append('PV.write')

                    # A ``Batch`` groups requests it is handed and sends them
                    # together, so its write carries the PV to drive rather
                    # than being bound to one. The channel is that PV's own
                    # name and the step is measured over that PV's own read -
                    # the same two answers the PV.write wrapper uses, taken
                    # from the argument instead of from ``self``.
                    if _CaprotoBatch is not None and hasattr(_CaprotoBatch, "write"):
                        _orig_caproto_batch_write = _CaprotoBatch.write

                        def _checked_caproto_batch_write(self, *args, **kwargs):
                            '''Limits-checked wrapper for caproto threading Batch.write().'''
                            _pv = args[0] if args else kwargs.get('pv')
                            _data = args[1] if len(args) > 1 else kwargs.get('data')
                            if _pv is None:
                                raise ValueError(
                                    "caproto Batch.write requires a pv so the "
                                    "write can be limits-checked"
                                )
                            _limits_validator.validate(
                                getattr(_pv, 'name', _pv),
                                _data,
                                read_current=_caproto_pv_current_value(_pv),
                            )
                            return _orig_caproto_batch_write(self, *args, **kwargs)

                        _CaprotoBatch.write = _checked_caproto_batch_write
                        _caproto_threading_wrapped.append('Batch.write')

                    if _caproto_threading_wrapped:
                        print(
                            "✅ Monkeypatched caproto.threading.client: "
                            + ", ".join(_n + "()" for _n in _caproto_threading_wrapped)
                        )
                    else:
                        print(
                            "⚠️  caproto.threading.client guard failed: "
                            "no write method"
                        )
                except ImportError:
                    print(
                        "ℹ️  caproto.threading.client not available - "
                        "caproto limits checking disabled"
                    )
                except Exception as _caproto_error:
                    print(f"⚠️  caproto.threading.client guard failed: {{_caproto_error}}")

                try:
                    import caproto.asyncio.client as _caproto_asyncio

                    _CaprotoAsyncPV = getattr(_caproto_asyncio, 'PV', None)

                    if _CaprotoAsyncPV is not None and hasattr(_CaprotoAsyncPV, "write"):
                        _orig_caproto_async_write = _CaprotoAsyncPV.write

                        async def _checked_caproto_async_write(self, data, *args, **kwargs):
                            '''Limits-checked wrapper for caproto asyncio PV.write().

                            The guard is a coroutine, so unlike the synchronous
                            clients it can await the max_step read itself and
                            hand the validator a plain value. That read is
                            bought only by a channel that configures max_step,
                            and a read that fails answers None, which fails the
                            step check closed. The ceiling is passed explicitly
                            rather than left to the client context's own
                            timeout, which a script is free to set to None.
                            '''
                            _cfg = _limits_validator.get_limits_config(self.name)
                            _current = None
                            if _cfg and _cfg['max_step'] is not None:
                                try:
                                    _current = _caproto_scalar(
                                        await self.read(timeout=STEP_READ_TIMEOUT_SECONDS)
                                    )
                                except Exception:
                                    _current = None

                            _limits_validator.validate(
                                self.name, data, read_current=lambda _address: _current
                            )
                            return await _orig_caproto_async_write(
                                self, data, *args, **kwargs
                            )

                        _CaprotoAsyncPV.write = _checked_caproto_async_write
                        print("✅ Monkeypatched caproto.asyncio.client PV.write()")
                    else:
                        print(
                            "⚠️  caproto.asyncio.client guard failed: "
                            "no PV.write method"
                        )
                except ImportError:
                    print(
                        "ℹ️  caproto.asyncio.client not available - "
                        "caproto limits checking disabled"
                    )
                except Exception as _caproto_error:
                    print(f"⚠️  caproto.asyncio.client guard failed: {{_caproto_error}}")
            except Exception as e:
                print(f"⚠️  Limits checking setup failed: {{e}}")
                import traceback
                traceback.print_exc()
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

        The table covers three kinds of route: the control-system client
        libraries themselves, the process-spawning surface that could shell out
        to ``caput``, and ``ctypes``, which reaches Channel Access without
        importing any client package at all. Each patched attribute is also
        followed back to the module that defined it, so a write a package
        merely re-exports refuses under both of its spellings. That step is
        what covers the defining modules no table can enumerate — PyTango
        defines its writes in ``tango.device_proxy`` and ``tango.connection``
        and binds them onto ``DeviceProxy``, and every binding has its own
        such layout; the one row that does name a private module
        (``aioca._catools``) is belt-and-braces for the single library whose
        second spelling is known. It is emitted into the script
        rather than applied here because the objects to patch only exist in the
        subprocess.

        The connector side of the same contract lives in
        ``osprey_connectors.control_system.base`` (refuses ``write_channel``
        when ``OSPREY_EXECUTION_MODE`` says readonly) and in the EPICS
        connector's gateway selection (stays on the read_only gateway).
        """
        if self.execution_mode != "readonly":
            return ""

        guard = f"""
            # Readonly run: refuse every control-system write entry point, and
            # every route out of Python that could reach one. Installed before
            # user code, so an alias bound later resolves here.
            import importlib as _osprey_importlib
            import sys as _osprey_sys

            # CPython resolves ``platform.uname().processor`` lazily, by
            # shelling out to ``uname -p`` on first read — and h5py reads it
            # while ``import at`` initialises its type layer, so the subprocess
            # refusal below would kill the import of a pure-simulation library.
            # Resolve it once now, while spawning is still allowed; the cached
            # value answers every later lookup without touching subprocess.
            import platform as _osprey_platform

            _osprey_platform.processor()
            del _osprey_platform

            _osprey_targets = {_READONLY_WRITE_TARGETS!r}


            def _osprey_readonly_refuse(*_args, **_kwargs):
                raise RuntimeError("@@REFUSAL@@")


            def _osprey_resolve(dotted):
                \"\"\"Import the longest importable prefix of *dotted*, then walk attributes.

                One spelling for modules, module attributes and classes alike.
                Returns None when the target is not present, which is the
                ordinary case for most of the table — an uninstalled library,
                or an optional flavour of an installed one. That case has to
                stay SILENT: it is true on every ordinary deployment, and a
                warning per absent target would print on every readonly run.
                \"\"\"
                parts = dotted.split(".")
                for _cut in range(len(parts), 0, -1):
                    try:
                        obj = _osprey_importlib.import_module(".".join(parts[:_cut]))
                    except ImportError:
                        continue
                    for _attr in parts[_cut:]:
                        try:
                            obj = getattr(obj, _attr)
                        except AttributeError:
                            # An importable parent without the child: the
                            # target does not exist here either.
                            return None
                    return obj
                return None


            for _osprey_dotted, _osprey_attrs in _osprey_targets:
                try:
                    _osprey_obj = _osprey_resolve(_osprey_dotted)
                    if _osprey_obj is None:
                        continue
                    for _osprey_attr in _osprey_attrs:
                        if not hasattr(_osprey_obj, _osprey_attr):
                            continue
                        _osprey_original = getattr(_osprey_obj, _osprey_attr)
                        setattr(_osprey_obj, _osprey_attr, _osprey_readonly_refuse)
                        # A re-export leaves a second spelling behind that no
                        # table can name for every binding: PyTango defines
                        # its writes in ``tango.device_proxy`` and
                        # ``tango.connection`` and binds them onto
                        # ``DeviceProxy``, so patching the class attribute
                        # alone leaves the module-level function writing.
                        # Follow the original back to the module that defined
                        # it and refuse there too. The identity check is what
                        # makes this safe to run generically: ``__module__``
                        # and ``__name__`` are metadata a decorator or a rebind
                        # can leave pointing at a module holding something else
                        # entirely, and replacing an attribute on a name match
                        # alone could silently refuse an unrelated read.
                        try:
                            _osprey_home = _osprey_sys.modules.get(
                                getattr(_osprey_original, "__module__", None)
                            )
                            _osprey_name = getattr(_osprey_original, "__name__", None)
                            if (
                                _osprey_home is not None
                                and isinstance(_osprey_name, str)
                                and getattr(_osprey_home, _osprey_name, None)
                                is _osprey_original
                            ):
                                setattr(
                                    _osprey_home, _osprey_name, _osprey_readonly_refuse
                                )
                        except Exception as _osprey_home_error:
                            # Secondary, best-effort step: the attribute the
                            # table names already refuses. A failure here — an
                            # unhashable ``__module__``, a module ``__getattr__``
                            # that raises — must name the attribute and let the
                            # REST of the row be patched, so it is caught here
                            # rather than at the row level.
                            print(
                                "⚠️  readonly guard "
                                f"({{_osprey_dotted}}.{{_osprey_attr}}) "
                                f"defining-module step failed: {{_osprey_home_error}}"
                            )
                    # pvaPy spells one typed setter per scalar and array type
                    # (putDouble, putScalarArray, ...). Enumerating them would
                    # go stale against the binding; the prefix will not. Three
                    # writes sit outside that prefix — asyncPut, parsePut and
                    # parsePutGet — and they reach the machine exactly as the
                    # rest do, so they are swept with them.
                    if _osprey_dotted == "pvaccess.Channel":
                        for _osprey_attr in dir(_osprey_obj):
                            if _osprey_attr.startswith(
                                ("put", "asyncPut", "parsePut")
                            ):
                                setattr(_osprey_obj, _osprey_attr, _osprey_readonly_refuse)
                except Exception as _osprey_guard_error:
                    # A target that cannot be patched must not stop the ones
                    # after it, and the operator needs to know which one.
                    print(
                        f"⚠️  readonly guard ({{_osprey_dotted}}) failed: {{_osprey_guard_error}}"
                    )

            del _osprey_importlib, _osprey_sys, _osprey_targets, _osprey_resolve
        """
        return textwrap.dedent(guard).strip().replace("@@REFUSAL@@", READONLY_REFUSAL)

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
        """Get cleanup and results export code."""

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

        # Combine all parts properly (4-space indent to sit inside the finally block)
        indented_host_section = "\n".join(
            "    " + line if line.strip() else line for line in host_output_section.split("\n")
        )
        indented_guard_restore = "\n".join(
            "    " + line if line.strip() else line for line in guard_restore_section.split("\n")
        )
        indented_error_handling = "\n".join(
            "    " + line if line.strip() else line for line in metadata_error_handling.split("\n")
        )

        return "\n".join(
            [
                base_cleanup,
                indented_host_section,
                indented_guard_restore,
                file_persistence_section,
                indented_error_handling,
            ]
        )
