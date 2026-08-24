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


#: Every entry point a readonly run refuses, as ``(dotted target, attributes)``.
#: This table is the canonical machine-readable answer to "what counts as a
#: control-system write from Python" — the docs list is written from it, and a
#: library added here needs no other change to be enforced.
#:
#: The dotted target is resolved by importing its longest importable prefix and
#: then walking attributes, so a module (``epics``), a module attribute
#: (``epics.ca``) and a class (``p4p.client.thread.Context``) are all spelled
#: the same way. Attributes that do not exist on the resolved object are
#: skipped, which is what makes listing several client flavours free: an
#: uninstalled or older library simply contributes nothing.
#:
#: Patching the object in ``sys.modules`` — rather than inspecting the source —
#: is what makes this immune to spelling. ``importlib.import_module("epics")``,
#: ``from epics import caput as _w`` and ``getattr(epics, "ca" + "put")`` all
#: end up holding the refusing function, because they all resolve through the
#: one module object this mutates.
_READONLY_WRITE_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # --- EPICS Channel Access (pyepics) ---
    ("epics", ("caput", "caput_many")),
    ("epics.PV", ("put",)),
    ("epics.ca", ("put", "put_complete")),
    # --- PVAccess (p4p): one client Context per concurrency flavour, plus the
    # server-side SharedPV, which puts values on the wire when it is opened or
    # posted to.
    ("p4p.client.thread.Context", ("put", "rpc")),
    ("p4p.client.asyncio.Context", ("put", "rpc")),
    ("p4p.client.cothread.Context", ("put", "rpc")),
    ("p4p.server.raw.SharedPV", ("post", "open")),
    ("p4p.server.thread.SharedPV", ("post", "open")),
    ("p4p.server.asyncio.SharedPV", ("post", "open")),
    # --- Channel Access (caproto) ---
    ("caproto.sync.client", ("write", "write_read")),
    ("caproto.threading.client.PV", ("write", "write_all")),
    ("caproto.threading.client.Batch", ("write",)),
    ("caproto.asyncio.client.PV", ("write",)),
    # --- PVAccess (pvaPy). Its ``Channel`` carries one typed setter per scalar
    # and array type, so the ``put`` prefix is swept dynamically below rather
    # than enumerated here; ``put`` itself is listed so the table still names
    # the library.
    ("pvaccess.Channel", ("put", "putGet")),
    # --- Tango. ``command_inout`` is included because a Tango command is an
    # action on the device, not a read — refusing it is the readonly reading.
    # ``PyTango`` is the legacy alias for the same package; when both import,
    # they resolve to the same class object and the second patch is a no-op.
    (
        "tango.DeviceProxy",
        (
            "write_attribute",
            "write_attributes",
            "write_attribute_asynch",
            "write_attributes_asynch",
            "write_read_attribute",
            "write_read_attributes",
            "write_pipe",
            "put_property",
            "command_inout",
            "command_inout_asynch",
        ),
    ),
    ("tango.AttributeProxy", ("write", "write_asynch", "write_read")),
    (
        "PyTango.DeviceProxy",
        (
            "write_attribute",
            "write_attributes",
            "write_read_attribute",
            "command_inout",
        ),
    ),
    # --- Routes out of Python. A readonly run has no legitimate use for these:
    # ``import subprocess`` is already refused in every mode by the static
    # import check, so anything reaching the process-spawning surface at
    # runtime got there by an evasion. ``os.fork`` is deliberately absent —
    # forking alone cannot run a new program, and refusing it would break
    # ordinary multiprocessing for no security gain, since the exec half of
    # every fork+exec is refused here.
    (
        "subprocess",
        (
            "run",
            "Popen",
            "call",
            "check_call",
            "check_output",
            "getoutput",
            "getstatusoutput",
        ),
    ),
    ("_posixsubprocess", ("fork_exec",)),
    # ``os`` re-exports these from ``posix``; patching only ``os`` would leave
    # ``import posix; posix.system(...)`` open, so both modules are swept.
    (
        "os",
        (
            "system",
            "popen",
            "execl",
            "execle",
            "execlp",
            "execlpe",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
            "posix_spawn",
            "posix_spawnp",
        ),
    ),
    (
        "posix",
        (
            "system",
            "popen",
            "execv",
            "execve",
            "posix_spawn",
            "posix_spawnp",
        ),
    ),
    # --- Loading a shared library sidesteps every Python-level guard above:
    # ``ctypes.CDLL("libca")`` reaches Channel Access without importing a
    # single client package. ``LibraryLoader.__getattr__`` is patched too,
    # because ``ctypes.cdll.libca`` never goes through ``CDLL`` by that name.
    ("ctypes", ("CDLL", "PyDLL", "WinDLL", "OleDLL")),
    ("ctypes.LibraryLoader", ("LoadLibrary", "__getattr__")),
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
        """
        self.limits_validator = limits_validator
        self.execution_mode = execution_mode
        self.protected_roots = tuple(str(root) for root in protected_roots)
        self.permitted_roots = tuple(str(root) for root in permitted_roots)

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
                    LimitsValidator, ChannelLimitsConfig
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

                try:
                    import epics

                    # Store original functions
                    _original_caput = epics.caput
                    _original_PV_put = epics.PV.put if hasattr(epics.PV, 'put') else None

                    def _checked_caput(pvname, value, wait=False, timeout=60, **kwargs):
                        '''Limits-checked wrapper for epics.caput()'''
                        _limits_validator.validate(pvname, value)  # Raises if invalid
                        return _original_caput(pvname, value, wait=wait, timeout=timeout, **kwargs)

                    if _original_PV_put is not None:
                        def _checked_PV_put(self, value, wait=False, timeout=60, **kwargs):
                            '''Limits-checked wrapper for PV.put()'''
                            _limits_validator.validate(self.pvname, value)  # Raises if invalid
                            return _original_PV_put(self, value, wait=wait, timeout=timeout, **kwargs)

                        epics.PV.put = _checked_PV_put

                    epics.caput = _checked_caput
                    print("✅ Monkeypatched epics.caput() and PV.put()")

                except ImportError:
                    print("ℹ️  pyepics not available - EPICS limits checking disabled")

                # p4p / PVAccess clients: p4p ships parallel Context classes per
                # concurrency flavor, so each one is imported and patched in its
                # OWN try/except - patching only the thread client would leave an
                # approved `from p4p.client.asyncio import Context` put unvalidated.
                def _p4p_validate_put(_name, _values):
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

                    Returns the name to forward to the original put(); a
                    one-shot iterable is materialized so validation does not
                    consume the caller's names.
                    '''
                    if isinstance(_name, str):
                        _limits_validator.validate(_name, _values)
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
                        _limits_validator.validate(_pair_name, _pair_value)
                    return _names_seq

                def _p4p_install_guard(_context_cls):
                    '''Limits-check put() and refuse rpc() on one p4p Context class.'''
                    if hasattr(_context_cls, 'put'):
                        _original_p4p_put = _context_cls.put

                        def _p4p_checked_put(self, name, values, *args, **kwargs):
                            '''Limits-checked wrapper for p4p Context.put()'''
                            name = _p4p_validate_put(name, values)  # Raises if invalid
                            return _original_p4p_put(self, name, values, *args, **kwargs)

                        _context_cls.put = _p4p_checked_put

                    if hasattr(_context_cls, 'rpc'):
                        def _p4p_blocked_rpc(self, *args, **kwargs):
                            '''Unconditional refusal for p4p Context.rpc().

                            Limits semantics cannot apply to an arbitrary rpc
                            payload, so refusal is the only honest parity.
                            '''
                            raise RuntimeError(
                                "rpc is not mediated and cannot be approved — "
                                "use the supervised write path"
                            )

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
        importing any client package at all. It is emitted into the script
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
                        if hasattr(_osprey_obj, _osprey_attr):
                            setattr(_osprey_obj, _osprey_attr, _osprey_readonly_refuse)
                    # pvaPy spells one typed setter per scalar and array type
                    # (putDouble, putScalarArray, ...). Enumerating them would
                    # go stale against the binding; the prefix will not.
                    if _osprey_dotted == "pvaccess.Channel":
                        for _osprey_attr in dir(_osprey_obj):
                            if _osprey_attr.startswith("put"):
                                setattr(_osprey_obj, _osprey_attr, _osprey_readonly_refuse)
                except Exception as _osprey_guard_error:
                    # A target that cannot be patched must not stop the ones
                    # after it, and the operator needs to know which one.
                    print(
                        f"⚠️  readonly guard ({{_osprey_dotted}}) failed: {{_osprey_guard_error}}"
                    )

            del _osprey_importlib, _osprey_targets, _osprey_resolve
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

    def _get_metadata_init(self) -> str:
        """Initialize execution metadata tracking."""
        return textwrap.dedent(
            """
            # Execution metadata
            execution_metadata = {
                "start_time": _datetime.now().isoformat(),
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
        execution_metadata["end_time"] = _datetime.now().isoformat()

    except Exception as user_code_error:
        # Capture user code errors
        execution_metadata["success"] = False
        execution_metadata["error_type"] = type(user_code_error).__name__
        execution_metadata["error_message"] = str(user_code_error)
        execution_metadata["end_time"] = _datetime.now().isoformat()
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
                execution_metadata["end_time"] = _datetime.now().isoformat()

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
