"""The environment every agent-code execution sandbox spawns its child with.

Three call sites spawn a local subprocess to run agent-generated Python:
``python_executor.executor`` (the general-purpose Python execution sandbox),
``workspace.execution.sandbox_executor`` (the lighter visualization-only
sandbox), and ``services.bluesky_bridge.plan_validation`` (plan import
checking). All three build the child's environment with
:func:`scrub_sandbox_child_env`, which starts from an ALLOWLIST: the child gets
the names its own code reads (process basics, locale, interpreter and linker,
TLS trust, proxies, plotting caches, the control-system client families and the
``OSPREY_*`` names osprey reads in the child) plus the names a deployment lists
in ``python_executor.child_env_passthrough``, and nothing else from the host. A
secret added to a host later stays out of agent code without anyone listing it.

Two denylists still run on top of the allowlist, so neither the allowlist nor
the config key can re-admit what they drop. The CREDENTIAL set is not defined
here: it lives in :mod:`osprey.utils.sensitive_env`, the dependency-free leaf
module that ``agent_runner`` also uses, so that the PTY child and the execution
sandboxes cannot drift apart. What IS defined here is the sandbox-only drop:
the web-terminal address book and the navigation-only perimeter stamp. Those
are not credential policy - the PTY child that shares
``osprey.utils.sensitive_env`` *is* the web terminal and must keep them - but
no execution sandbox has any use for them.

The child is also stamped with
:data:`~osprey_connectors.dotenv.ENV_CHAIN_APPLIED_ENV`, so its own config
loading does not read the project's env chain back into it.

The notebook sidecar is the one caller of the two denylists alone
(:func:`drop_sandbox_denied_env`): its kernels run code the operator types, the
PTY child's policy rather than an execution sandbox's.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

from osprey.utils.sensitive_env import (
    SENSITIVE_ENV_EXACT,
    SENSITIVE_ENV_SUFFIXES,
    is_sensitive,
    strip_sensitive,
)
from osprey_connectors.dotenv import ENV_CHAIN_APPLIED_ENV

__all__ = [
    "CHILD_ENV_PASSTHROUGH_KEY",
    "ENV_CHAIN_APPLIED_ENV",
    "PERIMETER_DENY_PORTS_ENV",
    "PERIMETER_MARKER_ENV",
    "SANDBOX_CHILD_ENV_ALLOW_NAMES",
    "SANDBOX_CHILD_ENV_ALLOW_PREFIXES",
    "SANDBOX_CHILD_ENV_DROP_NAMES",
    "SANDBOX_CHILD_ENV_DROP_PREFIXES",
    "SENSITIVE_ENV_EXACT",
    "SENSITIVE_ENV_SUFFIXES",
    "WEB_TERMINAL_ENV_NAMES_TO_DROP",
    "configured_child_env_passthrough",
    "drop_sandbox_denied_env",
    "scrub_sandbox_child_env",
    "scrub_sensitive_env",
]

# The web-terminal address family, dropped from every sandbox child on top of
# the shared credential scrub. A sandbox's only callback surface is the
# `save_artifact` helper its execution wrapper injects, which writes to the
# filesystem: nothing in a child resolves a terminal URL or calls a web-terminal
# route, so these names buy it nothing and only tell agent code where a surface
# it must not reach is listening. OSPREY_TERMINAL_SECRET is already gone via
# scrub_sensitive_env; dropping the whole family is still right, because the
# rest of it (bind host, landing URL, external origin, the per-user
# OSPREY_TERMINAL_SECRET_<USER> names) is the same address book.
#
# Deliberately NOT added to the shared deny-list in osprey.utils.sensitive_env:
# that set is shared with the PTY child, and the PTY child *is* the web terminal
# - it must keep these (see the module docstring). This is a per-sandbox
# narrowing, not a credential policy.
WEB_TERMINAL_ENV_NAMES_TO_DROP: tuple[str, ...] = ("OSPREY_WEB_PORT",)

#: The navigation-only perimeter stamp, rendered onto every per-user web
#: terminal container by the deployment's compose overlay when
#: ``modules.web_terminals.auth.method`` is ``none``. The marker names the
#: posture; the deny-list names the deployment's own web ports (nginx, the TLS
#: listener when TLS is on, and every roster user's terminal), which under that
#: posture are reachable from inside such a container as whoever owns them -
#: nginx injects each user's operator secret, and these containers share the
#: host network namespace.
#:
#: Read in the PARENT process and handed to the sandbox as a wrapper argument.
#: They are NOT in the ``OSPREY_TERMINAL_`` family on purpose: that prefix is
#: dropped from the child too, and a stamp the parent could not read back would
#: be inert. The child never sees them either (they are in
#: :data:`SANDBOX_CHILD_ENV_DROP_NAMES`) - executed code is told what it may not
#: reach by the process that spawned it, and a sandbox that re-derived the list
#: could equally derive an empty one.
PERIMETER_MARKER_ENV = "OSPREY_WEB_PERIMETER"
PERIMETER_DENY_PORTS_ENV = "OSPREY_WEB_PERIMETER_DENY_PORTS"

#: Every exact name dropped from a sandbox child's environment: the web-terminal
#: address book plus the perimeter stamp the parent has already consumed by the
#: time the child is spawned.
SANDBOX_CHILD_ENV_DROP_NAMES: tuple[str, ...] = (
    *WEB_TERMINAL_ENV_NAMES_TO_DROP,
    PERIMETER_MARKER_ENV,
    PERIMETER_DENY_PORTS_ENV,
)

#: Matched by prefix so a terminal variable added later is covered without a
#: code change here - the same reasoning as SENSITIVE_ENV_SUFFIXES.
SANDBOX_CHILD_ENV_DROP_PREFIXES: tuple[str, ...] = ("OSPREY_TERMINAL_",)

# ``OSPREY_AUDIT_IDENTITY`` is the one ``OSPREY_``-family name that must never
# join either drop list above, however much it reads like a neighbour of the
# terminal prefix, and must stay on the allowlist below. The child resolves its
# own record directory through it (see ``osprey_connectors.identity``), and
# inside a container it is the only rung that survives this severing - the
# terminal prefix is dropped by design and the process account names ``osprey``
# or ``root``. Dropped, the child does not fail: it files its records under a
# name no reader looks for. A plain comment, not a ``#:`` block: it documents a
# rule about the lists around it rather than the next definition below, which is
# what a ``#:`` block would attach it to.

#: Every exact name an execution child is handed from the parent environment.
#: Each is one the child's own code (the interpreter, osprey's runtime, audit
#: and connector packages, the libraries they load) reads. The osprey names are
#: literals rather than imports so this module stays importable without
#: ``osprey.runtime`` or ``osprey.audit``; a test pins each to its source.
SANDBOX_CHILD_ENV_ALLOW_NAMES: tuple[str, ...] = (
    # Process basics and locale.
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "TZ",
    "LANG",
    "LANGUAGE",
    # Interpreter, virtual environment and dynamic linker.
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    # TLS trust.
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    # Proxies, in both spellings.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    # Plotting and library caches.
    "MPLBACKEND",
    "MPLCONFIGDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    # DOOCS name server.
    "ENSHOST",
    # Osprey names read in the child.
    "CONFIG_FILE",
    "OSPREY_CONFIG",
    "OSPREY_QUIET",
    "OSPREY_SESSION_ID",
    "OSPREY_DISPATCH_RUN_ID",
    "OSPREY_AUDIT_IDENTITY",
    "OSPREY_AGENT_DATA_ROOT",
    "OSPREY_CONTROL_CONTEXT_DIR",
    "OSPREY_CONTROL_CONTEXT_TREE",
    "OSPREY_CONTROL_OWNER",
    "OSPREY_LAUNCH_POSTURE",
    "OSPREY_EXECUTION_MODE",
    "OSPREY_POSTURE_SOURCE",
    "OSPREY_POSTURE_SESSION",
    "OSPREY_CONTROL_TARGET",
    "OSPREY_CONTROL_TARGET_GENERATION",
    "OSPREY_CONTROL_TARGET_REFUSAL",
)

#: Name families an execution child is handed whole: locale categories, the
#: interpreter's own settings, and the control-system client libraries. ``OSPREY_``
#: is deliberately not a prefix here: the family includes the web-auth passwords
#: and session secrets and the archiver password, so osprey names are listed one
#: by one above.
SANDBOX_CHILD_ENV_ALLOW_PREFIXES: tuple[str, ...] = (
    "LC_",
    "PYTHON",
    "EPICS_",
    "PYEPICS_",
    "TANGO_",
)

#: The config key that lists extra names a deployment passes to every
#: agent-code execution child.
CHILD_ENV_PASSTHROUGH_KEY = "python_executor.child_env_passthrough"

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def scrub_sensitive_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of *env* with agent-forbidden credentials removed.

    Drops any key in :data:`SENSITIVE_ENV_EXACT` and any key ending in one of
    :data:`SENSITIVE_ENV_SUFFIXES`, delegating the match to
    :func:`osprey.utils.sensitive_env.strip_sensitive`. Used to build the
    environment passed to an agent-code execution subprocess, so the
    sandboxed code cannot read these secrets even though the parent process
    needs them for its own MCP/server plumbing. *env* is not mutated, so
    ``os.environ`` may be passed directly.
    """
    return strip_sensitive(env)


def _is_denied(name: str) -> bool:
    """Whether a name is dropped by the credential set or the sandbox-only drop."""
    return (
        is_sensitive(name)
        or name in SANDBOX_CHILD_ENV_DROP_NAMES
        or name.startswith(SANDBOX_CHILD_ENV_DROP_PREFIXES)
    )


def configured_child_env_passthrough(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the deployment's extra child-environment names from *config*.

    Args:
        config: The loaded osprey config mapping.

    Returns:
        The names listed under ``python_executor.child_env_passthrough``, or an
        empty tuple when the key is absent or null.

    Raises:
        ValueError: The ``python_executor`` section is not a mapping, the key is
            not a list of variable names, or it lists a name the
            credential set or the sandbox-only drop removes. An operator who
            lists a name expects it to arrive, so a name that cannot is an error.
    """
    section = config.get("python_executor") or {}
    if not isinstance(section, Mapping):
        raise ValueError(
            f"{CHILD_ENV_PASSTHROUGH_KEY}: python_executor must be a mapping, "
            f"got {type(section).__name__}"
        )
    raw = section.get("child_env_passthrough")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"{CHILD_ENV_PASSTHROUGH_KEY} must be a list of environment variable names, "
            f"got {type(raw).__name__}"
        )
    names: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not _ENV_NAME_RE.fullmatch(entry):
            raise ValueError(
                f"{CHILD_ENV_PASSTHROUGH_KEY}: {entry!r} is not an environment variable name"
            )
        if _is_denied(entry):
            raise ValueError(
                f"{CHILD_ENV_PASSTHROUGH_KEY}: {entry} is a credential or terminal name "
                "and cannot be passed to executed code"
            )
        names.append(entry)
    return tuple(names)


def drop_sandbox_denied_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of *env* without the credential set and the sandbox-only drop.

    :func:`scrub_sensitive_env` first (the credential policy shared with the PTY
    child), then every name in :data:`SANDBOX_CHILD_ENV_DROP_NAMES` and every
    name starting with one of :data:`SANDBOX_CHILD_ENV_DROP_PREFIXES`.

    :func:`scrub_sandbox_child_env` runs this over its allowlisted result, and
    the notebook sidecar uses it alone. *env* is not mutated, so ``os.environ``
    may be passed directly.
    """
    scrubbed = scrub_sensitive_env(env)
    for name in tuple(scrubbed):
        if name in SANDBOX_CHILD_ENV_DROP_NAMES or name.startswith(SANDBOX_CHILD_ENV_DROP_PREFIXES):
            scrubbed.pop(name, None)
    return scrubbed


def scrub_sandbox_child_env(
    env: Mapping[str, str], *, passthrough: Iterable[str] = ()
) -> dict[str, str]:
    """Return the environment an agent-code execution child may be spawned with.

    Keeps a name when it is in :data:`SANDBOX_CHILD_ENV_ALLOW_NAMES`, starts with
    one of :data:`SANDBOX_CHILD_ENV_ALLOW_PREFIXES`, or is in *passthrough*; then
    runs :func:`drop_sandbox_denied_env` over the result, so neither list can
    re-admit a credential or a terminal name; then sets
    :data:`ENV_CHAIN_APPLIED_ENV` so the child does not load the env chain.

    Every spawn site that runs agent code calls THIS function, so a name added
    for one path cannot go missing on another. *env* is not mutated, so
    ``os.environ`` may be passed directly.

    Args:
        env: The parent environment.
        passthrough: Extra names to keep, from
            :func:`configured_child_env_passthrough`.
    """
    extra = frozenset(passthrough)
    kept = {
        name: value
        for name, value in env.items()
        if name in SANDBOX_CHILD_ENV_ALLOW_NAMES
        or name.startswith(SANDBOX_CHILD_ENV_ALLOW_PREFIXES)
        or name in extra
    }
    child = drop_sandbox_denied_env(kept)
    child[ENV_CHAIN_APPLIED_ENV] = "1"
    return child
