"""Renderer for the runtime network guard emitted into executed scripts.

PURPOSE: When a deployment runs with an **open navigation perimeter**
(``auth.method: none``), nginx authenticates requests to the deployment's own
web ports on the caller's behalf — a request arriving at one of those ports is
credentialed as whoever owns the port, whoever made it. Executed agent code
therefore must not originate connections to those ports: from inside the
deployment host's network namespace such a connection would arrive at nginx
already trusted. This module renders the guard that refuses exactly those
connections, as source code spliced into the wrapped script ahead of user code
(``services/python_executor/execution/wrapper.py``), sibling to ``fs_guard``.

Why *emitted source* and not an importable guard: the guard runs in a spawned
child that must not depend on osprey being importable there. Everything below
the ``render_net_guard`` return is self-contained text — it imports only
``socket`` and closes over nothing from the parent. The denied ports are baked
in as literals, resolved *by the parent* from the deployment's perimeter stamp
(``OSPREY_WEB_PERIMETER_DENY_PORTS``): a child that re-derived the set could
equally derive an empty one.

Patch surface — three entry points, and why they are enough:

* ``socket.socket.connect`` and ``socket.socket.connect_ex`` — the class
  methods every Python-level TCP connection resolves through.
* ``socket.create_connection`` — the module-level convenience the high-level
  clients call; patched in its own right so the refusal fires before any DNS
  resolution rather than inside the per-address connect loop.

The high-level HTTP stacks are deliberately **not** patched one by one, and
the two funnels they split across are both rows above — neither row is
redundant: ``urllib.request`` and ``http.client`` open their connections via
``socket.create_connection``, while ``urllib3`` (and therefore ``requests``)
ships its *own* ``create_connection`` that builds a socket and calls
``sock.connect`` itself — which resolves through the ``socket.socket.connect``
class rebind. Between the class rebind and the module-level funnel every
Python-level client is covered; enumerating the clients themselves would go
stale against every library added to the environment.

Why the **port alone** is the criterion and the host is never inspected:
loopback, the LAN address, and any hostname of the deployment host all reach
the same nginx in the host network namespace, and SSH-tunnel deployments make
the connection's apparent source indistinguishable from a local one — a host
allowlist would be a sieve. The ports, by contrast, are osprey-allocated for
this deployment, so refusing them toward *any* host costs only the ability to
reach an unrelated remote service that happens to sit on an identical port
number — accepted, and far cheaper than the hole. Non-``(host, port)`` address
families (``AF_UNIX`` string/bytes paths, abstract sockets) carry no port and
pass through untouched, which is what keeps ``multiprocessing`` — whose
parent/worker plumbing runs over ``AF_UNIX`` — working under the guard in
every mode.

``connect_ex`` raises the same refusal instead of returning an errno. Its
callers read a nonzero return as an ordinary, often transient, network
failure — a polling loop would retry it silently and a port scanner would
tally it as "closed" — so an errno would bury the one thing the refusal
exists to deliver: the explanation. Raising keeps every refusal loud and
identically worded, whichever entry point it came through.

LIMIT — defense in depth, not a security boundary
-------------------------------------------------
Same stance as ``fs_guard``, stated rather than defended: the guard installs
itself in the child's own namespace, restore handle
(``_restore_net_patched_targets``) included, so code that knows it is there
takes it down in one call — and the C ``_socket`` module underneath is left
unpatched on purpose, as is ``importlib.reload(socket)``, either of which
re-opens the route. ``NET_GUARDED_MODULES`` names what a static reload check
*could* refuse, but ``socket`` is not folded into ``fs_guard.GUARDED_MODULES``:
that set feeds ``path_policy``'s reload refusal for every execution, while the
net guard exists only under an open perimeter — refusing ``reload(socket)`` in
every deployment would police a guard that is usually not installed. The
residual (a reload disarms the net guard silently where it *is* installed)
adds nothing beyond the restore handle already in scope. What contains code
deliberately attacking the boundary is the perimeter design itself — the
deploy-time gate that refuses to grant shell-capable personas an open
perimeter — not this monkeypatch.
"""

import textwrap
from collections.abc import Iterable

#: Leading text on every refusal the emitted guard raises. The wrapper's
#: callers and the tests match refusals by this prefix, mirroring
#: ``fs_guard``'s ``DEFAULT_DENYLIST_PREFIX`` pattern.
NET_GUARD_REFUSAL_PREFIX = "Refused (navigation-only perimeter):"

#: Modules the emitted guard rebinds names in — the analogue of
#: ``fs_guard.GUARDED_MODULES``, exported so a static layer that chooses to
#: refuse ``importlib.reload`` of them can consume the list. The intended
#: consumer is the static reload check (task 3.3 / ``path_policy``'s layer),
#: not the emitted guard itself. Deliberately NOT merged into
#: ``fs_guard.GUARDED_MODULES`` — see the module docstring's LIMIT
#: section for why ``reload(socket)`` stays a documented residual instead.
NET_GUARDED_MODULES: tuple[str, ...] = ("socket",)

#: Every entry point the guard rebinds, as ``(holder dotted name, attribute)``.
#: ``socket.socket`` rows are rebound on the class (shadowing the inherited
#: ``_socket.socket`` method); the ``socket`` row is a module-dict rebinding.
#: The table is the single source the emitted install loop iterates, so a new
#: entry point needs a row here and, if its holder is new, an alias in the
#: emitted holder map — nothing else.
_NET_PATCH_TARGETS: tuple[tuple[str, str], ...] = (
    ("socket.socket", "connect"),
    ("socket.socket", "connect_ex"),
    ("socket", "create_connection"),
)


def render_net_guard(
    *,
    denied_ports: Iterable[int],
    refusal_prefix: str | None = None,
) -> str:
    """Render the network guard as source code to embed in a child script.

    Args:
        denied_ports: The deployment's own web ports — every port a connection
            from executed code must be refused toward, whatever the host.
            Resolved by the parent from the perimeter stamp and passed as
            literals. Must be non-empty: an empty set means no perimeter is
            open, and the caller skips emission entirely rather than shipping
            an inert guard (``ExecutionWrapper._get_net_guard`` does exactly
            that).
        refusal_prefix: Leading text on every refusal. Defaults to
            :data:`NET_GUARD_REFUSAL_PREFIX`.

    Returns:
        Left-aligned Python source. Indent it yourself (``textwrap.indent``)
        if it lands inside a block.

    Raises:
        ValueError: On an empty port set, a port outside 1–65535, a
            non-integer port, or an empty ``refusal_prefix``.
    """
    ports: list[int] = []
    for port in denied_ports:
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"denied_ports must be integers; got {port!r}")
        if not 1 <= port <= 65535:
            raise ValueError(f"denied_ports must be in 1..65535; got {port!r}")
        ports.append(port)
    if not ports:
        raise ValueError(
            "denied_ports must be non-empty — with nothing to deny the guard is "
            "dead weight, and the caller skips emission instead of rendering it"
        )
    denied = tuple(sorted(set(ports)))

    prefix = NET_GUARD_REFUSAL_PREFIX if refusal_prefix is None else refusal_prefix
    if not prefix.strip():
        raise ValueError("refusal_prefix must be non-empty — refusals are matched by their prefix")

    ports_text = ", ".join(str(port) for port in denied)

    return textwrap.dedent(
        f'''
        # --- OSPREY network guard (generated by render_net_guard) ------------
        # Self-contained: imports nothing from osprey, because the child that
        # runs this may not have osprey importable at all.
        import operator as _osprey_net_operator
        import socket as _osprey_net_socket

        _OSPREY_NET_DENIED = frozenset({denied!r})
        _OSPREY_NET_PORTS_TEXT = {ports_text!r}
        _OSPREY_NET_PREFIX = {prefix!r}
        _OSPREY_NET_TARGETS = {_NET_PATCH_TARGETS!r}
        _OSPREY_NET_HOLDERS = {{
            "socket": _osprey_net_socket,
            "socket.socket": _osprey_net_socket.socket,
        }}

        # Originals, captured BEFORE the corresponding name is rebound. This is
        # also the restore table: a name absent here was never patched.
        _osprey_net_originals = {{}}


        def _osprey_net_check(address):
            """Refuse *address* when it names a denied port; return quietly otherwise.

            Only ``(host, port, ...)`` sequences are judged — AF_INET's 2-tuple
            and AF_INET6's 4-tuple both carry the port at index 1. Anything
            else (an AF_UNIX path string or bytes, an abstract-socket name)
            names no port and passes through. The port is coerced through
            ``operator.index`` because the C layer accepts any ``__index__``
            object (``numpy.int64``, a ctypes integer) wherever it accepts an
            int — a plain ``isinstance(int)`` gate would wave a
            ``numpy.int64`` port straight past the deny set. A value
            ``operator.index`` cannot coerce names no port either, and the
            real entry point rejects it on its own terms. The HOST is
            deliberately never inspected — loopback, LAN address and hostname
            all reach the same perimeter, so the osprey-allocated port is the
            whole criterion.
            """
            if not isinstance(address, (tuple, list)) or len(address) < 2:
                return
            try:
                _port = _osprey_net_operator.index(address[1])
            except TypeError:
                return
            if _port in _OSPREY_NET_DENIED:
                raise PermissionError(
                    f"{{_OSPREY_NET_PREFIX}} connect to port {{_port}} refused. "
                    f"This deployment runs with an open navigation perimeter: "
                    f"requests reaching its own web ports "
                    f"({{_OSPREY_NET_PORTS_TEXT}}) are authenticated at the "
                    f"edge on the caller's behalf, so a connection from "
                    f"executed code would arrive already credentialed. "
                    f"Connections to every other port (EPICS gateways, the "
                    f"archiver, external services) are unaffected."
                )


        def _osprey_net_capture(_dotted, _attr):
            _original = getattr(_OSPREY_NET_HOLDERS[_dotted], _attr)
            _osprey_net_originals[(_dotted, _attr)] = _original
            return _original


        def _osprey_net_wrap_method(_dotted, _attr):
            _original = _osprey_net_capture(_dotted, _attr)

            def _osprey_guarded_connect(self, address, *args, **kwargs):
                _osprey_net_check(address)
                return _original(self, address, *args, **kwargs)

            return _osprey_guarded_connect


        def _osprey_net_wrap_function(_dotted, _attr):
            _original = _osprey_net_capture(_dotted, _attr)

            def _osprey_guarded_create_connection(address, *args, **kwargs):
                _osprey_net_check(address)
                return _original(address, *args, **kwargs)

            return _osprey_guarded_create_connection


        def _install_net_patched_targets():
            for _dotted, _attr in _OSPREY_NET_TARGETS:
                _holder = _OSPREY_NET_HOLDERS[_dotted]
                if not hasattr(_holder, _attr):
                    # Future-proofing against a renamed stdlib entry point:
                    # nothing to patch, nothing to restore.
                    continue
                if _dotted == "socket.socket":
                    _replacement = _osprey_net_wrap_method(_dotted, _attr)
                else:
                    _replacement = _osprey_net_wrap_function(_dotted, _attr)
                setattr(_holder, _attr, _replacement)


        def _restore_net_patched_targets():
            """Put every rebound name back. Idempotent, and safe in a finally."""
            for (_dotted, _attr), _original in list(_osprey_net_originals.items()):
                setattr(_OSPREY_NET_HOLDERS[_dotted], _attr, _original)
            _osprey_net_originals.clear()


        _install_net_patched_targets()
        # --- end OSPREY network guard ----------------------------------------
        '''
    ).lstrip("\n")
