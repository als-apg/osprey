=========================
Run the Multi-User Demo
=========================

How to stand up the two-persona multi-user Web Terminal from a fresh checkout:
one ``osprey build`` renders the demo project, one ``osprey deploy up`` brings up
a landing page and a separate container for each user — **alice**, a control-room
operator, and **bob**, a physicist with the extra scan tooling. No per-user
builds, no registry.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Building the demo project from the ``control-assistant`` preset
   - How ``osprey deploy up`` auto-renders both persona projects and builds
     their images
   - The grouped landing page, and what separates the operator and physicist
     tiers
   - Logging out and switching between users
   - What it takes to move from the ``mock`` control system to the Virtual
     Accelerator so the physicist can run real scans
   - The demo's honest security posture (plain HTTP on a trusted host)

   **Prerequisites:** Docker (or Podman) installed; your model-provider
   credentials (the preset defaults to Anthropic — set ``ANTHROPIC_API_KEY``).
   New to the Control Assistant? Start with
   :doc:`/getting-started/control-assistant`.

The demo in two commands
========================

From a fresh checkout, build the demo project from the bundled preset, then
bring the stack up from inside it:

.. code-block:: bash

   # 1. Render the demo project from the control-assistant preset
   osprey build control-assistant-demo --preset control-assistant

   # 2. From inside the project, bring the whole stack up
   cd control-assistant-demo
   osprey deploy up

That is the entire setup. The ``control-assistant`` preset ships a
``modules.web_terminals`` block that turns this single project into the
two-persona product, so no extra flags or configuration are needed.

.. note::

   The personas' agent needs your provider credentials at run time. Add them to
   the project's ``.env`` before ``osprey deploy up`` (the preset defaults to
   Anthropic — set ``ANTHROPIC_API_KEY``). ``osprey deploy up`` auto-generates
   the dispatcher's bearer tokens into the same ``.env`` for you, so a fresh
   deploy is otherwise secure by default with nothing else to edit.

.. note::

   Running from a **source checkout** of the OSPREY repository rather than a
   released install? Add ``--dev`` to ``osprey deploy up``. The images install
   the framework from PyPI by default, and a source tree's version isn't
   published there; ``--dev`` bakes your local checkout into the images
   instead.

What ``osprey deploy up`` does
==============================

The one ``osprey deploy up`` invocation does three things in order, all from the
single project you built:

#. **Auto-renders the two persona projects.** The preset declares an *operator*
   persona and a *physicist* persona, each its own rendered OSPREY project. For
   any persona whose project directory does not yet exist, ``deploy up`` renders
   it from the persona's ``build_profile`` preset — the equivalent of
   ``osprey build control-assistant-operator --preset control-assistant-operator``
   — landing it as a sibling of the demo project (``../control-assistant-operator``
   and ``../control-assistant-physicist``). An already-rendered project is
   user-owned and never overwritten; a half-written one errors with a
   remediation hint rather than being rebuilt over.

#. **Builds each persona's image.** In the preset's local mode
   (``image_source: local``), ``deploy up`` builds each persona's
   ``…-operator:local`` / ``…-physicist:local`` image itself from that rendered
   project — no registry, no CI.

#. **Brings up the stack.** An nginx reverse proxy (container ``ca-nginx``)
   serves the landing page on ``http://127.0.0.1:9080``, and one Web Terminal
   container comes up per user — ``ca-web-alice`` on host port ``9091`` and
   ``ca-web-bob`` on ``9092`` — each reached through the landing page. (The
   ``ca-`` prefix is the preset's ``facility.prefix``; change it for your site.)

Alongside the two user containers, the as-shipped preset brings up its supporting
services: PostgreSQL and OpenObserve (storage and telemetry), the Bluesky bridge
(port ``8090``), and the event dispatcher with one worker. The Virtual
Accelerator is **not** among them — it joins only when you add its build block,
covered below.

Stop the stack again with ``osprey deploy down``; check on it with
``osprey deploy status``.

.. note::

   The web stack runs with host networking. On Linux, ``http://127.0.0.1:9080``
   is reachable as-is. On **macOS**, a container's "host" is Docker Desktop's
   Linux VM — enable *host networking* in Docker Desktop (Settings → Resources
   → Network) so the stack's ports reach your browser.

   If another OSPREY deployment already occupies a service port on this host
   (the event dispatcher's default ``8020`` is the usual collision), change it
   in the project's ``config.yml`` (``services.event_dispatcher.port``) before
   ``osprey deploy up``.

The landing page
================

Open ``http://127.0.0.1:9080``. The landing page groups the users into cards,
each labelled with the persona it resolves to:

.. figure:: /_static/resources/multi_user_landing.png
   :alt: The multi-user landing page — two user cards, alice tagged OPERATOR and bob tagged PHYSICIST, under a Terminals heading
   :align: center
   :width: 100%

   The grouped landing page: alice resolves to the operator persona, bob to the
   physicist. Click a card to open that user's session.

alice is a bare roster entry, so she resolves to the preset's
``default_persona`` (operator). bob names his persona (physicist) explicitly.
Clicking a card opens that user's terminal at ``/u/<name>/``, proxied by nginx
to the user's own container.

Two tiers, two capability sets
==============================

Each persona is a self-contained OSPREY project with its **own** permissions,
because permissions are a property of a project's ``config.yml`` — the two tiers
are genuinely different agents, not one agent with a UI toggle.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - User
     - Persona
     - Scan tooling
   * - **alice**
     - operator
     - None. The scan-plan authoring skill is excluded and the scan MCP server
       is denied, so scan tooling never appears — even if a downstream overlay
       tried to re-enable it.
   * - **bob**
     - physicist
     - Full. The scan MCP server is switched on, so bob can author and validate
       the two shipped scan-plan types.

The physicist's two scan-plan types are an orbit **response matrix** and an
n-dimensional **grid scan** — the only scan plans the stack ships. They reach
the machine through the Bluesky bridge service the preset brings up on port
``8090``.

Note the verb: bob can *author and validate* scans out of the box, but
**launching** one is a hardware write, and the demo does not arm that path — see
`Running real scans`_ below.

Logging out and switching users
================================

Every session's header carries a logout button. It POSTs to the terminal's
logout route, clears the local session pointer, and returns you to the landing
page. From there, pick another card to open a different user — a switch always
starts a **fresh** session for the new user rather than resuming the previous
one.

Running real scans
==================

Out of the box the demo runs against the **mock** control system
(``control_system.type: mock``): every channel returns a synthesized value, with
no container and no network. That is deliberately safe for browsing and for the
operator tier — but the mock does not track setpoints, so it cannot actually
*run* a scan.

Giving the physicist a control system that scans can move is a **deliberate
reconfiguration**, not a runtime toggle. Two things have to change together, and
both are build-time changes — apply them, rebuild, and redeploy:

#. **Deploy the Virtual Accelerator.** The Virtual Accelerator is a
   containerized PyAT soft-IOC that serves real EPICS Channel Access, so
   corrector setpoints move a lattice and BPM readbacks respond. The shipped
   preset has **no** ``virtual_accelerator:`` block, so nothing serves live
   Channel Access. Supplying that block at build time — its minimal form is the
   single flag ``--set virtual_accelerator.port=5064`` — makes ``osprey build``
   inject the soft-IOC as a deployed service.

#. **Point the connector at it.** Set
   ``control_system.type: virtual_accelerator`` (a single dotted-key override,
   matching the preset's own config convention) in place of ``mock``, so the
   connector — and the scan bridge, which reads the same value — talk to that
   IOC. The physicist's response-matrix and grid-scan plans then settle-wait on
   real ``:RB`` readbacks tracking each ``:SP`` setpoint.

The Virtual Accelerator's full mechanics — the three-state ``control_system.type``
switch, starting the container, write limits, and how ``osprey sim apply``
scenarios behave — are covered in :doc:`/how-to/use-virtual-accelerator`; the
same ``--set`` / override shape used there applies to this stack.

.. note::

   On macOS the Virtual Accelerator runs inside a container VM, and EPICS
   broadcast discovery does not cross that boundary. The container is reached in
   EPICS name-server mode (``EPICS_CA_NAME_SERVERS``) instead — the one
   host-to-container configuration that works reliably across runtimes.

Arming a scan write
-------------------

Even with the Virtual Accelerator in place, **launching** a scan is a hardware
write, and the demo does not arm that path for you. The Bluesky bridge's promote
route is fail-closed: it returns HTTP 503 until ``BLUESKY_PROMOTE_TOKEN`` is set
in the bridge's environment. And ``osprey deploy up`` deliberately **refuses** to
mint that token while ``control_system.writes_enabled`` is on under local Python
execution — an unsandboxed agent could otherwise read the token back and bypass
the write-approval gate — so container-based Python execution is required to run
a scan with writes enabled. Until the path is armed, the physicist can author and
validate the two shipped plan types but not commit them. This is the same honest
posture as the rest of the demo: capabilities are visible, but real writes stay
gated.

Security posture
================

This demo is built to run on a **single trusted host** — your workstation or a
control-room machine you already trust. Be clear-eyed about what that means:

- **Traffic is plain HTTP.** nginx serves the landing page and proxies every
  terminal over unencrypted HTTP on ``127.0.0.1``. There is no TLS.
- **There is no authentication.** Anyone who can reach ``127.0.0.1:9080`` can
  open any user's terminal. The user cards are a convenience for switching
  identities on a shared trusted machine, not an access-control boundary.

Authentication and TLS are recognized, config-gated seams in the web-terminal
schema, but they are **fail-closed and inert** in this release: the schema
defaults to no auth and plain HTTP, and no other value is exercised yet. Do not
expose this stack to an untrusted network or treat the persona split as a
security control — it is a capability split, enforced per project, not an
identity or access boundary.
