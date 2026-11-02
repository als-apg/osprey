:html_theme.sidebar_secondary.remove: true

.. _architecture-applications:

Applications
============

An application is a deployment's own code running beside the agent: containers, a panel and a
tool server, described by ``apps/<name>/app.yml``. The build turns that manifest into the same
things a profile spells by hand, so an application adds no new way to the machine: its writes
meet the one chain every governed write meets.

.. raw:: html
   :file: ../_diagrams/applications.html

``src/osprey/templates/apps/`` is unrelated: it holds packaged data bundles, named by a preset's
``app_template:`` key and materialised by ``osprey init``. The word keeps that other meaning
there.

Projection in the build
=======================

``osprey build`` checks a deployment's applications before it installs or writes anything
(``osprey.apps.loader.check_deployment``): the manifests, each login's selection, the ports, the
pinned versions, the vendor files. It then projects them (``osprey.apps.projection``), through
the paths a profile's own entries take:

- **For the deployment**, each container becomes a ``services.<key>`` entry, its port published
  on loopback, so a panel is reached through the terminal's proxy only. An ``image:`` container's
  compose fragment renders from a stated, strict context (``compose_generator.FRAGMENT_CONTEXT``);
  a fragment that reads any other name, carries a compose key outside the stated schema, binds
  the guarded-run, limits or control-identity paths, or publishes a port off loopback is refused
  when the compose files render, before the build publishes its render. A container on the compose bridge that drives the target is named the
  Channel Access gateway the render's tier selects: the read-only one unless the tier arms writes.
- **For each login** that shows the application, its render gains the panel's ``web.panels.<id>``
  block, a tool server (``mcp__<name>__<tool>``) whose write tools are its ``writes:`` list, the
  approval declaration of each write tool (``apps.<name>.approval.<tool>``), and the tools a
  dispatch job may not run (``dispatch.denied_app_tools``).

A login arms an application when it shows it and its tier is ``asked`` or ``allowed``. A shared
login arms none. A login is one layer kind, a file under ``tiers/``, whichever applications it
shows. A container declaring ``drives_target`` is staged the login map its writes
are judged against.

In the web terminal the panel is a sidebar row. It comes from ``GET /api/panels`` as a
config-defined custom entry (id, label, url, path, health endpoint) and is available while that
url is published.

The approval path
=================

An application write is judged by the one chain: the login's ``control_system.writes`` ceiling
(``never``, ``asked`` or ``allowed``), the owner's per-machine switch position, then the approval
record. In code that is ``posture_store.write_verdict``, with its stdlib twin in the hooks
(``osprey_target_state.py``), and ``approval_record.require_approval_record``. The write targets
whichever machine the session points at.

The switch's position decides whether a person is asked. The application's declaration only
describes and binds the call: what the prompt shows, and which fields of the call and of the
panel's state it is held to. No per-tool approval policy exists for an application tool.

The tools enter the gate as their server's ``writes:`` list, through ``facility_write_tools``, the
one producer of the gated set. The build renders the declarations into ``hook_config.json`` as
``app_approvals``, so the approval hook reads them and never imports application code. At
the position ``asked`` a person answers a prompt bound to what it showed; at the position
``allowed`` the call runs bound and stamped the same way. A stamp's approver is ``human``,
``dispatch_policy`` or ``write_switch``.

A call is refused, not asked, when its ``require`` is unmet, when the state it binds cannot be
read, or when the render does not hand the tool to the switch. A dispatch run runs no tool that
binds fetched state or that ``dispatch.denied_app_tools`` names. On the far side of the hook the
tool server repeats the check (``osprey.apps.sdk.check_approved_call``), and refuses a stamp whose
machine is no longer the session's.

Trust boundaries
================

**The proxy write refusal.** On a login that does not arm the application, the terminal's proxy
refuses every write method and every WebSocket to its panel before anything leaves the terminal:
``refused: login '<login>' does not arm application '<app>' for writes``. The agent reaches a
backend only at the routes the manifest declares as ``agent_routes``, and such a login holds only
the safe-method ones. The refusal reads the tier, not the switch; a write that passes meets the
switch further down.

**The shell gate.** An agent with a shell could read a launch token from its environment or call
a write route directly. Lifting a tool in ``tool_names.SHELL_TOOLS`` on an ``asked`` or
``allowed`` tier fails the build (``_lint_write_tools_are_gated``). A render holding an
application's launch token or a write route beside an open shell
(``renders_with_open_shell``) is refused at deploy, and ``dangerously_allow_bash`` does not waive
it.

**The import boundary.** Application code imports OSPREY only through ``osprey.apps.sdk``,
``osprey.facility.kinds`` and the public ``osprey_connectors`` modules. ``osprey app check`` reads
each package's source for it (``osprey.apps.boundary``). It counts import statements, not an
``importlib`` string: it keeps a package on the contract and is not a sandbox.

**The command write.** ``guarded_run.command_write`` fires a set of commands: setpoints whose
write acts and holds no setting. The facility definition says what a command is, not the caller.
The whole set is refused before anything is written unless the process holds a run lock whose
owner and machine are the write's, the limits state it recorded still holds, and every address is
a command. The connector then judges each command as it does every write.

**First-party panels.** A projected panel is config-defined, so ``panel_cap.opaque_panel_kind``
answers ``None``: the terminal treats it as the deployment's own and does not cap it. In return,
a backend set up with the SDK's ``configure_app`` runs the interfaces gate on its own routes: it
refuses ``Origin: null``, and form bodies outside ``FORM_BODY_ROUTES``. Anything opaque it serves
carries ``opaque_origin.opaque_headers`` or ``document_headers``.

These stand in front of the machine, not in place of the connector: every put a backend or tool
server makes still meets the verdict, the limits and the audit trail on
:doc:`the safety chain <safety-chain>`.
