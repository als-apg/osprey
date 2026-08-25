.. _how-to-add-mcp-server:

Add an MCP Server
=================

OSPREY supports two ways to add MCP servers:

- **Declared in the build profile** — wire in an external server, written in
  any language, with an ``mcp_servers:`` block. No Python code required.
- **Framework server** — create a Python package under
  ``src/osprey/mcp_server/`` with full access to the framework's utilities,
  hooks, and permissions system.


Add an External Server (Profile-Only)
-------------------------------------

Declare the server in the deployment's ``profile.yml``, under the top-level
``mcp_servers:`` key:

.. code-block:: yaml

   mcp_servers:
     my-server:
       command: "npx"
       args: ["-y", "@my-org/my-mcp-server"]
       env:
         MY_API_KEY: "${MY_API_KEY}"

     my-python-server:
       command: "python"
       args: ["-m", "my_package.server"]
       env:
         OSPREY_CONFIG: "{project_root}/config.yml"

Each entry needs a ``command`` — or, for a server that is already running
somewhere, a ``url`` instead; the two are mutually exclusive. ``args`` and
``env`` are optional. ``{project_root}`` is expanded to the project directory
at build time; ``${VAR}`` is left alone for the shell or the container to
resolve at run time.

To set permissions on a server, name its tools:

.. code-block:: yaml

   mcp_servers:
     my-server:
       command: "npx"
       args: ["-y", "@my-org/my-mcp-server"]
       permissions:
         allow: [safe_tool, read_data]
         ask: [write_data, delete_item]

Then rebuild:

.. code-block:: bash

   osprey build

The build records the block in the project's rendered ``config.yml`` (under
``claude_code.servers``) and renders it from there into ``.mcp.json``, which is
what launches the server, and into ``.claude/settings.json``, where a server
named ``my-server`` with ``allow: [safe_tool]`` becomes the permission entry
``mcp__my-server__safe_tool``.

.. warning::

   Declare the server in ``profile.yml``, not in the rendered ``config.yml``.
   ``profile.yml`` is the only durable edit surface: every ``osprey build``
   re-renders ``config.yml``, ``.mcp.json`` and ``.claude/settings.json`` from
   it, so a server written straight into ``claude_code.servers`` in the
   rendered file is gone the next time you build. (``claude_code.servers.*`` is
   a protected key family for that reason — the setup tools and the web
   terminal's config panel refuse to write it.)

To turn off a framework-provided server you do not need, use the profile's
``config:`` overlay — a framework server has no ``mcp_servers:`` entry of its
own to edit:

.. code-block:: yaml

   config:
     claude_code.servers.ariel.enabled: false

.. note::

   One server key has no ``mcp_servers:`` spelling: ``hooks.pre_tool_use``,
   which attaches a hook preset (such as ``approval``) to every tool of one
   server. The renderer reads it from ``claude_code.servers`` in ``config.yml``,
   but the build rewrites each declared server's entry there wholesale after
   the ``config:`` overlay is applied, so an overlay cannot add it to a server
   the profile declares. Reach for a
   :ref:`tool permission <profile-tool-permissions>` — ``ask:`` prompts before
   the call — or a framework server, which can carry hooks of its own.

Every key an ``mcp_servers:`` entry accepts — the remote ``url`` and
``transport`` forms, the ``port`` shorthand, the placeholder rules, and how to
ship the server's own Python package inside the profile so the launch command
finds it — is in :ref:`profile-mcp-servers`.


Create Your Own Framework Server
--------------------------------

A framework server is a Python package inside Osprey itself, with access to
the shared startup helpers, workspace singletons and hook presets. Writing one
is developer work rather than configuration: see
:doc:`/contributing/extending-osprey` for the base pieces, the registry entry
that switches it on, and the test that pins it.
