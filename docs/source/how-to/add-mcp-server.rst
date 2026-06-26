Add an MCP Server
=================

Osprey supports two ways to add MCP servers:

- **Config-only** — add an external server (any language) via ``config.yml``.
  No Python code required.
- **Framework server** — create a Python package under
  ``src/osprey/mcp_server/`` with full access to the framework's utilities,
  hooks, and permissions system.


Add an External Server (Config-Only)
-------------------------------------

To wire in any MCP server, add it under ``claude_code.servers`` in your
project's ``config.yml``:

.. code-block:: yaml

   claude_code:
     servers:
       my-server:
         command: "npx"
         args: ["-y", "@my-org/my-mcp-server"]
         env:
           MY_API_KEY: ${MY_API_KEY}

       my-python-server:
         command: "python"
         args: ["-m", "my_package.server"]
         env:
           OSPREY_CONFIG: "{project_root}/config.yml"

Each entry needs ``command`` and ``args``.  ``env`` is optional.
``{project_root}`` is expanded to the project directory at build time;
``${VAR}`` passes through shell environment variables.

To set permissions and hooks on a custom server:

.. code-block:: yaml

   claude_code:
     servers:
       my-server:
         command: "npx"
         args: ["-y", "@my-org/my-mcp-server"]
         permissions:
           allow: [safe_tool, read_data]
           ask: [write_data, delete_item]
         hooks:
           pre_tool_use: [approval]

After editing, regenerate the Claude Code configuration:

.. code-block:: bash

   osprey claude regen

The server will appear in ``.mcp.json`` and its permissions will be added
to ``.claude/settings.json``.

To disable a framework-provided server you do not need:

.. code-block:: yaml

   claude_code:
     servers:
       ariel:
         enabled: false


Create a Framework Server
--------------------------

For deeper integration — shared startup utilities, workspace singletons,
hook presets — create a Python package. This section uses the **controls**
server (``osprey.mcp_server.control_system``) as the canonical example.

Every framework MCP server follows a four-step pattern:

1. Create a Python package under ``src/osprey/mcp_server/<name>/``.
2. Define a ``FastMCP`` server instance in ``server.py``.
3. Register tools using ``@mcp.tool()`` decorators in a ``tools/`` sub-package.
4. Add a ``ServerDefinition`` to the framework registry.


Step 1: Create the Package
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   src/osprey/mcp_server/my_server/
   ├── __init__.py
   ├── __main__.py
   ├── server.py
   └── tools/
       ├── __init__.py
       └── my_tool.py

``__init__.py`` needs only a module docstring.

``__main__.py`` provides the ``python -m`` entry point using the shared
startup helper:

.. code-block:: python

   from osprey.mcp_server.startup import run_mcp_server

   def main() -> None:
       run_mcp_server("osprey.mcp_server.my_server.server")

   if __name__ == "__main__":
       main()


Step 2: Define the Server Instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``server.py``, create a module-level ``FastMCP`` instance and a
``create_server()`` factory that initializes dependencies and imports tools:

.. code-block:: python

   import logging
   from fastmcp import FastMCP

   logger = logging.getLogger("osprey.mcp_server.my_server")

   mcp = FastMCP(
       "my-server",
       instructions="One-line description of what the server does",
   )

   def create_server() -> FastMCP:
       """Initialize context, import tools, and return the server."""
       from osprey.mcp_server.startup import (
           initialize_workspace_singletons, prime_config_builder, startup_timer,
       )

       prime_config_builder()
       initialize_workspace_singletons()

       with startup_timer("tool_imports"):
           from osprey.mcp_server.my_server.tools import my_tool  # noqa: F401

       logger.info("My Server MCP server initialised")
       return mcp

Key points:

* The ``mcp`` instance is defined at **module level** so tool modules can
  import it directly.
* ``create_server()`` is called by the startup machinery; it must return
  the ``mcp`` instance.
* Tool modules are imported inside ``create_server()`` so that
  ``@mcp.tool()`` decorators run after context is ready.


Step 3: Register Tools
^^^^^^^^^^^^^^^^^^^^^^

Each tool lives in its own module under ``tools/``.  Import the ``mcp``
instance from ``server.py`` and decorate async functions:

.. code-block:: python

   """MCP tool: my_tool."""

   import json
   from osprey.mcp_server.my_server.server import mcp

   @mcp.tool()
   async def my_tool(name: str, count: int = 1) -> str:
       """Do something useful.

       Args:
           name: The thing to operate on.
           count: How many times to do it.

       Returns:
           JSON result string.
       """
       return json.dumps({"name": name, "count": count, "status": "ok"})

Tool guidelines:

* **Return type** -- always ``str`` (typically JSON).
* **Docstring** -- becomes the tool description the LLM sees; be specific.
* **Error handling** -- return structured JSON errors via
  ``osprey.mcp_server.errors.make_error`` rather than raising exceptions.
* **One tool per file** keeps modules focused and avoids circular imports.


Step 4: Register in the Framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open ``src/osprey/registry/mcp.py`` and add a ``ServerDefinition`` to
``FRAMEWORK_SERVERS``:

.. code-block:: python

   "my-server": ServerDefinition(
       name="my-server",
       module="osprey.mcp_server.my_server",
       env={"OSPREY_CONFIG": "{project_root}/config.yml"},
       permissions_allow=["my_tool"],
       hooks_post=[_post_error("mcp__my-server__.*")],
   ),

Important ``ServerDefinition`` fields:

``name``
    Server name.  Tools are referenced as ``mcp__<name>__<tool_name>``.

``module``
    Python module path.  Launched via ``python -m <module>``.

``env``
    Environment variables.  ``{project_root}`` is the workspace path;
    ``${VAR:-default}`` passes through host env vars.

``permissions_allow`` / ``permissions_ask``
    Tools allowed without confirmation vs. tools requiring operator approval.

``condition``
    Optional context key; server is disabled when the key is falsy.

``hooks_pre`` / ``hooks_post``
    Use ``_APPROVAL`` for human-in-the-loop on safety-critical tools and
    ``_post_error()`` for standard error guidance.

After adding the entry, run ``osprey claude regen`` to regenerate the Claude
Code configuration.  The server will appear in ``.mcp.json``.


Testing
-------

Unit-test tools by calling the async functions directly:

.. code-block:: python

   @pytest.mark.asyncio
   async def test_my_tool():
       from osprey.mcp_server.my_server.tools.my_tool import my_tool
       result = await my_tool("example", count=2)
       assert '"status": "ok"' in result

Place tests under ``tests/mcp_server/test_my_server.py``.
