Installation & Setup
====================

Get OSPREY running in five steps. The whole process takes about 10 minutes.

.. dropdown:: **What you'll have when done**
   :color: info
   :icon: check-circle

   - Node.js, Claude Code, and ``uv`` installed
   - An API key configured for your AI provider
   - OSPREY cloned and ready to use
   - The ability to create projects with ``osprey init`` or ``osprey build``


Step 1: Install Node.js
------------------------

Claude Code requires `Node.js <https://nodejs.org/>`_ 18+.

.. tab-set::

   .. tab-item:: macOS (Homebrew)

      .. code-block:: bash

         brew install node

      If you don't have Homebrew, install it first:

      .. code-block:: bash

         /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   .. tab-item:: Linux

      .. code-block:: bash

         # Ubuntu/Debian
         curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
         sudo apt-get install -y nodejs

         # Or use your distribution's package manager

   .. tab-item:: Windows (WSL2)

      Install Node.js inside your WSL2 environment:

      .. code-block:: bash

         curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
         sudo apt-get install -y nodejs

Verify:

.. code-block:: bash

   node --version
   # You should see v18.x.x or higher


Step 2: Install Claude Code
-----------------------------

.. code-block:: bash

   npm install -g @anthropic-ai/claude-code

Verify:

.. code-block:: bash

   claude --version


Step 3: Install Python tools
------------------------------

OSPREY uses `uv <https://docs.astral.sh/uv/>`_ for fast Python package management.
It handles Python versions and virtual environments automatically.

.. tab-set::

   .. tab-item:: macOS (Homebrew)

      .. code-block:: bash

         brew install uv

   .. tab-item:: Linux / WSL2

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

Verify:

.. code-block:: bash

   uv --version


Step 4: Set up your API key
-----------------------------

Claude Code needs an API key for the AI provider. Set it in your shell profile
so it's always available.

.. tab-set::

   .. tab-item:: CBORG (LBNL users)

      `CBORG <https://api.cborg.lbl.gov>`_ is the LBNL AI proxy. If you don't have
      a key yet, sign up at https://api.cborg.lbl.gov.

      .. code-block:: bash

         echo 'export CBORG_API_KEY="your-key-here"' >> ~/.zshrc
         source ~/.zshrc

      Replace ``your-key-here`` with your actual key. To verify:

      .. code-block:: bash

         echo $CBORG_API_KEY

   .. tab-item:: Anthropic (direct)

      Get an API key from https://console.anthropic.com/.

      .. code-block:: bash

         echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.zshrc
         source ~/.zshrc

   .. tab-item:: Other providers

      OSPREY supports 100+ providers via LiteLLM. Set the appropriate key:

      .. code-block:: bash

         # OpenAI
         export OPENAI_API_KEY="sk-..."

         # Google
         export GOOGLE_API_KEY="..."

      See :doc:`/how-to/configure-providers` for the full list.

.. note::
   Using ``bash`` instead of ``zsh``? Replace ``~/.zshrc`` with ``~/.bashrc``.


Step 5: Clone and install OSPREY
---------------------------------

.. code-block:: bash

   git clone https://github.com/als-apg/osprey.git
   cd osprey
   uv sync --extra dev

This downloads OSPREY and installs all dependencies into a ``.venv`` virtual
environment. It may take a minute or two on first run.

Verify:

.. code-block:: bash

   uv run osprey --version


You're done! 🎉
-----------------

OSPREY is installed and ready to use. Here's what to do next:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **Hello World Tutorial**
      :link: hello-world-tutorial
      :link-type: doc

      Build your first agent with a mock control system. One MCP server, zero
      complexity. Takes about 10 minutes.

   .. grid-item-card:: **Control Assistant**
      :link: control-assistant
      :link-type: doc

      Production control system patterns with channel finder, logbook search, and
      comprehensive tooling.


.. _guided-project-setup:

.. dropdown:: **Guided Project Setup (Build Interview)**
   :color: success
   :icon: comment-discussion

   If you're setting up OSPREY for a specific detector, beamline, or accelerator
   subsystem, the **build interview** walks you through a guided conversation that
   generates a ready-to-build project profile tailored to your system. It also
   handles **migration from existing OSPREY projects** — point it at your old
   project directory and it will scan, classify, and extract everything reusable.

   **Install the interview skill**

   Copy the interview skill to your Claude Code skills directory:

   .. code-block:: bash

      mkdir -p ~/.claude/skills
      cp -r ~/osprey/src/osprey/templates/skills/build-interview ~/.claude/skills/build-interview

   This makes the ``/build-interview`` command available in any Claude Code session.

   **Run the interview**

   Create a working directory for your project and start Claude Code:

   .. code-block:: bash

      mkdir -p ~/my-osprey-project
      cd ~/my-osprey-project
      claude

   In the Claude Code session, type:

   .. code-block:: text

      /build-interview

   Claude will walk you through:

   1. What system you work with and what you need the AI for
   2. Whether you're starting fresh or **migrating from an existing OSPREY project**
      (if migrating, just point it to the directory and it will scan and reuse what it can)
   3. Your EPICS PV names (if you have them — it's OK if you don't yet)
   4. Whether you need read-only or write access
   5. How to connect (simulated data is recommended for starting out)
   6. Whether you'd like a custom monitoring panel in the web dashboard
   7. A review step that checks for anything missing

   The whole interview takes about 10--15 minutes.

   **Tips during the interview:**

   - If you're not sure about a question, say "I'm not sure" — it'll pick a safe default
   - If you have a spreadsheet of PV names handy, that's helpful but not required
   - If you're migrating, have the path to your existing project directory ready
   - You can always re-run the interview later to adjust things

   **Build your project**

   When the interview is done, Claude generates a ``build-profile/`` directory. Then:

   .. code-block:: bash

      uv run osprey build my-project build-profile/profile.yml

   One command. OSPREY reads your profile, validates your selections, copies your
   channel database into the right place, and produces a ready-to-use project.

   To start using it:

   .. code-block:: bash

      cd my-project && claude

   Or for the web dashboard:

   .. code-block:: bash

      uv run osprey web

   **Send feedback**

   After you've tested your project, you can send feedback to the OSPREY team by
   starting a Claude Code session and typing ``/build-interview feedback``. It takes
   about 30 seconds and helps us improve the process.

   See :doc:`/how-to/build-profiles` for the full build profile reference.


.. dropdown:: **Advanced: Container runtime, services & detailed configuration**
   :color: secondary
   :icon: gear

   The steps above cover the core installation. The following are only needed for
   specific use cases.

   **Container Runtime (Docker or Podman)**

   A container runtime is only required if you plan to deploy containerized services
   (Jupyter, simulation IOCs, databases). The core agent workflow does not require
   containers.

   .. tab-set::

      .. tab-item:: Docker Desktop

         Download from the `Docker website <https://www.docker.com/products/docker-desktop/>`_
         and verify:

         .. code-block:: bash

            docker --version
            docker compose version

      .. tab-item:: Podman

         Install from the `Podman website <https://podman.io/docs/installation>`_
         and verify:

         .. code-block:: bash

            podman --version

         On macOS/Windows, also run:

         .. code-block:: bash

            podman machine init
            podman machine start

   **Installing from PyPI** (instead of from source)

   If you prefer to install OSPREY as a package rather than cloning the repo:

   .. code-block:: bash

      uv pip install osprey-framework

   **Deploying Services**

   See :doc:`/how-to/deploy-project` for setting up containerized services like Jupyter
   notebooks, databases, or simulation IOCs.

   **Detailed Configuration**

   See :doc:`/how-to/configure-providers` for provider setup and
   :doc:`/how-to/build-profiles` for the full build profile YAML reference.


Troubleshooting
~~~~~~~~~~~~~~~~

.. dropdown:: Common issues
   :color: warning
   :icon: alert

   **"claude: command not found"**
      Install Claude Code: ``npm install -g @anthropic-ai/claude-code``

   **"osprey: command not found"**
      Make sure you're in the OSPREY directory with the venv active, or prefix with
      ``uv run``: ``uv run osprey --version``

   **MCP connection failed**
      Ensure you're running ``claude`` from your project root where ``.mcp.json`` lives.

   **Provider authentication error**
      Check that your API key is exported: ``echo $CBORG_API_KEY`` (or whichever key
      you're using). Re-source your shell profile if needed: ``source ~/.zshrc``

   **Python version mismatch**
      OSPREY requires Python 3.11+. Check with ``python3 --version``. The ``uv`` tool
      can install the right version automatically.

   **Verification checklist:**

   .. code-block:: bash

      node --version          # Should be 18+
      claude --version        # Should print version
      uv --version            # Should print version
      uv run osprey --version # Should print version
