.. _reference-environment-variables:

=====================
Environment Variables
=====================

The environment variables the OSPREY CLI reads, and where they belong.

.. code-block:: bash

   ANTHROPIC_API_KEY=sk-...          # Or OPENAI_API_KEY, GOOGLE_API_KEY, etc.

Provider keys live in the deployment repository's ``.env`` and are read from
there. No environment variable selects which deployment a command acts on:
every lifecycle verb finds the repository by walking up from the working
directory, and ``--repo DIRECTORY`` names another one.
