.. _reference-environment-variables:

=====================
Environment Variables
=====================

Most of what a deployment can be told lives in the build profile and is
rendered into ``config.yml``. What is left over — secrets, and the handful of
choices that belong to the *host* a command runs on rather than to the
deployment — arrives from the environment. This page is what that is.

Secrets, and ``.env``
=====================

.. code-block:: bash

   ANTHROPIC_API_KEY=sk-...          # Or OPENAI_API_KEY, GOOGLE_API_KEY, etc.

Provider keys live in the deployment repository's ``.env`` and are read from
there. No environment variable selects which deployment a command acts on:
every lifecycle verb finds the repository by walking up from the working
directory, and ``--repo DIRECTORY`` names another one.

``.env.example``, rendered into the repository by ``osprey build``, is the
starting point: copy it to ``.env`` and fill in what you need. It carries the
variables **the deployment supplies** — the provider keys, whatever the
profile's ``env.required`` / ``env.optional`` declares, and the tokens
``osprey up`` mints for the services it starts. It is not an inventory of every
name the framework reads: the host-level knobs below are set in the environment
of the command you run, not in a file the build renders.

Host-level knobs
================

Each of these belongs to the machine or the invocation rather than to the
deployment, which is why none of them is a config key.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Variable
     - What it does
   * - ``CONTAINER_RUNTIME``
     - ``docker`` or ``podman``, overriding the runtime recorded for this
       deployment for one invocation. See
       :doc:`/how-to/deploy-project/index`.
   * - ``OSPREY_IMAGE_REGISTRY``, ``OSPREY_IMAGE_TAG``
     - The two image axes for one build, outranking ``images.registry`` /
       ``images.tag`` — see :ref:`the two image axes <deployment-image-overrides>`.
   * - ``OSPREY_OFFLINE``
     - ``1`` switches the web interfaces from CDN-hosted libraries to the local
       bundles ``osprey vendor fetch`` downloads. The equivalent config key is
       ``offline``; use the variable for a single run, the key for a
       deployment that is always firewalled.
   * - ``OSPREY_CA_BUNDLE``
     - Path to a CA bundle for hosts behind a TLS-intercepting proxy, so
       ``osprey vendor fetch`` verifies rather than skips. ``SSL_CERT_FILE``
       works too; this one wins when both are set.
   * - ``REGISTRY_PATH``
     - Path to a facility's own component registry module, outranking the
       ``registry_path`` config key. Meant for pointing a container at a
       registry mounted somewhere the config could not have named.
   * - ``OSPREY_TERMINAL_BIND_HOST``
     - The address ``osprey web`` binds to, and **authoritative over both
       ``--host`` and the config**. The multi-user compose sets it on every
       per-user container so the reverse proxy stays the only path in from off
       the host; a single-user ``osprey web`` sets nothing and ``--host`` is
       honoured as given.

``OSPREY_SITE_CA`` and the other names in
:doc:`/how-to/deploy-project/project-image` are Docker **build arguments**
rather than runtime variables: they are passed with ``--build-arg`` when the
project image is built, and mean nothing to a running process.

Names the framework stamps
==========================

A deployment's containers carry more OSPREY variables than the ones above, and
they are not settings. The build renders them: which user a container serves,
which identity its audit records are filed under, which control target and
write posture a session is running with. Setting one by hand does not
reconfigure anything — it detaches a process from the thing that was tracking
it, which is why the audit-critical names are stripped from every MCP server
spec the build renders.

:ref:`The audit identity ladder <audit-trail-identity-ladder>` covers that set
and why it is closed.

.. seealso::

   :doc:`config`
      The settings that arrive from ``config.yml`` rather than the environment.

   :doc:`/how-to/deploy-project/env-chain`
      Which ``.env`` file wins where, and how ``${VAR}`` placeholders in the
      compose files are resolved.
