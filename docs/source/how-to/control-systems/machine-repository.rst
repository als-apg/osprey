.. _how-to-machine-repository:

========================================================
Serve the Virtual Accelerator from a Machine Repository
========================================================

A machine repository versions the machine facts the simulator is built from
apart from any deployment that drives it. It serves the Virtual Accelerator on
its own, so the machine can be checked and run without an agent, and every
deployment of the same machine mounts the same commit of it.

What goes where
===============

.. code-block:: text

   example-machine/              the machine repository
     profile.yml                 its VA-only setup
     facility/                   the machine facts
       identity.yaml  records/  limits.yaml  imported/<layer>/
       vocabulary/  provenance/  simulation/

   my-deployment/                a deployment
     profile.yml                 states machine: vendor/example-machine/facility
     vendor/example-machine/     the machine repository, mounted
     facility/
       knowledge/  ariel/  benchmarks/

The machine repository holds every machine fact, including its start set: the
scenarios marked ``start: true`` or ``start_on``, which the simulator
activates when it first comes up. The deployment keeps only what it writes about its own use of the
machine: ``knowledge/``, ``ariel/`` and ``benchmarks/``
(:doc:`/how-to/describe-your-facility`).

The machine's own profile
=========================

The machine repository's ``profile.yml`` serves the Virtual Accelerator and
records it in a stored archive. It runs no agent, so it needs no provider key:

.. literalinclude:: ../../../../tests/fixtures/machine_repository/profile.yml
   :language: yaml
   :caption: profile.yml

``osprey facility import``, ``osprey build`` and ``osprey up`` work there as in
any deployment. Import a new machine there before any deployment mounts it: a
mount needs a machine root that already holds ``identity.yaml``.

Mount it
========

In the deployment, add the repository as a submodule (a nested clone works the
same) and name its facility directory in the root ``profile.yml``:

.. code-block:: bash

   git submodule add <url> vendor/example-machine

.. code-block:: yaml

   machine: vendor/example-machine/facility

``machine:`` resolves against the repository root, and the tree it names is
folded into the profile hash like ``facility/`` (:doc:`/how-to/build-profiles`),
so a new machine commit reports the build out of date.
Only the root ``profile.yml`` states it; every login drives the same
machine. A machine builds to the same facility file standalone and mounted,
except for a measurement kind only the mounting deployment's package registers:
standalone, the kind is deferred with a ``kind-deferred`` warning and the
deployment judges its block (:doc:`/how-to/pyaml/register-a-measurement-kind`).

With ``machine:`` set, ``osprey facility import`` writes into the machine root,
except the knowledge index, which goes under the deployment's
``facility/knowledge/``. Commit the change in the machine repository, then
the new submodule commit in the deployment.

Stops
=====

``osprey facility validate`` and ``osprey build`` stop, naming the path and the
fix, on:

``machine-missing``
   The machine root holds no ``identity.yaml``: the submodule is not
   initialised (``git submodule update --init``), or ``machine:`` names the
   repository top instead of its ``facility``.
``machine-restated``
   The deployment's ``facility/`` holds a file outside ``knowledge/``,
   ``ariel/`` and ``benchmarks/``. Edit the fact in the machine repository and
   remove the deployment's copy.
``profile-invalid``
   ``machine:`` and ``facility/`` overlap; a login file states ``machine:``; or the
   profile states ``simulation.default_scenarios`` beside ``start: true`` or
   ``start_on`` marks or a mounted machine. ``simulation.default_scenarios: null`` cancels a
   preset's value.

.. _machine-repository-upgrade:

Upgrade
=======

When an upgrade moves a facility path, upgrade OSPREY in
every deployment that mounts the machine repository before the machine
repository changes: an older OSPREY reads a moved machine without the moved
files, and says nothing.

#. In each mounting deployment, upgrade OSPREY and run
   ``osprey init --revalidate``. It leaves the machine as it is and prints the
   commands that move it under "Left for you to do by hand"; its builds refuse
   the machine until it moves (:ref:`profile-root-migration`).
#. In the machine repository, upgrade OSPREY, run
   ``osprey init --revalidate`` (or the printed commands) and commit.
#. In each deployment, commit the new submodule commit.
