.. _how-to-canopy-screens:

===============================================
Author and Review Screens in the CANOPY Library
===============================================

Every web terminal runs its own CANOPY (see :doc:`web-terminal/canopy`), and
every one of them serves the same read-only screen library. This page is how a
screen gets into that library, and how it gets out again. There are two ways
in. A **developer** edits the library in the
deployment's source tree with a local CANOPY server and no OSPREY stack
running. A **user** builds a screen in a web terminal, the agent files it as
a push request, and an administrator reviews it on the host. Both end the
same way: one reviewed commit to ``data/canopy/library``, then a build.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Where the library lives, what its report is, and what the build checks
   - The developer avenue, step by step, from a scratch environment to a build
   - When to regenerate the report, including after a CANOPY upgrade
   - How to review a push request: ``list``, ``show``, ``accept``, commit, build
   - What to do when an ``accept`` was interrupted
   - How to remove or restore a screen, and how to roll back a promotion
   - What the freshness check refuses, and what it prints for each refusal

   **Prerequisites:** a deployment that shows the CANOPY panel and has been
   built once (see :doc:`deploy-a-facility`), kept under version control. The
   examples use ``~/my-assistant`` for the deployment and ``~/canopy-dev`` for
   scratch files.

The library and its report
==========================

The library is the directory ``data/canopy/library`` of the deployment
repository. It is the one set of screens every user sees. It holds:

- ``screens/``, with one ``<id>.screen.json`` per screen and the images the
  screens show;
- ``LIBRARY-REPORT.json`` and ``LIBRARY-REPORT.md``, the **report**. It lists
  every file of the library with its hash and, per screen, what the screen
  can write, run, open and load. A reviewer reads the markdown file; the
  tools read the JSON file.

Nothing else belongs there. A build copies the library into every image that
shows the CANOPY panel, and every user's CANOPY serves it read-only as
``library/<id>``. A user's own screens live in a personal store in that user's
agent data and never appear in the repository until a push is accepted.

``web.panels.canopy.home_screen`` names the screen CANOPY opens first (see
:ref:`config-web-panels-canopy`).

The report is generated, never edited by hand, and it must match the files.
Three checks hold that together:

- ``osprey build`` runs a freshness check before it builds anything (see
  `The freshness check`_);
- the image build recomputes the whole report with the CANOPY inside the image
  and fails when a row differs;
- ``push show`` and ``push accept`` run the same freshness check before they
  review against the library.

So a change to the library is always two things in one commit: the screen
files and the regenerated report.

.. note::

   Both avenues assume the deployment is a git repository of its own, which is
   what ``osprey init`` creates. Git holds the history of the library: the
   push tool keeps replaced files for 14 days only, and every removal, restore
   and rollback below is a git operation.

.. warning::

   ``osprey init --force`` replaces the deployment's ``data/`` directory, the
   library included. A screen that was accepted or authored but not committed
   is destroyed by it. Commit the library first.

Facility collections
====================

A collection opens several library screens and tools together in one view.
Facility collections are JSON files in ``data/canopy/collections`` of the
deployment repository, one ``<id>.json`` per collection, and every user sees
them read-only. A tool member may carry its own ``query``; one without opens at
the tool's default and is the same tile as the bare tool. The image build checks the directory with
``python -I -m canopy_mcp.collections_cli data/canopy/collections
--library data/canopy/library`` and fails when a collection is refused, for
example one that names a screen the library does not hold. The format and
the check are CANOPY's; see the Collections page,
``docs/how_to/collections.md`` in CANOPY, at the commit this OSPREY is proven
against.

The CANOPY skill
================

The agent learns the screen tools from the ``canopy-panel-builder`` skill.
The text is CANOPY's own and ships with the ``canopy_mcp`` package; OSPREY
bundles an identical copy. A profile that shows the CANOPY panel selects it
under ``skills``, and a build writes
``.claude/skills/canopy-panel-builder/SKILL.md`` into the project. The bundled
apps inherit nothing from the deployment's profile, so they carry neither. A
profile of your own that excludes the panel should exclude the skill with it:

.. code-block:: yaml

   exclude:
     skills:
       - canopy-panel-builder
     web_panels:
       - canopy

The skill is the same for both avenues and names no deployment path. Everything
specific to an OSPREY deployment is on this page.

The developer avenue
====================

You run a CANOPY server on your own machine, pointed at the library in the
deployment's source tree, and let Claude Code author screens through it. No
OSPREY container runs, and this server is nobody's panel: it is a scratch
instance that exists to write files into the source tree. Screens render against mock PV values, so a screenshot
shows the layout, not the machine.

1. Create a scratch environment
-------------------------------

Install the two wheels the build staged into the deployment root, CANOPY with
its ``mcp`` extra and PyDMConverter, into a fresh virtual environment, then
install the screenshot browser:

.. code-block:: bash

   python3 -m venv ~/canopy-dev/venv
   ~/canopy-dev/venv/bin/pip install "<canopy wheel>[mcp]" <PyDMConverter wheel>
   ~/canopy-dev/venv/bin/python -m playwright install chromium

Use the wheels of the payload the deployment builds with. The report is
recomputed inside the image by that same CANOPY, so a report written by a
different version can fail the image build.

2. Install the skill
--------------------

.. code-block:: bash

   ~/canopy-dev/venv/bin/python -m canopy_mcp.skill install ~/canopy-dev/project

This writes ``.claude/skills/canopy-panel-builder/SKILL.md`` below the scratch
project, creates the directories it needs, and prints the path. Start Claude
Code in that directory.

3. Create an access token
-------------------------

The server asks every caller for a credential. Write a fresh one into a file
named ``*.key`` that only you can read:

.. code-block:: bash

   (umask 077 && python3 -c 'import secrets, sys; sys.stdout.write(secrets.token_urlsafe(32))' > ~/canopy-dev/access.key)

Keep the file outside the deployment repository.

4. Start the server
-------------------

.. code-block:: bash

   umask 022
   unset CANOPY_PUSH_DIR
   CANOPY_ACCESS_TOKEN_FILE="$HOME/canopy-dev/access.key" \
   CANOPY_DRAFTS_DIR="$HOME/my-assistant/data/canopy/library" \
   CANOPY_VIEWER_DIST="$HOME/my-assistant/canopy_payload/viewer" \
   CANOPY_PV_MODE=mock \
   CANOPY_VIEWS_DIR="$HOME/canopy-dev/views" \
   CANOPY_BROWSER_SANDBOX=1 \
   ~/canopy-dev/venv/bin/python -m canopy_mcp --http --port 8765

Each setting has a reason:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - Why
   * - ``umask 022``
     - New screen files must be readable by everyone and writable by you
       alone. The freshness check refuses anything else (see
       `The freshness check`_).
   * - ``CANOPY_ACCESS_TOKEN_FILE``
     - The token file of step 3. Without it the server answers anyone who can
       reach the port.
   * - ``CANOPY_DRAFTS_DIR``
     - The library is the store the tools write to. Screen ids are plain
       ``<id>`` here, not ``library/<id>``.
   * - ``CANOPY_VIEWER_DIST``
     - The viewer the build staged under ``canopy_payload/``, so screens
       render as they will in the deployment. Its ``ir-map.json`` is the
       widget catalog, and a staged spec is found beside it, at
       ``canopy_payload/spec``. A payload without one makes the checks answer
       ``spec_unavailable`` (see `Checks without a staged spec`_).
   * - ``CANOPY_PV_MODE=mock``
     - No control system is reachable from here.
   * - ``CANOPY_VIEWS_DIR``
     - Saved views default to a ``views/`` directory inside the store. The
       library takes nothing but screens and its report, so they go elsewhere.
   * - ``CANOPY_BROWSER_SANDBOX=1``
     - On your own machine the screenshot browser runs with its sandbox.
   * - no ``CANOPY_PUSH_DIR``
     - This server takes no push requests: sharing is the commit of step 7.

Any free port will do; the next step uses the same one.

5. Register the server with Claude Code
---------------------------------------

.. code-block:: bash

   claude mcp add -s local --transport http canopy-dev http://127.0.0.1:8765/mcp \
       --header 'X-Canopy-Access: ${CANOPY_DEV_ACCESS:-}'

The single quotes matter: the entry stores the placeholder, not the token.
Export the token in the shell that starts Claude Code:

.. code-block:: bash

   export CANOPY_DEV_ACCESS="$(cat ~/canopy-dev/access.key)"

Without the variable the server refuses the connection, which is how you can
tell the credential is being asked for. Now build, port and check screens
with the agent; the skill tells it how.

6. Regenerate the report
------------------------

After the last change and before the commit:

.. code-block:: bash

   ~/canopy-dev/venv/bin/python -m canopy_mcp.report ~/my-assistant/data/canopy/library --served-at library --write

It rewrites both report files from the screens. If it refuses, it names the
file and the reason: fix that and run it again. The skill gives the agent the
same command without ``--served-at``; ``library`` is the default, and it is
spelled out here because the report must describe the tree as the deployment
serves it.

7. Commit the library
---------------------

Commit ``data/canopy/library`` and nothing else, with the path-limited form:

.. code-block:: bash

   git -C ~/my-assistant add -- data/canopy/library
   git -C ~/my-assistant status --ignored -- data/canopy/library     # nothing may be ignored
   git -C ~/my-assistant commit -m 'canopy: add sr/rf overview' -- data/canopy/library

Read the diff of ``LIBRARY-REPORT.md`` before you commit, and have it reviewed
as you would any change: it says, in a table, what the new screens can write
and run. The ``status --ignored`` line is a check. A file that git ignores
would be missing from the commit but present in the image.

8. Build
--------

How the change reaches the running stack depends on the deployment's
``modules.web_terminals.image_source``:

- **local**: run ``osprey build``, then ``osprey up``;
- **registry**: push to the deployment remote, let the pipeline build and
  publish the images, then run ``osprey up``.

.. _canopy-screens-after-upgrade:

After a CANOPY upgrade
----------------------

First upgrade OSPREY to a release proven against the new CANOPY (see
:ref:`canopy-upgrading`). ``osprey build`` refuses a payload built from any
other commits.

Then regenerate the report every time the deployment moves to a new CANOPY
payload, even when no screen changed. A newer CANOPY may compute the rows differently,
and the image build compares the committed report with what the CANOPY inside
the image computes. Build a scratch environment from the **new** wheels
(step 1), run the command of step 6, read the diff, and commit it with the
form of step 7. An empty diff needs no commit.

The pre-build check prints the same command whenever the report cannot vouch
for the library, for example when the report was written with a format this
OSPREY release does not read:

.. code-block:: text

   review the tree, then regenerate the report from a venv built from the staged payload's wheels: python -m canopy_mcp.report <library> --served-at library --write

Checks without a staged spec
----------------------------

The schema of the screen format and the property schemas of the widgets come
from the CANOPY specification, staged into the payload as ``spec/``. The
payload of this release carries none, as an interim state (see
:doc:`web-terminal/canopy`). ``check`` then returns one warning,
``spec_unavailable``, and runs neither the schema check nor the property
check. Everything the report lists is computed all the same. Do not read a
``check`` with that warning as a screen that passed the schema.

Reviewing a push request
========================

A user builds a screen in the personal store of their own CANOPY, asks the
agent to share it, and the agent calls ``request_push``.
That files a bundle in the user's own volume: the screen, every screen and
image it needs, the directory it should land in, and a note. It publishes
nothing. The agent gives the user a request id, the **rid**, and the user
passes it on to an administrator.

The review runs on the host, with the interpreter that runs ``osprey``, from
inside the deployment:

.. code-block:: bash

   cd ~/my-assistant
   python -m osprey.integrations.canopy.push list
   python -m osprey.integrations.canopy.push show <user>/<rid>
   python -m osprey.integrations.canopy.push accept <user>/<rid> --reviewer <name> --digest <d> [--replace]
   python -m osprey.integrations.canopy.push reject <user>/<rid> --reviewer <name> --reason <text>

From any other directory, name the deployment with ``--repo <dir>``. The
option goes before the verb, and so does ``--library <dir>``, which reviews
against a library other than ``data/canopy/library``. After the verb both are
a usage error:

.. code-block:: bash

   python -m osprey.integrations.canopy.push --repo ~/my-assistant list

Rules that hold for every verb:

- Run it as the user who owns ``data/canopy/library``, never with ``sudo``.
  The tool starts containers, so that user needs to be in the ``docker``
  group.
- The container runtime must be docker; podman is refused in this release.
- Nothing in a bundle is trusted. The tool reads it from a throwaway
  container with no network, re-applies every rule, and prints each value on
  one line with control characters escaped, so a pushed string cannot forge a
  line of the review.
- The tool runs no git command. It prints the git commands for you to run.
- Only one review runs at a time (``var/canopy/push.lock``).

``list``
--------

Prints one line per pending request of every user on the roster:

.. code-block:: text

   bob/0123456789abcdef  created=2026-01-15T09:30:00Z  rf/view -> sr/rf  note=RF overview for the control room

A user's volume keeps the newest 8 requests. An older one is displaced by a
new push, and ``show`` then answers ``bundle evicted from bob's volume
(newest-8 limit); ask the user to re-push``.

``show``
--------

.. code-block:: bash

   python -m osprey.integrations.canopy.push show bob/0123456789abcdef

``show`` validates the bundle, stages it beside a copy of the current library
under ``var/canopy-push/<user>/<rid>/``, has CANOPY recompute the report with
and without the push, and prints the review. Read all of it:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Section
     - What to look for
   * - ``filed from <user>'s volume: …``
     - The reminder that the request comes from that user's volume. An
       approval prompt in the terminal is not proof the user filed it: ask.
   * - ``request:`` and ``note:``
     - The source screen, the target directory and the user's words.
   * - ``== files ==``
     - Every file that would land, with its size and hash.
   * - ``== writeBindings of the pushed screens (n) ==`` and one section per
       other report key
     - What the pushed screens can do: PVs they can write, command buttons,
       execute actions, displays they open, absolute URLs. This is the heart
       of the review.
   * - ``== deltas: would-be minus current (both recomputed) ==``
     - Rows that change for screens **already in the library**, for example
       an existing screen whose dangling reference the push resolves.
   * - ``== destinations ==``
     - Each file as ``new`` or ``COLLIDES with an existing library file``. A
       collision needs ``accept --replace``; commit the library first, so the
       replaced file is in git history.
   * - ``== free text of the pushed screens (JSON pointer: value) ==``
     - Every other string in the screens, with where it sits: labels, rule
       values, metadata.
   * - ``spec_unavailable: …``
     - Printed when the checks ran without a staged spec, that is, without the
       screen-format schema and the property check. The review is then yours
       alone for those two; ``accept`` prints the line again and records it in
       the log.
   * - ``digest: <64 hex>``
     - The last line. It covers the pushed files, the request and the current
       library. ``accept`` must repeat it.

A request that breaks a rule is refused with a message on stderr and its own
exit code, and nothing is staged for ``accept``.

``accept``
----------

.. code-block:: bash

   python -m osprey.integrations.canopy.push accept bob/0123456789abcdef \
       --reviewer carol --digest <the digest show printed>

``--reviewer`` names the administrator who read the review. It must be a
roster user whose render holds the deployment-editing setup tool, as the
admin tier does. It is an attestation, recorded in the log
together with the operating-system user, not an authentication.

``accept`` refuses unless the digest is the one ``show`` printed for the
library as it is now. If the request or the library changed since, run
``show`` again and read the new review. It then appends the decision to
``var/canopy/push-log.jsonl``, writes the files into
``data/canopy/library/screens/<target_dir>/``, writes the report last, and
prints what to do next:

.. code-block:: text

   accepted bob/0123456789abcdef -> screens/sr/rf/ (reviewer carol, digest <d>)
   library: <deployment>/data/canopy/library
   pushed (compare with: git -C <deployment> diff --cached -- data/canopy/library):
     screens/sr/rf/view.screen.json
     LIBRARY-REPORT.md
     LIBRARY-REPORT.json
   commit the library with the path-limited form; an uncommitted developer-avenue change in the library is committed with it:
     git -C <deployment> add -- data/canopy/library
     git -C <deployment> status --ignored -- data/canopy/library     # nothing may be ignored
     git -C <deployment> commit -m 'canopy: accept bob/0123456789abcdef (sr/rf), reviewer carol, digest <d>' -- data/canopy/library
   then build: run osprey build, then osprey up
   until osprey build runs, osprey up and osprey restart refuse on source drift; osprey up --as-built restarts the stack as last built

Commit and build
----------------

Run the three git commands as printed. After ``add``, compare the staged diff
with the list of pushed paths: they must be the same files. The commit takes
everything under ``data/canopy/library``, so a developer-avenue change that
was not committed yet goes into the same commit.

The ``then build:`` line names the step for this deployment. With a registry
image source it reads ``push to the deployment remote, let the pipeline build
and publish the images, then osprey up``.

Between the accept and the build, the source tree is ahead of the running
stack. ``osprey up`` and ``osprey restart`` refuse in that state;
``osprey up --as-built`` restarts the stack as it was last built.

If the deployment is not a git repository, the first line of the output says
so: ``this deployment has no repository of its own; put it under version
control, or the replaced panels are kept 14 days only``.

``reject``
----------

.. code-block:: bash

   python -m osprey.integrations.canopy.push reject bob/0123456789abcdef \
       --reviewer carol --reason "writes SR:RF:PHASE:SP without a confirmation"

This records the rejection and its reason in the log, under the same reviewer
rule as ``accept``. It changes nothing in the library or in the user's volume;
tell the user what to fix.

Exit codes
----------

Every verb exits 0 on success, and every refusal has its own code, so a script
can tell them apart. The ones a reviewer meets most:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - 2
     - The command line is wrong.
   * - 10
     - Run as root: ``run push review as the deployment's owner; add that user
       to the docker group instead of using sudo``.
   * - 11
     - The invoking user does not own the library.
   * - 14
     - The runtime is podman.
   * - 20
     - Another review holds the lock.
   * - 21
     - The library is not fresh (see `The freshness check`_).
   * - 25
     - The bundle was evicted from the user's volume.
   * - 56
     - ``--reviewer`` is not a roster user whose render holds the setup tool.
   * - 57
     - No checked stage exists for the request: run ``show`` first.
   * - 58
     - ``--digest`` is not the digest of the request against the current
       library.
   * - 60
     - A destination collides and ``--replace`` was not given.

Codes 26 to 52 are the refusals of a bundle itself (a link, a reserved name,
an SVG, a reference that would not resolve in the library, and so on); the
message names the file and the rule.

An interrupted accept
---------------------

``accept`` logs first and writes the report last, so a run that was killed
part-way leaves a library the freshness check refuses. No build, ``show`` or
``accept`` proceeds until you choose one of two recoveries. The refusal
(exit 21) lists the paths that differ from the report and prints both:

- **Discard.** Remove the added paths and any leftover ``.*.tmp`` file, and
  restore each changed path from the replaced-bytes directory the log entry
  names (``replaced_dir`` in ``var/canopy/push-log.jsonl``) or from git. Then
  run ``show`` and ``accept`` again.

  The refusal also prints a blanket form, ``git restore --source=HEAD --staged
  --worktree -- data/canopy/library`` followed by ``git clean -fd --
  data/canopy/library``. It discards **every** uncommitted change in the
  library, and the warning beside it names each accepted request that would be
  lost.
- **Keep.** Review the tree, regenerate the report (step 6 of the developer
  avenue), then run ``show``, which prints a new digest, and ``accept``. The
  files that already landed equal the request's, so they do not collide.

Removing or restoring a screen
==============================

There is no push verb for either. Both are changes to the source tree, made
and committed like a developer-avenue change.

To remove a screen:

.. code-block:: bash

   git -C ~/my-assistant rm -- data/canopy/library/screens/sr/rf/view.screen.json

To bring back a screen as it was at an earlier commit:

.. code-block:: bash

   git -C ~/my-assistant restore --source=<sha> -- data/canopy/library/screens/sr/rf/view.screen.json

Then, for both, regenerate the report (step 6), commit the library with the
path-limited form (step 7) and build (step 8). Read the report diff first: a
screen that embedded or opened the one you removed now lists a reference that
no longer resolves.

A file that an ``accept --replace`` overwrote is in git history when the
library was committed before the accept. It is also kept under
``var/canopy-push/<user>/replaced/<rid>.<time>/`` for 14 days.

Rolling back a promotion
========================

A promotion is the commit you made after an ``accept``. To undo one or
several:

1. Revert the accept commits one at a time, newest first, without committing
   yet. Run both commands for each commit:

   .. code-block:: bash

      git -C ~/my-assistant revert --no-commit <accept commit>
      git -C ~/my-assistant checkout HEAD -- data/canopy/library/LIBRARY-REPORT.json data/canopy/library/LIBRARY-REPORT.md

   Expect the first command to stop with a conflict in the two report files
   and a non-zero exit, unless the commit is the newest one in the library.
   Every library commit rewrites the report, so a revert of an older commit
   cannot merge it. The screens are reverted all the same. The second command
   puts the committed report back, which ends the conflict; step 3 rewrites
   both files.

   Do not name several commits in one ``git revert``: git stops at the first
   conflict and never reaches the others.

   A conflict in any other file means a later commit changed a screen this
   one added. Decide what the file should hold, write it, and ``git add`` it
   before the next revert.

2. Leave the revert state and check what is staged:

   .. code-block:: bash

      git -C ~/my-assistant revert --quit
      git -C ~/my-assistant status

   ``status`` must not say ``You are currently reverting`` and must list no
   unmerged path. Under ``Changes to be committed`` it must list every screen
   of every request you are rolling back, as ``deleted``, or as ``modified``
   where the accept replaced a file. A screen that is missing from that list
   is still in the library: revert its commit as in step 1.

3. Regenerate the report (step 6 of the developer avenue). Until then the
   report describes the library as it was before the roll-back.

4. Commit the library with the path-limited form:

   .. code-block:: bash

      git -C ~/my-assistant add -- data/canopy/library
      git -C ~/my-assistant status --ignored -- data/canopy/library     # nothing may be ignored
      git -C ~/my-assistant commit -m 'canopy: roll back <user>/<rid>' -- data/canopy/library

5. Build as in step 8: ``osprey build`` and ``osprey up`` with a local image
   source, or the pipeline and ``osprey up`` with a registry.

Retagging or restoring an image rolls nothing back: with a local image source,
``osprey up`` rebuilds the images from the source tree, so the library in git
is what runs. The push log keeps the original ``accepted`` entry. It records
who attested what, and git records what the library held.

The freshness check
===================

One check answers whether the report still covers the library. ``osprey
build`` runs it before the build, and no switch turns it off. ``push show``
and ``push accept`` run it too. A deployment with no library, or an empty one
with no report, passes. Otherwise it refuses, with one line per problem, when:

- the library holds files but no ``LIBRARY-REPORT.json``, or the report
  carries no digest of the file list;
- the digest in the report is not the digest of the files that are there. The
  refusal lists every path that was added, changed or is missing;
- the report was written with a format this release does not read;
- the library holds something it does not take: a file outside ``screens/``
  other than the two report files, a symlink, a dotfile, a leftover
  ``.*.tmp``, a file that is neither a screen nor an image, or two names that
  differ only in case. The line names the file. A ``.DS_Store`` is ignored;
- an ``.svg`` file is present. SVG is markup a browser runs: convert it to a
  raster image;
- a file or directory has the wrong permission bits for others.

A refusal about the report ends with the regeneration command shown under
`After a CANOPY upgrade`_. Review the tree before you run it: regenerating
makes the report vouch for whatever is there.

File modes
----------

The mode rule looks only at the bits for *others*. A file must be readable by
others, a directory readable and searchable by others, and neither may be
writable by others. The image copies the library as files owned by root, so
those bits are all that reaches the user the terminal runs as: it must be able
to read every screen and to change none.

The group bits are not checked. A checkout made under ``umask 002``, where
every file is group-writable, is fine. A refusal looks like this and ends
with the command that fixes it:

.. code-block:: text

   mode: screens/sr/rf/view.screen.json: not readable by others (mode 0600)
   fix the modes with: chmod -R o+rX,o-w <library>

.. seealso::

   :doc:`web-terminal/canopy`
      The CANOPY panel itself: one instance per user, writes, costs, upgrades.

   :doc:`build-profiles`
      Selecting and excluding skills and web panels in a profile.

   :doc:`deploy-a-facility`
      The deployment repository, its image source and the build pipeline.
