.. _how-to-canopy-library:

Share CANOPY Screens
====================

The **library** is the set of CANOPY screens every user of a deployment sees.
It is a directory of the deployment repository, so a screen joins it the way
code does: someone reviews the change, commits it, and the next build ships it
to every terminal. This page is for the person who looks after the
deployment. Using the panel is on :doc:`canopy`.

.. raw:: html
   :file: ../../_diagrams/canopy-library.html

There are two ways in. A **user** builds a screen with the agent in their own
CANOPY and asks to share it; an administrator reviews the request and accepts
it. A **developer** runs a CANOPY on their own machine and edits the library in
the source tree directly. Both end in the same kind of commit.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the library holds, and why every change regenerates its report
   - How to review and accept a screen a user shared
   - How to build screens on your own machine
   - How to remove a screen, or undo an accepted one

   **Prerequisites:** a deployment that shows the CANOPY panel (see
   :doc:`canopy`), kept in git, as ``osprey init`` sets it up. The examples
   use ``~/my-assistant`` for the deployment.

The library and its report
--------------------------

``data/canopy/library`` holds:

- ``screens/``: one ``<id>.screen.json`` per screen, and the images the
  screens show. A screen's id is its path below ``screens/``, such as
  ``sr/rf/overview``.
- ``LIBRARY-REPORT.md`` and ``LIBRARY-REPORT.json``: the **report**. CANOPY
  generates it from the screens. It lists, per screen, the channels it can
  write, the commands it can run, and the displays and web addresses it can
  open. It is what a reviewer reads: the diff of ``LIBRARY-REPORT.md`` shows
  what a change lets people do to the machine.

One rule keeps the two honest: **every change to the screens is one commit
together with the regenerated report**. ``osprey build`` checks that the report
still matches the files before it builds anything, and the image build has
CANOPY recompute the report and compare.

A deployment made from the ``control-assistant`` preset starts with the
example facility's screens: ``facility/overview``, a plan of the whole machine
where each part opens its own screen (``gtl/GTL``, ``br/overview``,
``bts/BTS``, ``sr/sector-01`` to ``sr/sector-12``), and ``sr/orbit-feedback``.
They are a starting point to replace with your facility's own.

.. warning::

   ``osprey init --force`` replaces the deployment's ``data/`` directory, the
   library included. Commit the library first.

Accept a screen a user shared
-----------------------------

A user asks the agent to share a screen. The agent's ``request_push`` files a
**push request** in that user's own data: the screen, every screen and image
it needs, where it should go, and a note. Nothing is published yet. The user
gets a request id and passes it to you.

You review it on the host, from inside the deployment, as the user who owns
``data/canopy/library``. Do not use ``sudo``: put that user in the ``docker``
group instead. Docker is the only container runtime the review supports.

.. code-block:: bash

   cd ~/my-assistant
   python -m osprey.integrations.canopy.push list                      # every pending request
   python -m osprey.integrations.canopy.push show bob/0123456789abcdef  # the review

``show`` checks the request again from scratch, because nothing a user sends is
trusted, and prints the review. Read all of it:

- the files that would land, and whether any would **replace** a library file;
- what the new screens can do: channels they write, commands they run,
  displays and addresses they open;
- what changes for screens already in the library;
- every other piece of text in the screens, so nothing hides in a label;
- on the last line, a **digest** that stands for exactly this request against
  the library as it is now.

If you agree, accept it with that digest and your name. ``--reviewer`` must be
a user allowed to edit the deployment, such as one on the admin tier. It is
recorded as your word, not checked like a password.

.. code-block:: bash

   python -m osprey.integrations.canopy.push accept bob/0123456789abcdef \
       --reviewer carol --digest <the digest show printed>

Add ``--replace`` when the review said a file would be replaced. If the
request or the library changed since ``show``, ``accept`` refuses: run
``show`` again. ``accept`` writes the files and the regenerated report into
the library, then prints the commands that finish the job: the ``git``
commands for one commit of the library, and how this deployment builds. Run
them.

If you do not agree, reject it with a reason the user can act on:

.. code-block:: bash

   python -m osprey.integrations.canopy.push reject bob/0123456789abcdef \
       --reviewer carol --reason "writes SR:RF:PHASE:SP without a confirmation"

Both decisions are logged in ``var/canopy/push-log.jsonl``. From another
directory, put ``--repo ~/my-assistant`` before the verb.

.. note::

   In this release the CANOPY payload carries no screen-format specification,
   so the review cannot check a screen against the format's schema. ``show``
   and ``accept`` say so in a ``spec_unavailable:`` line. Everything the review
   lists about writes and commands is computed all the same; the schema part is
   your reading alone.

If ``accept`` is interrupted, the library no longer matches its report, and
``osprey build``, ``show`` and ``accept`` refuse until you repair it. The
refusal lists the files that differ and prints two ways out: discard the
half-written files, or keep them and regenerate the report. Follow it.

Build screens on your own machine
---------------------------------

A developer can edit the library directly, with Claude Code and a CANOPY server
on their own machine. No OSPREY stack runs: channels show mock values, so a
screenshot shows the layout, not the machine.

1. Install the CANOPY and PyDMConverter wheels that ``osprey build`` copied
   into the deployment root (``canopy-*.whl`` and ``pydmconverter-*.whl``) into
   a fresh virtual environment, with the browser for screenshots. Use exactly
   these: the image build recomputes the report with the same CANOPY.

   .. code-block:: bash

      python3 -m venv ~/canopy-dev/venv
      ~/canopy-dev/venv/bin/pip install "<canopy wheel>[mcp]" <pydmconverter wheel>
      ~/canopy-dev/venv/bin/python -m playwright install chromium

2. Install the agent's skill into a scratch project, and make an access token
   that only you can read:

   .. code-block:: bash

      ~/canopy-dev/venv/bin/python -m canopy_mcp.skill install ~/canopy-dev/project
      (umask 077 && python3 -c 'import secrets, sys; sys.stdout.write(secrets.token_urlsafe(32))' > ~/canopy-dev/access.key)

3. Start the server on the library of the source tree:

   .. code-block:: bash

      umask 022                    # screen files readable by all, writable by you
      unset CANOPY_PUSH_DIR        # this server takes no push requests
      CANOPY_ACCESS_TOKEN_FILE="$HOME/canopy-dev/access.key" \
      CANOPY_DRAFTS_DIR="$HOME/my-assistant/data/canopy/library" \
      CANOPY_VIEWER_DIST="$HOME/my-assistant/canopy_payload/viewer" \
      CANOPY_VIEWS_DIR="$HOME/canopy-dev/views" \
      CANOPY_PV_MODE=mock \
      CANOPY_BROWSER_SANDBOX=1 \
      ~/canopy-dev/venv/bin/python -m canopy_mcp --http --port 8765

   ``CANOPY_DRAFTS_DIR`` makes the library the store the tools write to.
   ``CANOPY_VIEWS_DIR`` keeps saved views out of it, since the library holds
   only screens and its report.

4. Register the server with Claude Code, export the token, and start Claude
   Code in ``~/canopy-dev/project``. The single quotes store the placeholder,
   not the token:

   .. code-block:: bash

      claude mcp add -s local --transport http canopy-dev http://127.0.0.1:8765/mcp \
          --header 'X-Canopy-Access: ${CANOPY_DEV_ACCESS:-}'
      export CANOPY_DEV_ACCESS="$(cat ~/canopy-dev/access.key)"

   Now build, port and check screens with the agent; the skill tells it how.

5. Regenerate the report (next section), read its diff, and commit the library
   alone:

   .. code-block:: bash

      git -C ~/my-assistant add -- data/canopy/library
      git -C ~/my-assistant status --ignored -- data/canopy/library   # nothing may be ignored
      git -C ~/my-assistant commit -m 'canopy: add sr/rf overview' -- data/canopy/library

   A file git ignores would be missing from the commit but present in the
   image; the ``status --ignored`` line catches it.

6. Build. For a deployment that builds its images locally, run ``osprey
   build`` and then ``osprey up``. For one that pulls them from a registry,
   push the commit, let the pipeline build the images, then run ``osprey up``.

.. _canopy-library-report:

Regenerate the report
---------------------

Run this after every change to the screens, and after every CANOPY upgrade
even when no screen changed, because a newer CANOPY may compute the report
differently:

.. code-block:: bash

   ~/canopy-dev/venv/bin/python -m canopy_mcp.report ~/my-assistant/data/canopy/library --served-at library --write

Use a virtual environment built from the wheels of the payload the deployment
builds with (step 1 above): a report written by another CANOPY version can fail
the image build. Regenerating makes the report vouch for whatever is in the
library, so look at the files first. Then read the diff of
``LIBRARY-REPORT.md`` and commit it with the screens as in step 5. An empty
diff needs no commit.

Remove, restore or undo a screen
--------------------------------

There is no push command for these. Each is a change to the source tree: make
it with git, regenerate the report, commit and build as above.

.. code-block:: bash

   # remove a screen
   git -C ~/my-assistant rm -- data/canopy/library/screens/sr/rf/view.screen.json
   # bring one back as it was at an earlier commit
   git -C ~/my-assistant restore --source=<sha> -- data/canopy/library/screens/sr/rf/view.screen.json

Read the report diff before you commit: a screen that opened the one you
removed now lists a reference that no longer resolves.

To undo an accepted request, revert its commit. Revert several one at a time,
newest first. Every library commit rewrites the report, so reverting an older
one stops on a conflict in the two report files. Put the committed report
back each time, and regenerate it once at the end:

.. code-block:: bash

   git -C ~/my-assistant revert --no-commit <accept commit>
   git -C ~/my-assistant checkout HEAD -- data/canopy/library/LIBRARY-REPORT.json data/canopy/library/LIBRARY-REPORT.md
   # ...repeat both lines for each commit, then:
   git -C ~/my-assistant revert --quit
   git -C ~/my-assistant status       # every screen you are rolling back is staged

Then regenerate the report, commit the library and build. A conflict in a
screen file means a later commit changed that screen: decide what it should
hold and ``git add`` it before the next revert. Going back to an older image
is not a rollback: the next build ships what git holds.

Facility collections
--------------------

A **collection** opens several screens and tools together in one view. The
deployment's own collections are JSON files in ``data/canopy/collections``,
one ``<id>.json`` each, and every user sees them read-only. The image build
checks them against the library and fails on a collection that names a
screen the library does not hold. The format is CANOPY's own; see its
documentation on collections. The ``control-assistant`` preset ships one,
``control-assistant``, which opens by default: the overview, orbit feedback,
and Trace, the correlation plot and the waveform view, each already showing
channels of the example facility.

When the build refuses the library
----------------------------------

``osprey build`` refuses a library, with one line per problem and its fix,
when:

- the report is missing, or no longer matches the files (the line lists every
  file added, changed or missing);
- the report was written in a format this OSPREY release does not read;
- the library holds something other than screens, images and the report: a
  symlink, a dotfile, a leftover ``.*.tmp`` file, two names that differ only
  in case, or an SVG (SVG is markup a browser runs; convert it to a raster
  image);
- a file is not readable by other users, or is writable by them. The refusal
  prints the ``chmod -R o+rX,o-w`` command that fixes it.

.. seealso::

   :doc:`canopy`
      The CANOPY panel: turning it on, writes, the agent, upgrades.

   :doc:`../deploy-a-facility`
      The deployment repository, its image source and the build pipeline.
