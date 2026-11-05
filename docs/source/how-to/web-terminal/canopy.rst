.. _how-to-web-terminal-canopy:

CANOPY
======

**CANOPY** shows the machine the way a control-room display does: screens with
live readbacks, setpoints and buttons, plus **Trace** for archived history, a
correlation plot and a waveform view. In an OSPREY deployment it is the
**CANOPY** tab of the web terminal, and the agent works in it with you: it
opens the screen you ask for, arranges tabs, and builds new screens.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - How CANOPY fits into the web terminal, and where its screens come from
   - How to turn it on in a deployment
   - What a click on a screen may write, and what stops it
   - What the agent can and cannot do in the panel
   - How to upgrade CANOPY

   **Prerequisites:** a deployment built with ``osprey build`` (see
   :doc:`../deploy-a-facility`). Sharing screens with everyone has its own
   page, :doc:`canopy-library`.

How it fits together
--------------------

.. raw:: html
   :file: ../../_diagrams/canopy-panel.html

Every user's web terminal runs its **own** CANOPY. It reads and writes the
machine through the terminal's control session, the same one the agent and the
notebooks use. So a screen shows the machine the control-target chip names, and
a click on it is judged like any other write from that login: same tier, same
write switch, same limits, same audit trail. CANOPY has no permissions of its
own.

A screen comes from one of two places:

- **The library** holds the deployment's shared screens, in
  ``data/canopy/library`` of the deployment repository. Everyone sees them, and
  nobody can change them from the panel. :doc:`canopy-library` explains how a
  screen gets in.
- **Your personal store** holds the screens you or the agent built. It lives in
  your own agent data, survives rebuilds, and only you see it.

In a multi-user deployment, every login card has its own CANOPY. A
:ref:`shared card <multi-user-shared-card>` is one CANOPY for everyone who
opens it: they share its screens and saved views, and its writes are audited
under the card's name.

Turn it on
----------

OSPREY does not build CANOPY. It takes a ready-made **payload**: a directory
with CANOPY's Python packages and web pages, built from the canopy repository.

1. **Build the payload** with the canopy repository's
   ``scripts/build-payload.sh`` (its ``--help`` lists the inputs). Build it at
   the commits this OSPREY release was tested against, which this command
   prints:

   .. code-block:: bash

      python -m osprey.integrations.canopy --pins

   A payload built at any other commit stops the build.

2. **Name the payload** in the deployment's ``.env``:

   .. code-block:: text

      OSPREY_CANOPY_PAYLOAD=/srv/canopy/payload

3. **Add the panel to** ``profile.yml``. A deployment made from the
   ``control-assistant`` preset already has it, and the web terminal opens on
   it at the example facility's ``facility/overview``: skip this step. For any
   other profile, add ``canopy`` to ``web_panels``, its three skills to
   ``skills``, and the build step that copies the payload into the deployment:

   .. code-block:: yaml

      web_panels:
        - canopy
      skills:
        - canopy-panel-builder   # build and check screens
        - canopy-operator        # open and arrange what the panel shows
        - canopy-screens         # which screen shows which device
      lifecycle:
        pre_build:
          - name: canopy-payload
            timeout: 30
            run: '"{python}" -m osprey.integrations.canopy.stage'

4. Run ``osprey build`` and ``osprey up``. The **CANOPY** tab is in the
   sidebar.

If ``OSPREY_CANOPY_PAYLOAD`` is unset, the build leaves the panel and its
skills out and says so in one line. To leave the panel out of one tier, list
``canopy`` under ``exclude.web_panels`` and the three skills under
``exclude.skills`` (see :doc:`../build-profiles`).

The panel's optional settings are flat keys under ``config:``, for example the
screen CANOPY opens first and the channel Trace starts with:

.. code-block:: yaml

   config:
     web.panels.canopy.home_screen: facility/overview
     web.panels.canopy.trace_default_query: pv=SR:DIAG:BPM:03:POSITION:X&range=-1h

``osprey build`` refuses a ``home_screen`` the library does not hold. Every key
is in :ref:`the configuration reference <config-web-panels-canopy>`.

Use the panel
-------------

Open a screen from the panel's sidebar or the command palette, or ask the
agent ("open the RF overview"). Screens and tools open as tabs in tiles, which
you drag and split like the terminal's own (:ref:`web-terminal-tiles`). The
**☰** menu at the right end of the top-right tile's bar resets the layout,
saves the open tabs as a collection and reopens recently closed tabs.

**Which machine.** A badge names the machine the values come from. Switch the
control target on the chip and CANOPY follows: the screens go blank for a
moment and return with the new machine's values. A click made on a screen
that still showed the old machine is refused, never sent to the new one. If
the terminal has no control target on record, CANOPY shows the deployment's
default machine, read-only. If it cannot reach the chosen machine, every
channel shows disconnected and every write is refused.

**History.** Trace and the correlation plot read the archive of the machine
you are on. If the deployment archives another machine, or none, they say
*No archiver*. Live values and history never come from two different machines.

**Saved views** in Trace and the correlation plot are yours: they are kept in
your agent data.

Write to the machine from a screen
----------------------------------

A button, slider or entry field on a screen writes as you. The chip's write
switch for the machine decides how:

- **off** --- the write is refused.
- **asked** --- a dialog asks you to confirm each write. A slider asks once,
  when you let go. Declining sends nothing.
- **allowed** --- the write is sent at once, without a dialog.

Your tier, the channel limits and the safety watchdogs apply as well. While a
watchdog has flagged one of the terminal's agent sessions, the panel says
*writes held* with the reason and refuses every click, because the agent
arranges this panel and could have set up the click. When a write ends, a
short notice says whether the machine read back the value you sent, or why the
write was refused. Every write and every refusal is in the audit trail under
your name, with the surface ``canopy``. :ref:`A panel write
<architecture-safety-chain-panel-write>` on the safety chain page follows the
path step by step.

.. warning::

   At *allowed*, take care with a screen **the agent opened**. The agent chose
   the screen and its settings, so its write buttons address what the agent
   chose, and they write without a dialog. Read what a button addresses before
   you press it. The ``canopy-operator`` skill has the agent say which screen
   it opened and how.

After the agent changes the panel, the write buttons in that window pause for
750 ms with a visible notice, so a tab that moves under your pointer cannot
catch a click meant for something else.

What the agent can do in the panel
----------------------------------

The agent reaches your CANOPY and nobody else's. Its CANOPY tools fall into
three groups:

- **Look** --- read a screen, take a screenshot, read a channel, search channel
  names, check a screen for errors. No prompt.
- **Arrange** --- open, close, split and zoom tabs, open a collection, set
  Trace's channels (``get_workspace``, ``set_workspace``). No prompt, because
  none of it touches the machine. What changed is highlighted, each batch is a
  row in the terminal's activity log (**⋯ More**), and anything the agent
  closes comes back with the closed-tabs toast's **Undo** or **☰ → Recently
  closed**.
- **Author** --- create, edit or delete a screen in your personal store,
  convert a legacy panel (``.bob``, ``.opi``, ``.edl``, ``.ui``), or ask for a
  screen to join the library (``request_push``). Each call asks you first.

The agent never writes a channel through CANOPY: its writes stay on OSPREY's
own write tools and their approval. Your own actions win: if you change the
panel while an agent command is on its way, the command fails and the agent
looks again. While a write dialog is open, the agent's arranging waits for it
to close.

.. note::

   In this release the CANOPY payload carries no screen-format specification.
   When the agent checks a screen, the result carries one warning,
   ``spec_unavailable``: the schema checks did not run, everything else did.
   Read such a check as incomplete, not as a pass.

Size the host
-------------

Each CANOPY takes about 270 MiB, and one runs for every login whose tier shows
the panel, whether or not the tab is open. The agent's screenshot tool starts
a headless browser that adds up to 450 MiB while it is in use.
``web.panels.canopy.vole_bytes_max`` caps the live-value buffers of one
CANOPY. For a wall display that many people watch, use one shared card: ten
users on one screen are ten sets of subscriptions on the control system.

.. _canopy-upgrading:

Upgrade CANOPY
--------------

A CANOPY upgrade is an OSPREY upgrade plus a new payload:

1. Upgrade OSPREY to a release tested against the new CANOPY, and read its
   pins with ``python -m osprey.integrations.canopy --pins``.
2. Build a new payload at those commits and point ``OSPREY_CANOPY_PAYLOAD`` at
   it.
3. Regenerate the library's report with the new CANOPY and commit it, even if
   no screen changed (:ref:`canopy-library-report`).
4. Run ``osprey build`` and ``osprey up``.

The rebuild restarts every terminal and ends open agent sessions, so announce
it. Personal stores, saved views and pending push requests live in the users'
agent data and are kept.

To go back, keep the old payload directory: install the OSPREY release it was
built for, point ``OSPREY_CANOPY_PAYLOAD`` at it again, revert the report
commit of step 3, and rebuild.

.. seealso::

   :doc:`canopy-library`
      Add, review, remove and roll back shared screens.

   :ref:`architecture-safety-chain-panel-write`
      How a click on a screen is checked on its way to the machine.

   :ref:`The configuration reference <config-web-panels-canopy>`
      Every ``web.panels.canopy`` key.
