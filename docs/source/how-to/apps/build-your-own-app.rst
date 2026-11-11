.. _how-to-apps-build:

==================
Build Your Own App
==================

This page takes an app from nothing to running on your deployment. It starts
from the scaffold ``osprey app new`` writes — a working app whose panel shows
one channel and whose agent tools read it and set it behind an approval — and
then goes through each part you will change to make it your own. The example
app is called ``corrector-trim`` and shows and sets one horizontal corrector's
current; use any name of lower-case letters, digits and dashes.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - The six commands from ``osprey app new`` to a panel in the sidebar
   - What each file of the scaffold does and where your code goes
   - How the manifest's ``approval`` entry turns a tool into a write the
     operator approves
   - Which logins show the app, and which may write with it
   - How to give the agent a skill, and the same reach as the operator
   - How an app carries a subagent, and which parts an app can leave out

   **Prerequisites:** a deployment repository made with ``osprey init``
   (:doc:`/how-to/build-profiles`) that builds and starts; Node and npm on the
   machine where you run ``osprey app check``, for the panel's JavaScript
   checks and tests. Read :doc:`index` first for what the parts of an app are.

Make it run
===========

1. Make it
----------

.. code-block:: bash

   osprey app new corrector-trim

This copies the scaffold into ``apps/corrector-trim/``, rewrites every name in
it (``scaffold-app`` to ``corrector-trim``, the Python package to
``corrector_trim``), and lists the app in ``profile.yml``:

.. code-block:: yaml

   apps:
     corrector-trim: {}

Listing an app is what turns it on; delete the entry to turn it off while the
directory stays. The command also adds ``/apps/*/panel/vendor/`` to
``.gitignore``, for files a build fetches into a panel (`Manifest keys`_,
``vendor``).

2. Point it at a channel
------------------------

The scaffold names no channel of its own — which channel to show is the
deployment's choice. Until ``profile.yml`` gives it one, the panel says so and
a write is refused. The ``config:`` block under the app's entry is free-form;
the build writes it to a file the backend reads at start:

.. code-block:: yaml

   apps:
     corrector-trim:
       config:
         channel: SR:MAG:HCM:01:CURRENT:SP

In the example facility that is the current setpoint of horizontal corrector
01, in amperes, with a range of ±10 A and a readback pair the connector
compares it against. Any key you put there reaches your backend the same way, so a facility-specific
setting — a device list, a tolerance, a model name — goes here rather than into
the code.

3. Check it
-----------

.. code-block:: bash

   osprey app check

This runs what ``osprey build`` checks of every listed app — the manifest,
each login's selection, ports, pinned versions — and then what only the app's
own source can tell: the import rule over its Python, ESLint over ``panel/``
and its JavaScript tests under ``tests/``. It builds nothing and prints every
violation with the app and the field it is in. The scaffold passes as copied.
The first run installs a pinned JavaScript toolchain into your user cache,
which needs ``npm`` and network access once.

4. Build and start
------------------

.. code-block:: bash

   osprey build
   osprey up -d

``osprey build`` builds the app's Python package into a wheel, renders its
container, assigns its port, writes its config file and adds the panel and
the tool server to every login that shows the app. ``osprey up`` builds the
project image with the wheel installed and starts the container with the rest
of the stack. ``osprey health`` lists it beside the other services.

5. Open it
----------

Open a terminal and look at the sidebar: under **Apps** there is a
**Scaffold** row — the label is ``panel.label`` in ``app.yml``, which
``osprey app new`` leaves for you to change to *Corrector trim*. Click it: the
panel shows the channel's name, its readback and the machine the readback came
from — the one on your :ref:`control-target chip
<web-terminal-session-posture>`. Switch the chip to another machine and the
panel follows at its next refresh.

Then ask the agent: *what is corrector HCM01 set to?* It calls
``mcp__corrector-trim__read_channel`` and answers from the same backend. Ask
it to *set the corrector-trim channel to 0.5 A*:

- with the switch for that machine *off*, the call is refused and the agent
  says to arm the switch from the chip;
- at *asked*, an approval card appears in the chat: the app, the tool, the
  machine, and the line *Set SR:MAG:HCM:01:CURRENT:SP to 0.5, now 0.32* —
  the scaffold's prompt, filled from the call and from the backend's state.
  Approve it and the backend writes through the connector;
- at *allowed*, the write runs without a card.

Try the same on a read-only login (``bob`` in the ``demo``
preset): the panel shows the readback, and the agent's write tool is denied.

6. Make it yours
----------------

From here the app is ordinary code. The sections below go through the parts
in the order you will touch them: the manifest, the backend, the tool server,
the panel and its tests, then writes, logins, skills and subagents. Run
``osprey app check`` as you go and ``osprey build && osprey up -d`` to see a change in the
terminal.

The manifest
============

``app.yml`` is the one file OSPREY reads about your app. The scaffold's, as
copied:

.. literalinclude:: ../../../../src/osprey/apps/scaffold-app/app.yml
   :language: yaml

The first three blocks describe one part each:

``containers``
   The backend. ``runtime: osprey`` runs the app's Python package on the
   deployment's own image, where OSPREY and its dependencies are already
   installed; ``command`` is what it runs. ``drives_target: true`` gives the
   container the connection to the control system — without it the container
   cannot reach the machine. ``health`` is the route the terminal polls to
   light the sidebar row. ``state`` names a directory that survives restarts,
   for files the app keeps (saved plans or tables, for instance).

``panel``
   The sidebar entry: its ``label``, the container that serves it, and
   ``agent_routes``, the backend routes the agent's tools may call. A route
   with ``GET`` reads; any other method is a write route, and the terminal
   hands write routes to the agent only on a login that arms the app.
   ``launch_token: true`` makes the write route demand a token the terminal
   adds on an armed login only, so nothing else inside the stack can call it.

``tools``
   The tool server: the Python module to run beside the agent, the tools that
   only read, and the tools that write under ``approval``. ``errors`` are the
   two sentences the agent sees when the backend does not answer or refuses.

Each block is optional, and so are ``skills`` and ``agents``; an ``app.yml``
with none of the five is refused. Without ``panel`` the app has no sidebar
row. Without ``containers`` it has no backend: a ``panel`` is refused, and so
is a write tool (an ``approval`` entry), which needs a container with
``drives_target: true``; the tool server only reads or computes and keeps
nothing between calls. Anything that holds state is a backend.

Every key is listed under `Manifest keys`_. Unknown keys are refused, and a
key that belongs to the container's compose file (``volumes``, ``ports``,
``depends_on``) is refused with the place it lives instead.

The backend
===========

``corrector_trim/server.py`` is a FastAPI service. Four routes matter:

- ``GET /`` serves ``panel/index.html``; the panel's other files are served
  from ``static/`` by ``configure_app``;
- ``GET /health`` answers the terminal's poll;
- ``GET /api/state`` is what the panel and the read tool show: the channel,
  the readback and the machine it came from;
- ``POST /api/write`` performs the write.

Two things the backend does on every request are worth seeing in the source,
because your own routes will do the same:

**It answers for the login that asked.** The terminal passes the login's name
to the backend; ``owner_from_header`` reads it and ``owner_target`` turns it
into that login's control target, the machine its chip points at. The backend
never chooses a machine itself, and the same route answers two operators from
two machines when their chips differ. Two Channel Access machines are the
exception: a process has one Channel Access context, so ``build_connector``
refuses a second until the backend restarts.

**It writes through the connector, under that login.** ``build_connector``
asks OSPREY for a connector to the target; ``bind_owner`` names whose write it
is; ``write_channel_checked`` is the write. The connector asks the write
verdict (tier, switch, session), applies the limits and records the write —
your code does none of that. A refusal arrives as ``ChannelWriteBlockedError``
and the backend passes its sentence on.

A route that needs to know who is calling asks the terminal, never the
request: the terminal drops any ``X-Osprey-Caller`` header a request arrives
with and sets its own, ``agent`` or ``panel``. ``caller_kind(request.headers)``
reads it, and only beside the launch token the terminal adds
(``launch_token: true``, on a login that arms the app; ``launch_token_env()``
names its variable). A request without that token raises ``ValueError``: a
caller header the terminal did not set is never trusted.

The scaffold adds one check of its own, and it is the pattern for every
write you add: the write request carries the readback the operator saw on the
card (``readback``), and the backend re-reads the channel and refuses when the
value has moved. The approval was for a write *over* a known state; a state
that changed since is a new decision.

Reading the machine is ``read_channel`` on the connector. For more than one
channel, ``read_multiple_channels``. The facility description — which elements
exist, their channels, a model's BPMs and correctors — comes from the SDK
(``facility_file``, ``read_facts``, ``model_members``,
``model_response_matrix``), so the app carries no facility knowledge of its
own and moves to another facility with the deployment. The Bluesky bridges the
deployment runs for a target come from ``bridges_for_target(target)``: lane
name to bridge URL, empty when none.

The tool server
===============

``corrector_trim/tools.py`` runs beside the agent as an MCP server. ``register``
adds one function per tool; the function's docstring is the tool description
the agent reads, so write it for the agent: what the tool does, what each
argument means, what it answers. ``serve_tools(register)`` serves them; an
app that holds something for the server's lifetime (a session, say) passes
``on_shutdown=``, which runs once when the server stops, returned or raised.

A tool does not import the backend. It calls the backend's routes through the
terminal (``_panel_call``), at ``/panel/<app>/<route>``, with the headers
``agent_panel_headers`` provides — so the call carries the login's name and
reaches only the routes ``agent_routes`` declares. The terminal is the one door
to the backend for the panel and the agent alike.

A write tool starts with ``check_approved_call``. It runs the write chain for
this one call on the tool's side — is the tool offered on this login, does the
write verdict allow it, is there an approval record for exactly these
arguments, is the approved machine still the session's — and returns the state
values the approval bound (``approved_readback`` for the scaffold). The tool
sends those to the backend, never what the agent typed. The ``approved_*``
arguments in the tool's signature exist for the approval hook to fill in; the
agent leaves them unset, and a value it sends is replaced.

``make_error`` and ``ApprovalRefused`` are how a tool reports: an unreachable
backend becomes the manifest's ``errors.unreachable`` sentence, a refusal the
``errors.refused`` one, each with the detail appended.

``audit_surface()`` is the word the app's audit records file under: its name,
the one its tool server's records already carry. A backend that writes its own
records uses it, so they read under the same word.

The panel
=========

``panel/index.html`` is a plain page. It loads the terminal's
``/design-system/`` ``tokens.css``, ``base.css`` and ``primitives.css`` first,
so it follows the theme, and then its own ``static/panel.css``. Buttons,
toggles, chips and cards use the ``osprey-*`` classes from ``primitives.css``;
``panel.css`` holds only what is the app's own. ``panel/panel.js`` fetches
``api/state`` every two seconds and writes the fields into the page. Addresses are relative
(``api/state``, not ``/api/state``) because the page is served under the
terminal's ``/panel/<id>/`` prefix.

The scaffold's panel only shows; the write is the agent's tool. A panel that
writes — an *Apply…* button — posts to a write route the same way the tool
does, and the terminal refuses the request on a login that does not arm the
app, before it reaches the backend. Show that state rather than hiding the
control: have the backend's state say whether writes are on, and the panel
say *Writes are off*.

``tests/panel.test.js`` is a Vitest test of the page's functions in a DOM;
``osprey app check`` runs it. A JavaScript library the panel needs is not
committed: list it under ``vendor`` with its exact version and SHA-256, and the
build fetches it to the path you name — ``panel/vendor/…``, which
``.gitignore`` already covers.

.. _how-to-apps-writes:

Writes and approvals
====================

A tool under ``tools.approval`` is a write tool. The build lists it in the
tool server's ``writes:``, so every call is judged as a write, in this order:

1. **The tier** — the login's ``control_system.writes``. ``never`` denies the
   tool outright; ``asked`` and ``allowed`` arm the app on that login
   (:doc:`/how-to/web-terminal/multi-user/tiers`).
2. **The write switch** — the operator's position for the session's machine,
   set from the :ref:`control-target chip <web-terminal-session-posture>`.
   *Off* refuses; *asked* shows the approval card and waits for a person;
   *allowed* runs without asking.
3. **The approval entry** — what the card shows and what the call is held to:

   ``prompt``
      The card's text. ``{{ input.<arg> }}`` is a call argument;
      ``{{ state.<key> }}`` a value of the backend's state, fetched from
      ``state_path`` when the card is built. The card also always carries the
      app, the tool, the machine and the switch position.
   ``state_path``
      The backend route whose answer is ``state``. The scaffold uses its
      state route. An app that prepares a write in steps is better served by a
      route made for the card, one that answers only while there is something
      ready to write.
   ``display_lines_key``
      A key of that state holding lines to show under the prompt — one per
      channel for a write of several setpoints — so the operator sees every
      setpoint the write will make.
   ``bind.state_fields``
      Values of the state copied into the call as ``approved_*`` fields: the
      readback the person saw, the plan id, a digest of the setpoints. The
      tool sends these to the backend, and the backend refuses when they no
      longer match what it holds. A key may be a dotted path into nested
      state (``step.digest``); one that names nothing leaves its field
      unbound.
   ``bind.input_digests``
      A digest over named call arguments, so the arguments the agent sends
      after approval must be the ones that were approved. An app recomputes
      a digest only through the SDK's ``canonical_json`` and
      ``input_digest``, the recipe the check itself uses.
   ``require``
      The ``approved_*`` fields a call must carry (``all_of``, or exactly one
      of each ``one_of`` group). A ``one_of`` group holds readback
      (``bind.state_fields``) names only, never a digest. A call that cannot
      be bound is refused, not asked.

4. **The connector** judges the write once more and records it.

Under *allowed* the same binding happens with no person asked: the card's
content is stamped by the switch, and the tool and backend check it the same
way. The approver on the record is ``human``, ``write_switch`` or
``dispatch_policy``.

The approval hook reads the entry from the render and never runs app code;
the tool server re-checks with ``check_approved_call``; the backend checks the
bound values against its state. Three checks, one declaration.

**Dispatch jobs.** An agent run the event dispatcher starts has nobody to show
a card to. A write tool that binds the backend's state never runs in such a
job, and ``deny_in_jobs: true`` under ``tools`` keeps a job from running any of
the app's tools. Set it where every use of the app is an operator's decision.

Logins
======

``profile.yml`` says which apps the deployment runs. Each login's file under
``tiers/`` says which of them its card shows, and its tier says whether it may
write with them:

- a login shows every listed app unless its file says otherwise; under
  ``lists: own`` it shows none it does not name (:doc:`/how-to/build-profiles`);
- a login **arms** an app when it shows it and its tier is ``asked`` or
  ``allowed``. On an armed login the agent gets the write routes and the
  panel's launch token; on any other login the agent gets the read routes
  only and the terminal refuses the panel's write requests;
- a **shared** login (``logbook``, ``knowledge``, ``channel-finder`` in the
  preset) arms nothing, whatever its file says.

A login's ``apps.<name>.access`` says who on that login sees and opens the
panel, in the words a shared roster card uses (``self``, ``roster``, ``user:``,
``domain:``, ``tier:``). The build spells it out to ``user:`` and ``domain:``
members on the panel block, in the spelling the auth sidecar forwards, and
refuses a set no browser caller can match. The terminal compares the forwarded
identity with it: a caller it does not admit does not see the panel in the list
and is answered as for an unpublished one. The agent, on the panel token, is
judged by the login alone, so the login keeps the app's tools and skill. Without
the key every user of the login sees it. It never decides a write.

Hide an app from one login in that login's file:

.. code-block:: yaml

   # tiers/readonly.yml
   apps:
     corrector-trim: {shown: false}

.. _how-to-apps-skills:

Skills and agent parity
=======================

A skill is a ``SKILL.md`` the agent reads to know when and how to use a set of
tools. An app carries its own: list the name under ``skills:`` in ``app.yml``
and write ``skills/<name>/SKILL.md`` in the app's directory. The scaffold
ships one, which ``osprey app new`` renames with the app. The build installs
it into every login that shows the app, so the agent that has the tools also
has the instructions. A listed skill without its file, or a skill name two
apps share, fails the build.

**The parity rule.** Everything the operator sees in the panel, the agent can
read through a tool; everything the operator can press, the agent can do
through a tool or an argument. The panel and the tool server are two front
ends to one backend, so parity needs no second implementation: each panel
element maps to a backend route, and each route the panel uses has a tool.
A route can still tell the two apart: an agent's tool call arrives as
``agent`` and a panel call as ``panel`` (`The backend`_).

.. raw:: html
   :file: ../../_diagrams/app-parity.html

Design the skill from a parity table, one row per panel element:

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - Panel element
     - Tool or argument
     - Route
     - ``tools``
   * - a value it shows
     - the read tool, and the field it answers
     - ``GET``
     - ``read``
   * - a control that writes no channel
     - a tool, or an argument of one
     - any
     - ``read``
   * - a control that writes a channel
     - the write tool
     - ``POST``
     - ``approval``

Then:

1. **One read tool answers the whole screen**, in the panel's terms: the
   values per element, the statuses, and why a control is disabled, in the
   sentence the panel shows. The agent then answers *what does the panel
   say* and *why can't I apply* without guessing.
2. **Every control is a tool or an argument.** Toggles and menus become
   arguments (a checkbox *Include neighbours* is an argument
   ``include_neighbours``);
   buttons become tools. A gesture that only edits input — drag,
   nudge — needs no tool when the agent can pass the resulting values. A
   convenience the operator relies on is an argument, so the agent builds the
   same input the same way (an amplitude and a shape, not the setpoints
   they produce).
3. **Publish the operator's input.** What the operator has set but not yet
   sent is on screen too. The panel posts it to a route that writes no
   channel, and the read tool returns it, so the agent never has to ask what
   the operator is looking at. The route is the operator's: leave it out of
   ``agent_routes``.
4. **Classify by what is written, not by the method.** A route that writes
   no channel — compute a plan, drop it, rebuild a model — backs a ``read``
   tool even when its method is ``POST``; the terminal still hands ``POST``
   routes to the agent only on a login that arms the app. A route that
   writes a channel backs an ``approval`` tool, bound to what the operator
   saw (`Writes and approvals`_).
5. **Write the skill from the table**: which tool for which request, the
   arguments in the facility's terms, how to read each status, what each
   refusal means, and the order — look, plan, show, then ask to write. Name
   each tool's panel equivalent, so the agent can tell the operator where to
   look. Say what the agent does not get (view settings such as zoom) and
   why.

The scaffold's ``SKILL.md`` is the smallest case of this: one table of tool,
panel equivalent and whether it writes, then the order of calls and what a
refusal means. :doc:`/contributing/agent-skills` gives the format.

.. _how-to-apps-subagents:

Subagents
=========

A subagent is an agent the main agent hands a task to. It works in its own
context with its own tool list and returns an answer, so a task that reads a
lot — a search, a comparison over many records — does not fill the main
conversation. An app carries its own: list the name under ``agents:`` in
``app.yml`` and write ``agents/<name>.md`` in the app's directory, one file
per agent. The build installs it into every login that shows the app.

.. code-block:: markdown

   ---
   name: trim-history
   description: Finds and summarises earlier trims of a corrector.
   tools:
     - mcp__corrector-trim__list_trims
     - mcp__corrector-trim__read_trim
   delegates:
     - list_trims
     - read_trim
   delegation: >-
     For questions about earlier trims, delegate to trim-history; do not
     call {delegated_tools} yourself.
   ---

   You answer questions about earlier trims of one corrector. ...

The frontmatter keys, required unless marked optional:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``name``
     - The agent's name; the file name without ``.md``.
   * - ``description``
     - What the agent is for; the main agent routes by it.
   * - ``tools``
     - The tools the agent may call; a tool of the app is
       ``mcp__<app>__<tool>``.
   * - ``delegation``
     - The note the main agent's brief carries for this agent: when to hand
       it a task.
   * - ``delegates``
     - Optional. Tools of the app the main agent is told to leave to this
       agent; their names replace ``{delegated_tools}`` in ``delegation``.
   * - ``summary``, ``disallowedTools``, ``maxTurns``
     - Optional, as in OSPREY's own agent files.

The text below the frontmatter is the agent's instructions, copied as written.

Four keys an agent file may carry elsewhere are refused in an app:

- ``model`` — the deployment sets each agent's model
  (``claude_code.agent_models``);
- ``hooks`` — an app ships no hook code; it declares facts in ``app.yml`` and
  OSPREY's hooks act on them;
- ``mcpServers`` — an app's tools are its ``tools:`` server;
- ``permissionMode`` — the login's tier and the write switch decide.

A listed agent without its file, an agent name two apps share, and a name of
one of OSPREY's own agents each fail the build.

An app needs neither a skill nor a subagent. Use a skill when the main agent
should do the task itself and needs to know how; use a subagent when the task
is better done out of the main conversation; an app may carry both.

Dependencies
============

``pyproject.toml`` is the app's package. The scaffold lists no dependencies:
FastAPI, uvicorn, httpx and the connectors are already in the image. Add what
your app needs under ``dependencies`` — ``numpy``, for instance — and the
build installs it with the wheel. A package that would change a version the
framework pins is refused by ``osprey app check``.

The panel's files and the skill are included in the wheel by the
``force-include`` block, so the installed package carries them. When you
delete ``panel/`` or ``skills/``, delete its line there too.

Manifest keys
=============

Every level refuses unknown keys. Defaults in brackets.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Key
     - Meaning
   * - ``name``
     - The app's name, ``[a-z0-9-]+``; also its directory, its panel id and
       its tool server's name.
   * - ``summary``
     - One sentence about the app.
   * - ``containers``
     - Optional. Required by ``panel`` and by any ``tools.approval`` entry.
   * - ``containers.<key>``
     - One container; the key is its service name. Exactly one of
       ``runtime: osprey`` (the app's package on the project image) or
       ``image:`` (any image, with its compose fragment under
       ``compose/<key>/``).
   * - ``containers.<key>.command``
     - What a ``runtime: osprey`` container runs.
   * - ``containers.<key>.health``
     - The health route, polled by the terminal.
   * - ``containers.<key>.drives_target`` [false]
     - The container reaches the control system.
   * - ``containers.<key>.control_identity`` [false]
     - The container's process identity names the person behind the write
       to the control system (the Channel Access user), instead of the
       shared ``osprey`` account.
   * - ``containers.<key>.state``
     - A directory kept across restarts, relative to the deployment's state
       zone.
   * - ``containers.<key>.env``
     - Host environment variable names passed into the container.
   * - ``containers.<key>.port``
     - A fixed port; otherwise the build assigns one.
   * - ``panel.label``
     - The sidebar row's text.
   * - ``panel.container``
     - The container that serves the panel.
   * - ``panel.entry`` [``/``]
     - The route the panel opens at.
   * - ``panel.launch_token`` [false]
     - The write routes demand a token the terminal adds on an armed login.
   * - ``panel.read_timeout_s``
     - Seconds a proxied request may take; for a route that performs a long
       action, such as a write of many setpoints with read-back.
   * - ``panel.agent_routes``
     - ``[METHOD, /path]`` pairs the agent's tools may call; ``GET``,
       ``HEAD`` and ``OPTIONS`` read, every other method writes.
   * - ``panel.roster_grant.single_user_alias`` [false]
     - The panel works for the alias login of a single-user deployment,
       which has no roster.
   * - ``tools.module``
     - The Python module run as the tool server.
   * - ``tools.read``
     - Tools that only read.
   * - ``tools.approval``
     - Write tools, each with its approval entry (`Writes and approvals`_).
   * - ``tools.deny_in_jobs`` [false]
     - No dispatch job runs any of the app's tools.
   * - ``tools.shown_when`` [``always``]
     - ``panel``: the tool server is present only where the login shows the
       panel.
   * - ``tools.errors.unreachable``, ``tools.errors.refused``
     - The sentences the agent sees for a backend that does not answer, and
       for a refused call.
   * - ``vendor``
     - Files fetched into the app at build: ``{url, into, sha256}`` or
       ``{package, version, file, into, sha256}`` (an npm package through
       the CDN).
   * - ``skills``
     - The skills the app carries, each ``skills/<name>/SKILL.md``.
   * - ``agents``
     - The subagents the app carries, each ``agents/<name>.md``
       (`Subagents`_).

The command reference is :ref:`osprey app <cli-osprey-app>`; the SDK's names
are listed in ``osprey.apps.sdk``.
