.. _how-to-microsoft-teams:

===============
Microsoft Teams
===============

How to let your team ask the OSPREY agent questions from a Microsoft Teams
channel or chat, and get answers and plots back in the same conversation.

.. dropdown:: Before you start
   :color: info
   :icon: checklist

   - A project whose build profile has a ``dispatch:`` block — see
     :doc:`../event-dispatch`.
   - Docker or Podman, for the container path.
   - An Azure subscription in which you can create an app registration, a bot,
     a Service Bus namespace and a Function App.
   - Permission to add an app to your Teams tenant, or a Teams administrator who
     will do it for you.
   - The `Azure CLI <https://learn.microsoft.com/cli/azure/>`__ (``az``) and the
     `Azure Functions Core Tools
     <https://learn.microsoft.com/azure/azure-functions/functions-run-local>`__
     (``func``), signed in to that subscription.

Overview
========

The bridge turns a Teams conversation into a way of talking to the agent.
Someone mentions the bot in a channel or writes to it in a chat, the question is
handed to the :doc:`event dispatch pipeline <../event-dispatch>`, and the answer
is posted back in the same conversation.

Teams differs from the other two platforms in one way that shapes the setup. A
Teams bot receives messages **only** as an HTTPS POST to an address reachable
from the public internet — there is nothing for the bridge to read from. So the
deployment has two pieces rather than one:

- A **relay**: a small Azure Function that is the bot's public address. It
  checks that each incoming message really came from Microsoft, and puts it on a
  Service Bus queue. It is four files that ship with Osprey; you publish them,
  you do not write them.
- The **bridge** itself, running in your stack beside the dispatcher. It pulls
  that queue, and posts replies back through the bot connector.

The bridge is therefore still a **reader of a queue, not a server**: it opens no
port and nothing has to be able to reach it, which is what lets it run on a
control-room network with no inbound path from the internet. The relay is the
only public surface, and the only thing it can do is put a validated message on
your queue.

Like every bridge, it :ref:`remembers each question and conversation
<bridge-memory>`. What is particular to Teams is that there is no reading
position to keep: a message the bridge has not finished with stays on the queue,
so anything sent while the bridge was down is waiting when it comes back.

Each question gets an acknowledgement as soon as it is picked up and the answer
when the run finishes, each as its own message; any plots follow as further
messages. Nothing is edited after it is posted. In a channel all of them are
threaded under the question itself; in a one-to-one or group chat, where Teams
has no threads, the acknowledgement and the answer open with a quote of the
question's first line so it is clear which message is being answered.

.. note::

   **If your tenant is GCC High**, the differences are two configuration values,
   not a different deployment. Set ``TEAMS_CLOUD=gcchigh`` on the bridge and the
   same setting on the relay; the sovereign-cloud login host, token scope,
   signing-key metadata address and issuer are all selected from that one value.
   Point the CLI at the right cloud first with ``az cloud set --name
   AzureUSGovernment``, and read ``azurewebsites.us`` wherever the commands below
   say ``azurewebsites.net``. Everything else on this page is the same.

Enable It in a Profile
======================

Add a ``teams_bridge:`` block to your build profile. Two settings: which
dispatcher trigger the bridge fires — that trigger decides what the agent is
allowed to do with a chat question — and whether the agent may @mention people.
The block is only meaningful next to a ``dispatch:`` block:

.. code-block:: yaml

   teams_bridge:
     trigger: teams-question        # default; must exist in your triggers file
     mentions: true            # default; false posts @mentions as plain text

   dispatch:
     triggers: my_triggers.yml      # the file that trigger must be declared in
     worker_count: 1

   env:
     required:
       - TEAMS_APP_ID
       - TEAMS_APP_SECRET
       - TEAMS_TENANT_ID
       - TEAMS_SERVICEBUS_CONNECTION_STRING
       - TEAMS_SERVICEBUS_QUEUE

.. note::

   The bridge and the dispatch pair must sit on the same network. Workers go on
   the host's network whenever the agent has to reach a control system or a
   co-deployed bridge at a loopback address (:ref:`deployment-network-attachment`),
   and this bridge then has to move with them: add
   ``services.teams_bridge.network: host`` to your profile's ``config:`` block.
   ``osprey build`` refuses a split pair and names the key to change.

The bot's credentials and the queue it reads are **not** profile settings. Which
tenant this deployment authenticates in and which queue it drains differ per
deployment — and the client secret and the connection string are secrets that
must never be baked into a build. Listing them under ``env.required`` documents
them in ``.env.example``; fill the values into the **profile's** ``.env``, which
is where a secret survives a rebuild. The build derives the project's ``.env``
from it (created mode ``0600``, readable only by you).

Two mistakes are caught at **build** time rather than at runtime: declaring the
bridge without a ``dispatch:`` block, and naming a trigger your triggers file
does not declare. Both fail the build with a message naming the problem.

Runtime settings
----------------

These are the ones you create and set yourself.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Meaning
   * - ``TEAMS_APP_ID``
     - The bot's app-registration (client) id, a GUID. It is also what an
       @mention is matched against, so a wrong value makes the bridge ignore
       every channel message instead of failing loudly — worth double-checking.
       The relay needs the same value.
   * - ``TEAMS_APP_SECRET``
     - The client secret of that app registration, which the bridge exchanges for
       a token to post with. A secret: it belongs in the ``.env`` and nowhere
       else.
   * - ``TEAMS_TENANT_ID``
     - The directory (tenant) id the app registration lives in. Single-tenant
       bots have no default for this.
   * - ``TEAMS_SERVICEBUS_CONNECTION_STRING``
     - Connection string for the queue the relay writes to, taken from the
       **Listen-only** policy (``bridge-listen`` below). The bridge never sends
       to the queue, so it never holds a key that could.
   * - ``TEAMS_SERVICEBUS_QUEUE``
     - The name of that queue. Read the warning below before pointing a second
       deployment at it.
   * - ``TEAMS_CLOUD``
     - Optional. Leave it unset for an ordinary commercial tenant. Set it to
       ``gcchigh`` for a GCC High tenant; those are the only two values, and any
       other one stops the bridge at startup rather than failing later as an
       unexplained authentication error.

The bridge refuses to start if any of the required five is missing, naming all
the missing variables at once.

.. warning::

   **One bridge per queue.** Deploy exactly one bridge against a given queue.
   Service Bus hands each message to only *one* of a queue's readers, so a second
   deployment pointed at the same queue — a staging stack, a container someone
   forgot to remove, another facility reusing the name — does not get its own
   copy of every message. It **silently splits** them: each bridge answers only
   the messages it happened to receive, the other questions look to your team
   like they were ignored, and nothing anywhere logs an error. Give every
   deployment its own queue, and its own relay writing to it.

.. dropdown:: Settings that are filled in for you, or safe to leave alone
   :icon: gear

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Variable
        - Meaning
      * - ``DISPATCH_TRIGGER``
        - The trigger to fire. Comes from the profile block.
      * - ``TEAMS_MENTIONS``
        - Whether @mentions are on. Comes from the profile block.
      * - ``EVENT_DISPATCHER_TOKEN``, ``DISPATCH_WORKER_TOKEN``
        - The two shared secrets the bridge needs to reach the dispatcher and
          the worker, generated for you when unset — see
          :ref:`Authentication <event-dispatch-auth>`.
      * - ``DISPATCHER_URL``, ``WORKER_URL``
        - Where the dispatcher and worker are — the bridge collects answers and
          files from the worker directly. Filled in for you when they run in the
          same stack; when they run elsewhere you must set them, and the bridge
          refuses to start without them.
      * - ``DISPATCH_TIMEOUT_SEC``
        - How long the worker may spend on one run. Comes from your project
          configuration, so raising it raises it for both halves at once.
      * - ``POLL_BUDGET``
        - How long the bridge waits for an answer before giving up on it.
          Defaults to 30 seconds more than the worker's own limit, and may never
          be less than that limit — the bridge refuses to start if it is.
      * - ``POLL_INTERVAL``
        - Seconds between checks on an answer in progress (default 2).
      * - ``DRAIN_INTERVAL``
        - Seconds between sweeps of the queue of questions that could not be
          handed off yet (default 60).
      * - ``RETRY_MIN_AGE``
        - How long a failed hand-off is held before it is retried, so a brief
          outage has time to clear (default 20 minutes).
      * - ``RETRY_GIVE_UP``
        - Age at which a question that still cannot be handed off is abandoned
          (default 48 hours).
      * - ``RETRY_LIFETIME_CAP``
        - Hard ceiling on how long anything may sit in that queue, whatever its
          state (default 7 days).
      * - ``BRIDGE_TRUST_ENV``
        - Set to ``1`` only if this host's outbound calls must go through your
          site's web proxy. Off by default, so a proxy inherited from a shell or
          a CI runner cannot quietly place itself in front of Microsoft.
      * - ``GITLAB_URL``, ``GITLAB_PROJECT``, ``GITLAB_ISSUES_TOKEN``
        - Where to file an issue when a question is finally given up on. Leave
          unset if you have no such host: nothing is filed and nothing is
          checked.
      * - ``APP_VERSION_DISPLAY``
        - A release label shown with each acknowledgement, so a conversation
          shows which version answered: the acknowledgement reads ``On it —
          working on this now.  (OSPREY 2026.9.0)`` with this label in the
          parenthesis. When it is unset the bridge uses the version of the
          Osprey package it is running, and drops the parenthesis only if even
          that is unavailable.
      * - ``DEDUP_PATH``, ``HISTORY_PATH``
        - Where the bridge keeps what it remembers. Both default to files under
          ``/data``, its own volume; change them only if you deliberately
          relocate that state.
      * - ``TZ``
        - Timezone, taken from your project configuration so timestamps match
          the rest of the stack.

Bring It Up
===========

The commands below use one set of names throughout — resource group
``osprey-teams``, namespace ``osprey-teams-bus``, queue ``osprey-questions``,
Function App ``osprey-teams-relay`` — so you can paste them in order and change
the names once, at the top. Namespace, Function App and storage-account names are
part of a public address, so they must be unique across Azure; pick something
with your facility in it if these are taken.

**1. Create a resource group to hold everything.**

.. code-block:: bash

   az login
   az group create --name osprey-teams --location westus2

On a GCC High tenant, run ``az cloud set --name AzureUSGovernment`` before
``az login``, and use one of that cloud's regions (for example ``usgovvirginia``)
throughout.

**2. Give this deployment its own queue.** One namespace, and one queue that
nothing else reads — see the warning above for what a shared queue does.

.. code-block:: bash

   az servicebus namespace create \
     --resource-group osprey-teams --name osprey-teams-bus --sku Standard
   az servicebus queue create \
     --resource-group osprey-teams --namespace-name osprey-teams-bus \
     --name osprey-questions

**3. Create one key for writing and one for reading.** The relay may only send,
the bridge may only listen; neither half holds the other's rights.

.. code-block:: bash

   az servicebus queue authorization-rule create \
     --resource-group osprey-teams --namespace-name osprey-teams-bus \
     --queue-name osprey-questions --name relay-send --rights Send
   az servicebus queue authorization-rule create \
     --resource-group osprey-teams --namespace-name osprey-teams-bus \
     --queue-name osprey-questions --name bridge-listen --rights Listen

Then read the two connection strings out. Keep them somewhere safe for steps 5
and 8; they are credentials.

.. code-block:: bash

   az servicebus queue authorization-rule keys list \
     --resource-group osprey-teams --namespace-name osprey-teams-bus \
     --queue-name osprey-questions --name relay-send \
     --query primaryConnectionString --output tsv
   az servicebus queue authorization-rule keys list \
     --resource-group osprey-teams --namespace-name osprey-teams-bus \
     --queue-name osprey-questions --name bridge-listen \
     --query primaryConnectionString --output tsv

**4. Register the bot and the identity it speaks as.** The registration's name
is internal; the name your team sees replying is the one you give the Teams app
in step 7. Single-tenant keeps the bot usable only inside your own directory.

.. code-block:: bash

   az ad app create --display-name "OSPREY agent" --sign-in-audience AzureADMyOrg \
     --query appId --output tsv
   az account show --query tenantId --output tsv

Note both values — they are ``TEAMS_APP_ID`` and ``TEAMS_TENANT_ID``. Then create
a client secret, which is shown once and never again:

.. code-block:: bash

   az ad app credential reset --id <app id> --display-name teams-bridge \
     --query password --output tsv

Now create the bot resource. Its messaging endpoint is the relay's address,
which you can write down before the relay exists — it is the Function App name
you are going to use in step 5:

.. code-block:: bash

   az bot create \
     --resource-group osprey-teams --name osprey-agent-bot --sku F0 \
     --app-type SingleTenant --appid <app id> --tenant-id <tenant id> \
     --endpoint https://osprey-teams-relay.azurewebsites.net/api/messages
   az bot msteams create --resource-group osprey-teams --name osprey-agent-bot

The second command turns on the Teams channel; without it the bot exists but
Teams will not talk to it.

**5. Create the relay's Function App.** Flex Consumption with one always-ready
instance: Teams expects an answer from the endpoint within 15 seconds, and an
app that has scaled to nothing spends most of that budget starting up.

.. code-block:: bash

   az storage account create \
     --resource-group osprey-teams --name ospreyteamsrelay \
     --location westus2 --sku Standard_LRS
   az functionapp create \
     --resource-group osprey-teams --name osprey-teams-relay \
     --storage-account ospreyteamsrelay \
     --flexconsumption-location westus2 \
     --runtime python --runtime-version 3.11
   az functionapp scale config always-ready set \
     --resource-group osprey-teams --name osprey-teams-relay \
     --settings http=1

Then give it its settings. These are the relay's own names and are **not** the
bridge's: in particular the connection string here is the ``relay-send`` one, and
the setting is called ``SERVICEBUS_CONNECTION``, because the Functions host reads
it directly so the send key never passes through the relay's own code.

.. code-block:: bash

   az functionapp config appsettings set \
     --resource-group osprey-teams --name osprey-teams-relay \
     --settings TEAMS_APP_ID=<app id> \
                TEAMS_SERVICEBUS_QUEUE=osprey-questions \
                SERVICEBUS_CONNECTION="<the relay-send connection string>"

On a GCC High tenant add ``TEAMS_CLOUD=gcchigh`` to that same command. Leave it
unset anywhere else.

**6. Publish the relay.** ``osprey build`` copies it into your deployment
repository, under ``build/services/teams_bridge/relay/``. Publish from that
directory:

.. code-block:: bash

   cd build/services/teams_bridge/relay
   func azure functionapp publish osprey-teams-relay --python

The endpoint is then live at
``https://osprey-teams-relay.azurewebsites.net/api/messages`` — the address you
gave the bot in step 4. It accepts anonymous requests on purpose: the proof that
a message is genuine is the signed token Microsoft sends with it, which the relay
checks against the bot's app id before anything reaches your queue. A request
that fails that check is refused and never enqueued.

**7. Install the app in Teams.** Build a Teams app package — a ``manifest.json``
plus its two icons, zipped — whose ``bots`` entry carries the app id from step 4:

.. code-block:: json

   {
     "bots": [
       {
         "botId": "<app id>",
         "scopes": ["team", "groupChat", "personal"],
         "supportsFiles": false,
         "isNotificationOnly": false
       }
     ]
   }

Upload it in the Teams admin center, or in Teams itself if your tenant allows
custom apps, then add the app to each team or chat it should serve. It only sees
conversations it has been added to.

**8. Fill in the environment file.** Set the values in the **profile's** ``.env``
(the build derives the project's ``.env`` from it):

.. code-block:: bash

   TEAMS_APP_ID=00000000-0000-0000-0000-000000000000
   TEAMS_APP_SECRET=the-secret-from-step-4
   TEAMS_TENANT_ID=11111111-1111-1111-1111-111111111111
   TEAMS_SERVICEBUS_CONNECTION_STRING=<the bridge-listen connection string>
   TEAMS_SERVICEBUS_QUEUE=osprey-questions
   # TEAMS_CLOUD=gcchigh                  # only on a GCC High tenant

**9. Bring the stack up.** The bridge is registered in ``deployed_services``, so
it starts with everything else:

.. code-block:: bash

   osprey up        # add --dev to bake in a local osprey checkout

Then mention the bot in a channel it has been added to and ask it something. If
nothing happens, work outwards from the bridge: a missing credential stops it at
startup with the missing variable named. A wrong app id is the quieter failure —
the bridge runs happily and ignores every channel message, because nothing it
sees looks like a mention of itself. If the bridge looks idle, check the Function
App's logs: a message refused there never reaches the queue, and a relay whose
settings name the wrong queue writes where nobody is reading.

.. important::

   The bridge keeps what it remembers — which questions it has answered and the
   recent conversation — in a named volume mounted at ``/data``. Do not remove
   that volume. Without it, a restart in the middle of a question can answer it
   twice or drop it, and conversations lose their thread of context.

.. _what-is-shared-teams:

Who Can Ask, and What Is Shared
===============================

Who may reach the agent, and what the trigger lets it do, is the same for every
bridge — see :ref:`bridge-access`. What is particular to Microsoft Teams:

**The conversations are the ones the app was added to.** Teams — not Osprey —
decides who is in those teams and chats, so add the app as deliberately as you
would choose who gets an account.

**In a channel or a group chat, only a real @mention of the bot is answered.**
The bridge matches the mention Teams itself records, not the message text, so
writing the bot's name in passing does not trigger it. A one-to-one chat has
nobody else to address, so every message there is a question.

**Other people's names survive the question.** When a question mentions a
colleague as well as the bot, the bot's own mention is removed as addressing and
every other mention becomes that person's display name, so the agent sees who was
named.

**Who is in the conversation, and @mentions.** Each question reaches the agent
with who asked it and who is in the conversation; for a channel that is the
channel's members. The bridge lists them with the bot's own sign-in, so there is
no extra permission to grant. A member's name is the one Teams lists. If Teams
lists none, it is the name the bridge has seen that person use or be @mentioned
under; otherwise the member is listed without a name, and the agent is told not
to guess. The agent may @mention a member of the same conversation only when
someone in it asks the agent to pass something on or to notify someone. It
never does so on its own. Anyone not in the conversation is written as plain
text. Set ``mentions: false`` in the profile block to have every mention posted
as plain text instead.

**Plots come back inside the conversation.** Each plot arrives as its own
message directly after the answer, one image per message, attached inline rather
than linked from anywhere else, so it is visible exactly to the people who can
see the conversation and to nobody else. This is the opposite
of the :doc:`Google Chat <google-chat>` deployment, where files are published as
public links.

.. note::

   **What the agent cannot exchange in Teams.**

   - *Only PNG images come back.* Each one is fitted into a 1024×1024 box and
     re-encoded before posting; anything still over 1 MB after that is dropped,
     and one last message names what was dropped so the answer never quietly
     omits a plot. Documents — PDFs, CSVs, tables saved to file — are not delivered at
     all in a Teams deployment.
   - *Images need Pillow.* It is installed with the ``teams`` extra, which the
     shipped image uses. Without it the agent still answers in full text and
     every image is named in the same note rather than attached.
   - *Files attached to a question are ignored.* Teams attachments are not
     downloaded, so a question that says "look at this log" and attaches one gets
     an answer written without it. Paste the relevant part into the message
     instead.
