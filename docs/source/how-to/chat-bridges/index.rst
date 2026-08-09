============
Chat Bridges
============

A chat bridge lets your team ask the Osprey agent questions from a chat room
they already sit in. Someone mentions the agent in the room, and the answer
comes back in the same conversation — including plots and files.

Two chat systems are supported today, **Nextcloud Talk** and **Google Chat**.
They work the same way and are configured the same way. You can add others.

What a Bridge Is Not
====================

A bridge is not a second :doc:`Web Terminal <../web-terminal/index>`. There is
no session to keep open, and nobody is asked to approve anything while an answer
is being worked out.

A chat question is one headless run of the agent, and what the agent may do
during that run is decided in advance — by the dispatcher trigger the bridge
fires, not by the question and not by who asked it. That makes a bridge a good
way to give a whole team read access to the machine, and a poor way to hand out
control of it.

How It Works
============

A bridge sits between a chat room and the :doc:`event dispatch pipeline
<../event-dispatch>`. It never runs the agent itself. It hands the question to
the dispatcher, waits, and posts back what comes out.

.. code-block:: text

      Nextcloud Talk room                Google Chat space
              |                                  |
              | the bridge asks                  | the bridge reads
              | for new messages                 | a message queue
              v                                  v
      +--------------------+         +--------------------+
      |    Talk adapter    |         |    Chat adapter    |
      +--------------------+         +--------------------+
                \                                /
                 \                              /
                  v                            v
           +--------------------------------------+
           |         shared bridge engine         |
           |  remembers each question, keeps the  |
           |  conversation, retries what failed   |
           +--------------------------------------+
                      |                       ^
              question|                       | answer + any files
                      v                       |
           +--------------------------------------+
           |           Event dispatcher           |
           +--------------------------------------+
                             |
                             v
           +--------------------------------------+
           |   Dispatch worker  ->  Osprey agent  |
           +--------------------------------------+

Only the top box changes between chat systems. Each system has its own small
**adapter**, which knows how messages arrive there and how to post a reply
there. Everything below it — remembering questions, keeping the conversation,
handling failures — is shared code that every bridge uses unchanged.

Notice that every arrow leaving a bridge points outward. A bridge asks the chat
system for new messages rather than being called by it, so it opens no network
port, needs no public address, and nothing has to be able to reach it.

.. _bridge-memory:

What the Bridge Remembers
-------------------------

The bridge is the only part of the system that sees a whole conversation, so it
is the part that remembers. It keeps three things on its own disk volume: which
questions it has already answered, the recent exchanges in each conversation,
and (for Nextcloud Talk) how far it has read in each room.

That is what makes follow-up questions work — "now plot that over 24 hours"
arrives at the agent with the previous exchange attached. It is also what makes
a restart safe: a question interrupted halfway through is picked up again rather
than answered twice or lost.

.. _bridge-access:

Who Can Ask
===========

**Membership in the room is the access gate.** Anyone who can post in a room the
agent has been added to can ask it questions. Your chat system, not Osprey,
decides who those people are — so add the agent to rooms as deliberately as you
would decide who gets an account.

In a group room, only messages that mention the agent are answered; everything
else is ignored, so it can sit in a busy room quietly. In a one-to-one
conversation there is nobody else to address, so every message counts.

**What the agent may do is set by the trigger.** The trigger's list of permitted
tools is mounted read-only, so a question arriving through chat cannot widen what
the agent is allowed to do, however it is phrased.

Choosing Your Platform
======================

Both bridges are equally capable. The differences that matter when you pick one:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Nextcloud Talk
     - Google Chat
   * - The agent speaks as
     - A normal Nextcloud user account you create
     - A Chat app backed by a service account
   * - Messages arrive by
     - The bridge asking Nextcloud for them
     - A Google Cloud message queue
   * - Plots and files
     - Shared with the room, visible to its members only
     - **Published as a public link** anyone can open
   * - You need
     - A Nextcloud instance with the Talk app
     - A Google Cloud project

That last row is the one to read twice. Google Chat can only display an image if
Google itself can fetch it, so files are published to a world-readable address
rather than shared privately. If that is not acceptable at your facility, you can
turn files off and still get text answers — the Google Chat page explains how.

Learn More
==========

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Nextcloud Talk
      :link: nextcloud-talk
      :link-type: doc
      :class-header: bg-info text-white
      :shadow: md

      Deploy a bridge into Nextcloud Talk rooms, where files stay private to the
      room.

   .. grid-item-card:: Google Chat
      :link: google-chat
      :link-type: doc
      :class-header: bg-primary text-white
      :shadow: md

      Deploy a bridge into Google Chat spaces, and decide how plots and files
      are shared.

   .. grid-item-card:: Add Your Own
      :link: add-a-channel
      :link-type: doc
      :class-header: bg-success text-white
      :shadow: md

      What it takes to connect Slack, email, or any other service the agent
      should answer from.

.. seealso::

   :doc:`../event-dispatch`
       The dispatcher and worker every bridge hands its questions to, and how to
       write the trigger it fires.

.. toctree::
   :maxdepth: 2
   :hidden:

   nextcloud-talk
   google-chat
   add-a-channel
