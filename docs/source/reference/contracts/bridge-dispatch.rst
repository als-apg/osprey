.. _reference-bridge-dispatch:

==============================
The Bridge Dispatch Payload
==============================

A chat bridge never runs the agent. It fires one dispatcher webhook per
question and posts back what comes out, so everything the agent learns about
the conversation — the previous exchanges, the message being replied to, the
files attached to either — arrives as fields of that one JSON body.

This page is that body. Read it if you are writing an adapter for a chat system
OSPREY does not ship (:doc:`/contributing/extending-osprey` covers the seam
itself), or if you are authoring a trigger prompt and want to know what is
actually in front of the agent.

The request
===========

The bridge ``POST``\ s to ``<dispatcher>/webhook/<trigger>`` with a Bearer
token. The body is the question plus whatever context the engine assembled:

.. code-block:: json

   {
     "question": "now plot that over 24 hours",
     "conversation_so_far": [{"question": "...", "answer": "...", "ts": 1757000000.0,
                              "run_id": "run-...", "artifacts": []}],
     "reply_to": {"sender": "Alice", "text": "the vacuum trace from this morning"},
     "input_files": [{"filename": "trace.png", "mime": "image/png",
                      "content_b64": "...", "ingest": true}],
     "skipped_attachments": [{"filename": "scan.h5", "reason": "..."}]
   }

Only ``question`` is always present. A text-only message in a fresh
conversation sends that field and nothing else — the engine omits an empty
context rather than sending empty containers, so such a dispatch is
byte-identical to one made before any of this existed.

The whole body becomes the agent's context, exactly as it does for a webhook
fired by hand. A trigger's prompt does not have to name these fields for the
agent to see them.

``conversation_so_far``
=======================

The recent exchanges in this conversation, **oldest first**. This is what makes
"now plot that over 24 hours" resolve its referent. Each turn:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``question``
     - What was asked.
   * - ``answer``
     - What the agent replied. A turn whose run failed carries a marker here
       instead, so the referent survives even when the answer did not.
   * - ``ts``
     - Unix timestamp of the exchange.
   * - ``run_id``
     - The dispatch run that produced the answer, or ``null``.
   * - ``artifacts``
     - Descriptors for the files that run produced — round-tripped opaquely by
       the bridge, capped per turn, newest kept.

The bridge caps the list by turn count and by total serialized size, dropping
oldest first, so the newest exchanges always survive.

A descriptor may carry a ``note`` reading ``"may have expired"``. That is the
bridge saying it could not bring the artifact's bytes back — swept, deleted, or
too large for the budget below — so the agent should say the file is no longer
available rather than pretend to look at it. Nothing keys on the exact wording;
it is written to be read.

``reply_to``
============

Present only when the message quotes or replies to another one, and only when
the chat system has such a mechanic at all:

.. code-block:: json

   {"sender": "Alice", "text": "the vacuum trace from this morning"}

``text`` is the quoted message, truncated by the adapter to its own limit. Two
further fields appear when the **quoted** message had attachments:
``attachments`` (the filenames delivered from it) and ``skipped_attachments``
(the ones that were not). They live inside ``reply_to`` rather than beside it
precisely so the agent can tell a file quoted from earlier apart from a file
attached to the message in front of it.

``input_files`` and ``skipped_attachments``
===========================================

Files ride the payload as base64, in one list:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``filename``
     - The name the agent sees.
   * - ``mime``
     - The type the adapter fetched it as.
   * - ``content_b64``
     - The bytes, base64-encoded.
   * - ``ingest``
     - ``true`` for a file attached to this message (or quoted from the one it
       replies to); ``false`` for a prior image the bridge replayed.

Order is fixed: this message's own attachments, then the quoted message's, then
any replayed prior images. Replay exists so a follow-up about a plot can look
at the plot again rather than only at its descriptor; replayed bytes are
deduplicated by content against what the message already attached.

Two budgets bound the list, both enforced by the bridge before anything is
sent:

* **Fresh files** — at most 5 files and 10 MiB across ``ingest: true`` entries,
  own attachments first.
* **Replayed images** — at most 3 images and 8 MiB, newest first, and only
  images.

``skipped_attachments`` accounts for everything that did not make it, as
``{"filename": ..., "reason": ...}`` notes the agent can relay. The reasons the
engine itself writes:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Reason
     - When
   * - ``too many attachments (limit 5); this file was not processed``
     - The count budget was already spent.
   * - ``attachments exceed the total size budget; this file was not processed``
     - The byte budget was already spent.
   * - ``couldn't ingest (server too old)``
     - The worker on the other end does not advertise ``input_files``, so the
       bytes were never sent. Nothing is silently dropped: every file gains
       this note instead.

An adapter adds its own reasons for what it could not fetch — a download that
failed, a type the chat system will not hand over — in the same shape.

.. seealso::

   :doc:`/how-to/agent-interfaces/chat-bridges/index`
      What a bridge is, what it remembers, and who can ask it questions.

   :doc:`/how-to/agent-interfaces/event-dispatch`
      The dispatcher and worker on the receiving end of this payload, and how
      to write the trigger it fires.
