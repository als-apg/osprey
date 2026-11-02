=====================
Health and Monitoring
=====================

Two questions, two tools. *Is it up?* is answered by the health suite: a run of
diagnostics over configuration, the Python environment, container
infrastructure, the framework services you enabled, and any probes a facility
declares, read in the browser's SYSTEM panel or parsed from ``osprey health
--json`` by a CI job. *What is it doing?* is answered by the agent record:
what the agent was asked, what it answered and what it did, kept by the
``agent-record`` service every deployment runs.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Configure Health Checks
      :link: configure-health-checks
      :link-type: doc
      :shadow: md

      What the built-in diagnostics cover, and how a ``health:`` block in
      ``config.yml`` -- set through the profile's ``config:`` block -- adds a
      facility's own probes and tunes the suite's timing.

   .. grid-item-card:: Keep the Agent Record
      :link: agent-record
      :link-type: doc
      :shadow: md

      Where the record of what the agent was asked and did lives, how long it
      keeps an entry, how to check on it, and the daily append-only archive
      that keeps it for good.

   .. grid-item-card:: Read the Record
      :link: read-the-record
      :link-type: doc
      :shadow: md

      Name the people who may read the record, hand over their keys, and know
      what each one can see and what the record can vouch for.

.. toctree::
   :hidden:

   configure-health-checks
   agent-record
   read-the-record
