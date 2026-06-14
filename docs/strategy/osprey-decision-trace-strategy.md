# Osprey & the Decision-Trace Thesis — strategic design analysis

> **What this is.** A multi-step analysis applying the "decision traces / context
> graph" argument (Gupta & Garg, 2025-12-22 — saved in
> [`decision-traces-and-context-graphs.md`](./decision-traces-and-context-graphs.md))
> to Osprey. The goal is not to adopt the article's market framing wholesale, but
> to extract the *mechanism* it describes and decide, concretely, what Osprey should
> build, keep, or deliberately not build.
>
> **Author of analysis:** Claude (strategy working note). **Date:** 2026-06-14.
> **Status:** discussion draft — decisions are proposals, not commitments.

The workflow runs in six stages:

1. Distill the source thesis to its load-bearing claims.
2. Position Osprey inside the article's framework.
3. Inventory what Osprey already captures vs. discards at decision time.
4. Derive candidate strategic design decisions (options + tradeoffs).
5. Sequence them against Osprey's safety-first constraints.
6. Stress-test: where the analogy breaks and what *not* to do.

---

## Stage 1 — The source thesis, distilled

Strip the enterprise-SaaS packaging and four load-bearing claims remain:

1. **Rules ≠ decision traces.** A rule says what *should* happen in general; a
   decision trace records what happened *in this case* — which inputs were
   gathered, which policy version applied, what exception was invoked, who
   approved, what state was written, and **why**.
2. **The "why" is normally never captured as data.** It dies in side-channels
   (Slack, calls, people's heads). The final record keeps only the outcome
   ("20% discount"), not the justification or the world-state that justified it.
3. **Whoever sits in the execution path at commit time has a structural
   monopoly on capturing it.** Read-path systems (warehouses, ETL) see it too
   late; current-state systems (CRMs) overwrite it. Only the orchestrator
   present *at decision time* can persist it as a first-class record.
4. **Persisted traces compound into a context graph** — entities linked by
   decision events and "why" links — which becomes the authoritative artifact
   for auditing autonomy and turning exceptions into reusable precedent, and the
   substrate that lets autonomy expand safely over time.

The actionable kernel: **treat the reasoning that connects data to action as
data.** Everything else (trillion-dollar markets, incumbent dynamics) is
framing we can discard for our purposes.

---

## Stage 2 — Where Osprey sits in this framework

The article's structural argument lands on Osprey almost verbatim, because Osprey
is, by construction, a *system of agents in the execution path*:

- **It is in the write path at commit time.** Every hardware write passes through
  the in-line safety chain `writes_check → limits → approval` before reaching the
  MCP server (`docs/source/architecture/index.rst`;
  `src/osprey/templates/claude_code/claude/hooks/osprey_approval.py`). This is
  precisely the "commit-time orchestration" position the article says incumbents
  structurally cannot occupy.
- **It does cross-system synthesis.** A single operator request fans out across
  Channel Finder (PV address resolution), machine-state reads, the archiver
  (historical context), and the Python executor before an action is proposed.
  That is the accelerator analog of "ARR in Salesforce + escalations in Zendesk +
  Slack churn thread."
- **It already has a record/retrieval layer.** ARIEL (Agentic Retrieval Interface
  for Electronic Logbooks) ingests logbook entries and exposes semantic / keyword
  / SQL / browse search (`src/osprey/mcp_server/ariel/`). This is Osprey's
  existing "system of record" candidate.
- **Operations is a glue function.** The article's strongest "new system of
  record" signal is *organizations that exist at the intersection of systems*
  (RevOps, DevOps). Accelerator **operations** is exactly this: it sits between
  accelerator physics, controls/EPICS, diagnostics, RF, vacuum, and safety
  systems precisely because no single subsystem owns the cross-cutting
  operational decision. The control-room operator *is* the human carrying context
  the software never captured.

So Osprey is not "an agent bolted onto a system of record." It is already
standing in the one position the article says is defensible. The question is
whether it persists what flows through that position.

---

## Stage 3 — What Osprey captures vs. discards at decision time

This is the crux, and the finding is sharper than expected: **Osprey is
architecturally positioned to capture decision traces and currently, by design,
throws the "why" away.**

| Decision-trace element (article) | Osprey today | Verdict |
|---|---|---|
| Inputs gathered across systems | Channel Finder lookups, reads, archiver pulls happen, and land in the session transcript / audit trail (`gather_context`, `TranscriptReader`) | **Transient.** Captured in the run transcript, not as a structured, queryable artifact tied to the action. |
| Policy/rule that applied | Limits DB (min/max/step), approval policy per tool (`osprey_approval.py`), `writes_enabled` kill switch | **Present but unversioned & unlinked.** The limit that gated a write is not snapshotted with the decision. |
| Exception / override | Approval is `ask/allow/deny`; hook logs `status` + a short `detail` string (`log_hook(...)`) | **Outcome only.** No structured "exception granted, reason, scope." |
| Who approved + why | Human approves in the web terminal; the rationale is spoken/typed in chat and not bound to the write | **The "why" is discarded.** This is the gap. |
| World-state at decision time | Reads are live; no snapshot of machine state + archiver context is frozen at commit | **Not replayable.** Mirrors the article's "you can't replay the state of the world at decision time." |
| Final state written | Channel write result is confirmed and verifiable | **Captured** (this is the one thing that *is* durable — same as the CRM's "20% discount"). |
| The narrative record | ARIEL logbook entries, composed from transcript + artifacts | **Deliberately strips intent** (see below). |

The most striking confirmation is in the logbook composer. `BASE_PREAMBLE` and the
`SYSTEM_PROMPT` in `src/osprey/interfaces/artifacts/logbook.py` instruct the model,
in strict terms:

> "State ONLY verifiable facts… **Never infer operator intent or motivation.**
> … Use the conversation log to understand what was asked for, **not to speculate
> about why.**"

This is *correct and necessary* — an LLM must not fabricate the "why" in a
safety-critical logbook. But it means Osprey's durable record is, by policy,
exactly the article's impoverished artifact: **what happened, never why.** The
reasoning that connected data to action is treated as untrustworthy narration to
be suppressed, rather than as data to be captured from an authoritative source
(the human in the loop).

That reframes the whole opportunity. The gap is not "Osprey can't see the why."
It's "Osprey sees it, then deliberately drops it because the only capture path it
has (LLM inference) is unsafe." The fix is a *trusted* capture path:
human-supplied rationale at the approval gate, recorded as fact rather than
inferred as narration.

---

## Stage 4 — Candidate strategic design decisions

Each is framed as a decision with options and a recommendation. Numbered D1–D8.

### D1 — Make the decision trace a first-class persisted artifact

**Decision:** Should an agent run that culminates in a gated action (write,
execute-with-writes, setup_patch) emit a structured, persisted **Decision
Record**, distinct from the transcript and the logbook entry?

- *Option A — status quo.* Keep transcript + hook logs + optional logbook entry.
  Cheap; but the trace stays transient and unqueryable.
- *Option B — Decision Record as a new artifact type.* On every gated action,
  persist `{run_id, operator, request, resolved_channels, policy_snapshot,
  inputs_referenced, proposed_action, approval{decision, approver, rationale},
  state_before, result, timestamp}` into the artifact store
  (`src/osprey/stores/artifact_store.py` already gives typed, queryable storage).

**Recommendation: B.** This is the single highest-leverage move and it reuses
existing infrastructure (artifact store + hook chain). The Decision Record is the
atom of everything downstream (D3 graph, D5 precedent). Without it, the rest is
not buildable.

### D2 — Capture the human "why" at the approval gate (the trusted-rationale path)

**Decision:** Where does the "why" come from, given the logbook's (correct)
prohibition on LLM-inferred intent?

- *Option A — keep suppressing it.* Safe, but perpetuates the gap.
- *Option B — capture human rationale at approval time.* When the approval hook
  returns `ask`, optionally require/offer a free-text + structured reason from the
  approver ("service-impact exception," "physics-study override of nominal
  limit," "matches precedent <id>"). This is *fact* (the human said it), not
  inference, so it does not violate the logbook's truth constraint.

**Recommendation: B, additive and optional-by-policy.** Reconciles the tension
cleanly: the LLM still may not invent intent; the human supplies it; the system
records it verbatim and attributes it. Lightweight: extend the approval
hook/response and the web-terminal approval UI to carry a `rationale` field into
the Decision Record (D1). For routine `skip`/auto-approved writes, no rationale is
needed — which naturally matches the article's "routine workflows don't need
decision lineage."

### D3 — Build the context graph on top of entities Osprey already owns

**Decision:** Should Osprey stitch Decision Records into a queryable graph, and on
what substrate?

The article's entities map directly to accelerator-native ones:

| Article | Osprey analog |
|---|---|
| accounts, tickets, incidents | channels/PVs, subsystems, machine-state snapshots, fault/downtime events |
| policies, approvers | limits DB + approval policy (versioned), operators/physicists-on-shift |
| agent runs, decision events | Osprey run_id, gated actions |
| "why" links | human rationale + precedent references |

- *Option A — defer.* Decision Records sit as flat artifacts; "graph" is just
  search.
- *Option B — explicit graph layer* linking records ↔ channels ↔ policies ↔
  approvers ↔ ARIEL entries ↔ fault events, queryable via ARIEL's existing
  SQL/semantic/keyword surfaces.

**Recommendation: B, but staged after D1/D2.** Don't build a graph database for
its own sake. Start with foreign-key-style links in Decision Records and lean on
ARIEL (`mcp_server/ariel/tools/{sql_query,semantic_search,browse}`) as the query
plane. Promote to a dedicated graph representation only when precedent retrieval
(D5) demonstrably needs multi-hop traversal.

### D4 — Snapshot world-state at decision time (replayability)

**Decision:** Should Osprey freeze a machine-state + archiver-context snapshot at
commit time, so a decision is replayable?

This is the article's "you can't replay the state of the world at decision time,"
and the accelerator case is *stronger* than the SaaS case: machine state is
high-dimensional, fast-moving, and a setpoint that was safe at injection may be
unsafe at a different fill. The justification for a write is often "given that the
machine looked like *this*."

- *Option A — none.* Rely on the archiver to reconstruct post-hoc (lossy; sample
  rates, which PVs, alignment all uncertain).
- *Option B — bounded snapshot at commit:* the specific channels read during the
  run + their values + the active limits + relevant archiver window references,
  frozen into the Decision Record.

**Recommendation: B, bounded.** Snapshot only what the run actually touched (the
inputs already referenced), not the whole machine — keeps it cheap and exactly
scoped to what justified the action. This is what makes "audit the decision" and
"learn from it" real rather than aspirational.

### D5 — Surface precedent at approval time (the compounding loop)

**Decision:** Should the agent retrieve and present prior similar Decision Records
when proposing a gated action?

- *Option A — no.* Each decision is context-free.
- *Option B — precedent retrieval:* before/at the approval prompt, query the
  graph for "prior gated actions on this channel/subsystem under similar machine
  state," and present `{what was done, who approved, rationale, outcome}` to the
  approver.

**Recommendation: B — this is where the value compounds**, and it is the article's
core feedback loop ("captured traces become searchable precedent; every decision
adds a trace"). It also *improves safety*, not just efficiency: an operator
approving a corrector bump sees "last time this was done during a physics study,
it tripped interlock X — approver reverted within 2 min." Build it after D1–D4
exist, because it consumes their output.

### D6 — Positioning: which of the three paths is Osprey on?

**Decision:** Frame Osprey's record ambition explicitly.

Osprey is **not** path 1 (it does not replace EPICS/the control system — it
orchestrates over it) and **not** path 2 (it doesn't replace a module's ledger).
It is squarely **path 3 — a new system of record for *operational decisions***,
built in the glue function (operations) the article says is the tell. ARIEL is
already the embryonic form: the place operators go to ask "what happened?" The
strategic move is to extend ARIEL from a *log of observations/actions* into the
authoritative record of **"why did we do that?"** — the question no current
accelerator system can answer.

**Recommendation:** Adopt "system of record for operational decisions" as the
explicit north star for the ARIEL + approval + trace stack, and let it guide
roadmap tie-breaks (prefer features that make the *why* durable and queryable).

### D7 — Decision-quality observability (the Arize analog)

**Decision:** Do we need an eval/observability layer over agent decisions?

Once Decision Records accumulate, you can measure: approval/override rates per
tool, exception frequency per subsystem, time-to-approve, precedent-reuse rate,
and post-hoc "was this decision later reverted?" — i.e., **decision quality over
time**, the Arize role.

**Recommendation: lightweight, internal, and late.** Don't build/adopt a heavy
observability platform now. The Decision Record schema (D1) is what makes this
*possible later*; design the schema with these metrics in mind, then defer the
dashboards until there's volume to analyze.

### D8 — Governance: version the policy, scope the exception

**Decision:** How do exceptions and policy versions get recorded so the trace is
audit-grade?

The article's example carries "under policy v3.2, VP exception." Osprey's policy
is the limits DB + approval config. Today neither is versioned into the decision.

**Recommendation:** Snapshot the *active policy version* (limits DB hash/version +
approval policy in effect) into each Decision Record, and record exceptions as
structured `{limit_overridden, granted_by, scope, expires}` rather than a free
`detail` string. This is what turns "we let it through" into a defensible,
auditable exception — essential in a safety-critical / DOE-funded context where
auditability is not optional.

---

## Stage 5 — Sequencing against safety-first constraints

The article's "none of this requires full autonomy on day one — start
human-in-the-loop" is not a concession for Osprey; it is *already* Osprey's
operating model (mandatory approval gates). That alignment makes the sequencing
unusually clean:

| Wave | Build | Why first | Reuses |
|---|---|---|---|
| **1. Persist** | D1 Decision Record + D2 human rationale + D8 policy snapshot | Highest leverage, lowest lift; nothing downstream exists without it; pure additive to the approval hook | artifact store, approval hook, web terminal approval UI |
| **2. Freeze** | D4 bounded world-state snapshot | Makes records replayable/auditable; small extension of inputs already gathered | machine-state/archiver reads, transcript reader |
| **3. Link** | D3 context-graph links via ARIEL | Turns records into queryable precedent substrate | ARIEL sql/semantic/browse |
| **4. Compound** | D5 precedent retrieval at approval; D6 reframed ARIEL | The payoff loop; safety *and* efficiency upside | everything above |
| **5. Measure** | D7 decision-quality metrics | Needs accumulated volume; informs where autonomy can safely expand | Decision Record schema |

Progressive autonomy then follows the article's logic **and** Osprey's safety
posture: a class of decision earns reduced approval friction only once the graph
shows a stable precedent of human approvals with consistent rationale and no
reverts. The trace is what *licenses* autonomy — it doesn't bypass the gate, it
justifies relaxing it, with evidence.

---

## Stage 6 — Where the analogy breaks (what *not* to over-rotate on)

Intellectual honesty matters more than a clean thesis here:

- **Osprey is not a startup chasing a trillion-dollar TAM.** It is a BSD-licensed,
  DOE/LBNL-funded research framework. The article's *market* conclusions
  (lock-in, egress fees, displacing incumbents, platform economics) are
  irrelevant and should not leak into Osprey's design rationale. Adopt the
  *mechanism*, discard the *business model*.
- **Decision volume is lower; safety dominates over precedent reuse.** Enterprise
  value comes partly from *automating* high-volume exception-heavy work. In an
  accelerator, the primary value of the decision trace is **auditability, safety,
  and institutional memory across shift changes and operator turnover**, with
  efficiency a secondary benefit. Frame and prioritize accordingly — don't sell
  this internally as "automate the control room."
- **"Replace the system of record" is the wrong verb.** EPICS/the control system
  is the canonical state plane and stays that way (the article's own path-3
  pattern *syncs final state back* and adds the decision layer beside it). Osprey
  should be the system of record **for decisions**, layered over an unchanged
  control system — not a replacement narrative that would (rightly) alarm facility
  stakeholders.
- **Don't build the graph before the records.** The seductive failure mode is
  starting with a "context graph" abstraction. The article is explicit that the
  graph is an *emergent* consequence of persisting traces. D1 first; graph last.
- **The LLM-intent prohibition must survive.** D2 works *only* because the "why"
  is human-attributed fact. If anyone proposes letting the model backfill
  rationale to populate the graph, that re-introduces exactly the hazard
  `logbook.py` was written to prevent. The trust boundary is non-negotiable.

---

## One-paragraph summary for a busy reader

Osprey already occupies the one defensible position the article identifies — the
agentic orchestrator in the write path at commit time, doing cross-system
synthesis in the operations "glue" function — but it currently discards the most
valuable thing flowing through that position: *why* a gated action was taken. The
strategic opportunity is to stop discarding it. Capture a first-class **Decision
Record** at every approval gate (proposed action + policy snapshot + frozen
world-state + **human-supplied** rationale), link those records through ARIEL into
a precedent graph, and surface precedent back at the next approval. Done in that
order, this turns Osprey's approval gate from a one-shot yes/no into a compounding
**system of record for operational decisions** — improving auditability and
institutional memory first, and *licensing* progressive autonomy second, all
without weakening the safety model and without importing the article's irrelevant
market framing.

---

### Appendix — code touchpoints referenced

- Safety chain & data flow: `docs/source/architecture/index.rst`
- Approval gate (where D1/D2/D8 hook in): `src/osprey/templates/claude_code/claude/hooks/osprey_approval.py`
- "Never infer the why" logbook constraint: `src/osprey/interfaces/artifacts/logbook.py` (`BASE_PREAMBLE`, `SYSTEM_PROMPT`)
- Decision Record storage substrate: `src/osprey/stores/artifact_store.py`
- Context gathering / transcript / audit trail: `gather_context` and `TranscriptReader` (`logbook.py`, `mcp_server/workspace/transcript_reader.py`)
- Query plane for the graph: `src/osprey/mcp_server/ariel/tools/{sql_query,semantic_search,keyword_search,browse}.py`
