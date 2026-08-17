// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: message-list renderer.
 *
 * View-model + DOM builders for the operator chat log (the `op-*` console
 * styled in operator.css). A {@link createChatRenderer} instance owns the
 * message-list state a single stream of SSE-style events would otherwise
 * scatter across the controller: the accumulating agent-text buffer, the
 * transient activity line, and the first-turn bookkeeping that decides
 * whether a `session_reset` is worth a divider.
 *
 * Trust boundary: unlike scaffold/detail-content.js (which renders trusted,
 * on-disk artifact markdown straight to innerHTML), chat renders *model*
 * output — including verbatim tool results from outside the trust boundary —
 * so every markdown-to-HTML path runs through {@link renderMarkdownInto},
 * which sanitises with DOMPurify before anything reaches the DOM and degrades
 * to inert `textContent` when the vendored libraries are absent.
 *
 * The libraries (`marked`, `DOMPurify`, `hljs`) are vendored classic-script
 * globals loaded before this module; `DOMPurify` has no ambient declaration,
 * and any of the three can be missing under a partial load, so they are read
 * off `globalThis` behind an `any` cast and every use is guarded.
 *
 * @module chat-render
 */

/**
 * A chat event, post server-side stripping (see routes/chat.py
 * `_strip_for_chat`). Only the fields this renderer reads are typed; the
 * stripped wire objects carry a few more (`tool_use_id`, `error_type`, …).
 *
 * @typedef {object} ChatEvent
 * @property {string} type - discriminator: `text` | `thinking` | `tool_use`
 *   | `tool_result` | `result` | `session_reset` | `error` | `system` | …
 * @property {string} [content] - incremental text (`text` events)
 * @property {string} [tool_name] - display name, prefix-stripped (`tool_use`)
 * @property {string} [tool_name_raw] - the SDK's own tool name, prefix intact
 *   (`tool_use`); the preferred {@link TOOL_PHRASES} key
 * @property {string} [message] - human-readable error text (`error` events)
 * @property {boolean} [is_error] - turn/tool errored (`result`, `tool_result`)
 */

/**
 * @returns {{ DOMPurify: any, hljs: any }} the vendored sanitiser/highlighter
 *   globals, each possibly `undefined`. (`marked` is reached via
 *   {@link chatMarkedParse}, which isolates chat from the shared singleton.)
 */
function markdownGlobals() {
  const g = /** @type {any} */ (globalThis);
  return { DOMPurify: g.DOMPurify, hljs: g.hljs };
}

/**
 * Identity of the global `marked` the memoised {@link chatMarkedParse} was
 * last built against, and the parse function derived from it. Kept module-level
 * so the (stable) production singleton yields exactly one isolated instance,
 * while per-test global swaps — which change the identity — force a rebuild.
 * @type {any}
 */
let lastMarked = null;
/** @type {((text: string) => string) | null} */
let chatParse = null;

/**
 * A `parse(markdown) -> html` function isolated from the shared `marked`
 * singleton, or `null` when `marked` is absent.
 *
 * The scaffold hub configures the global singleton in place —
 * `scaffold/utils.js` calls `marked.use({ renderer, walkTokens, … })`, loaded
 * on every hub page — and that custom renderer emits empty
 * `<pre><code></code></pre>` for chat's fenced blocks. marked v12 exposes its
 * `Marked` class on the UMD global, so chat builds its own instance with a
 * plain config and parses through that, immune to whatever the rest of the page
 * does to the singleton. When the constructor is absent (an older or partial
 * vendored build) we fall back to the global `marked.parse` — degraded, but no
 * worse than before this isolation existed.
 *
 * Resolved lazily rather than at module load: the vendored global loads before
 * this module in the browser, but tests stub `globalThis.marked` per case, so
 * the instance is (re)built whenever the observed global changes identity.
 *
 * @returns {((text: string) => string) | null}
 */
function chatMarkedParse() {
  const marked = /** @type {any} */ (globalThis).marked;
  if (marked === lastMarked) return chatParse;
  lastMarked = marked;

  if (!marked) {
    chatParse = null;
  } else if (typeof marked.Marked === 'function') {
    try {
      const instance = new marked.Marked({ gfm: true, breaks: false });
      chatParse = (text) => instance.parse(text);
    } catch {
      chatParse =
        typeof marked.parse === 'function' ? (text) => marked.parse(text) : null;
    }
  } else {
    chatParse =
      typeof marked.parse === 'function' ? (text) => marked.parse(text) : null;
  }
  return chatParse;
}

/**
 * Render markdown text into `el` as sanitised HTML, then syntax-highlight any
 * code blocks. This is the chat trust boundary.
 *
 * HTML is produced only when BOTH `marked` and `DOMPurify` are available: the
 * marked output is untrusted HTML, so it must never reach `innerHTML` without
 * passing through `DOMPurify.sanitize` first. If either library is missing (or
 * either step throws), the text is written as inert `textContent` — never as
 * markup. `hljs` is a purely additive enhancement and is skipped when absent.
 *
 * @param {HTMLElement} el - target element; its contents are replaced
 * @param {string} text - raw markdown (model output)
 * @returns {void}
 */
export function renderMarkdownInto(el, text) {
  const { DOMPurify, hljs } = markdownGlobals();
  const parse = chatMarkedParse();

  // Fail safe: without a parser or a sanitiser we cannot emit HTML, so degrade
  // to text.
  if (!parse || !DOMPurify || typeof DOMPurify.sanitize !== 'function') {
    el.textContent = text;
    return;
  }

  let clean;
  try {
    clean = DOMPurify.sanitize(parse(text));
  } catch {
    // Any failure in parse/sanitise falls back to inert text rather than
    // risking unsanitised markup reaching the DOM.
    el.textContent = text;
    return;
  }
  el.innerHTML = clean;

  if (hljs && typeof hljs.highlightElement === 'function') {
    el.querySelectorAll('pre code').forEach((block) => {
      try {
        hljs.highlightElement(/** @type {HTMLElement} */ (block));
      } catch {
        // A single block failing to highlight must not break the message.
      }
    });
  }
}

// ---- Tool vocabulary ---- //

/**
 * Operator phrases for the tools a turn is likely to use, keyed by normalised
 * tool name (see {@link normaliseToolName}).
 *
 * In Simple mode the chat is the whole interface, so this line is the only
 * account an operator gets of what the agent is doing. Phrases say it in
 * control-room terms and share the activity strip's vocabulary (open/close/
 * focus/arrange for panels, "wrote" for channels).
 *
 * They are lower-case gerunds — "writing control channels" — so a caller can
 * drop one mid-sentence; {@link activityLabel} capitalises for the activity
 * line. Coverage is deliberately partial: a tool whose formatted name already
 * reads as plain English ("List Panels", "Session Summary") falls through to
 * the raw-name fallback instead of earning a row. Adding one is a single line.
 *
 * Phrases cannot name the specific panel, channel, or file involved: the chat
 * route (`_strip_for_chat`) drops a `tool_use` event's `input` before it leaves
 * the server, so no arguments reach this module.
 *
 * @type {Readonly<Record<string, string>>}
 */
export const TOOL_PHRASES = Object.freeze({
  // Control system — the writes an operator most needs to see coming.
  channel_write: 'writing control channels',
  channel_read: 'reading control channels',
  channel_limits: 'checking channel limits',
  archiver_read: 'reading archived data',
  archiver_downsample: 'thinning archived data',

  // Python executor.
  execute: 'running Python',
  execute_file: 'running Python',

  // Workspace panels. The synthetic panel activity uses these same names.
  add_panel_to_rail: 'making a panel available',
  remove_panel_from_rail: 'removing a panel',
  open_panel: 'opening a panel',
  close_panel: 'closing a panel',
  arrange_workspace: 'arranging the workspace',
  register_panel: 'adding a panel',
  manage_window: 'arranging a window',
  screenshot_capture: 'taking a screenshot',

  // Workspace artifacts (plots, documents and stored datasets alike).
  artifact_save: 'saving an artifact',
  artifact_get: 'opening an artifact',
  artifact_list: 'listing saved artifacts',
  artifact_read: 'reading a saved artifact',
  artifact_focus: 'showing an artifact',
  artifact_pin: 'pinning an artifact',
  artifact_export: 'exporting an artifact',
  artifact_delete: 'deleting an artifact',
  artifact_delete_all: 'deleting artifacts',
  create_static_plot: 'drawing a plot',
  create_interactive_plot: 'drawing an interactive plot',
  create_dashboard: 'building a dashboard',
  create_document: 'writing a document',

  // Workspace project setup and session record.
  setup_inspect: 'inspecting the project setup',
  setup_patch: 'changing the project setup',
  session_log: 'writing to the session log',

  // Plan queue (bluesky).
  queue_status: 'checking the plan queue',
  queue_list: 'listing the plan queue',
  queue_add: 'queueing a plan',
  queue_start: 'starting the plan queue',
  queue_stop: 'stopping the plan queue',
  stop_run: 'stopping the running plan',
  write_plan: 'drafting a plan',
  validate_plan: 'checking the plan',
  get_draft: 'reading the plan draft',
  set_draft: 'editing the plan draft',
  clear_draft: 'clearing the plan draft',
  get_run: 'reading a plan run',
  get_run_data: 'reading run data',
  get_run_figure: 'reading a run figure',

  // Phoebus displays.
  phoebus_open_panel: 'opening a Phoebus display',
  phoebus_open_databrowser: 'opening the data browser',
  phoebus_list_displays: 'listing Phoebus displays',
  phoebus_perceive: 'reading a Phoebus display',
  phoebus_perceive_region: 'reading part of a Phoebus display',
  phoebus_snapshot: 'capturing a Phoebus display',
  phoebus_drive: 'operating a Phoebus display',

  // Logbook (ARIEL).
  browse: 'browsing the logbook',
  keyword_search: 'searching the logbook',
  semantic_search: 'searching the logbook',
  sql_query: 'querying the logbook',
  filter_options: 'listing logbook filters',
  entry_get: 'reading a logbook entry',
  entries_by_ids: 'reading logbook entries',
  entry_create: 'drafting a logbook entry',
  entry_publish: 'publishing a logbook entry',

  // Channel finder.
  list_channels: 'looking up channels',
  ask_channels: 'looking up channels',
  run_sql: 'querying the channel database',
  build_channels: 'building a channel list',
  list_families: 'looking up channel families',
  list_systems: 'looking up systems',
  get_common_names: 'looking up channel names',
  inspect_fields: 'inspecting channel fields',

  // Facility knowledge.
  list_concepts: 'browsing facility knowledge',
  read_concept: 'reading facility knowledge',
  draft_concept: 'drafting a facility note',

  // Lattice model. The mutators are phrased; the getters fall back.
  lattice_init: 'loading the lattice',
  lattice_state: 'reading the lattice',
  lattice_set_param: 'changing a lattice parameter',
  lattice_set_baseline: 'setting the lattice baseline',
  lattice_clear_baseline: 'clearing the lattice baseline',
  lattice_update_settings: 'changing lattice settings',
  lattice_refresh: 'refreshing the lattice view',

  // Health.
  health_check: 'checking system health',
  health_check_full: 'checking system health',

  // Built-in agent tools.
  read: 'reading a file',
  write: 'writing a file',
  edit: 'editing a file',
  bash: 'running a shell command',
  glob: 'looking for files',
  grep: 'searching files',
  task: 'delegating to a helper agent',
  todowrite: 'updating its plan',
  webfetch: 'fetching a web page',
  websearch: 'searching the web',
});

/**
 * Fold a tool name to a {@link TOOL_PHRASES} key: drop the `mcp__<server>__`
 * prefix, collapse whitespace and hyphens to underscores, lower-case.
 *
 * A `tool_use` event carries the name twice — `tool_name_raw`
 * (`mcp__osprey__channel_write`) and the server-formatted `tool_name`
 * (`Channel Write`, from operator_session `_format_tool_name`) — and both fold
 * to the same key, so the table works whichever spelling an event carries.
 *
 * @param {string} name
 * @returns {string}
 */
export function normaliseToolName(name) {
  return name
    .replace(/^mcp__[^_]+__/, '')
    .trim()
    .replace(/[\s-]+/g, '_')
    .toLowerCase();
}

/**
 * The operator phrase for a `tool_use` event, or null when the tool has no
 * table entry. Lower-case and sentence-fragment shaped; capitalise at the
 * point of display.
 * @param {ChatEvent} event
 * @returns {string | null}
 */
export function toolPhrase(event) {
  for (const name of [event.tool_name_raw, event.tool_name]) {
    if (!name) continue;
    const phrase = TOOL_PHRASES[normaliseToolName(name)];
    if (phrase) return phrase;
  }
  return null;
}

/**
 * The activity-line label for a `tool_use` event. A mapped tool reads as a
 * sentence ("Writing control channels…"); anything unmapped keeps its raw name
 * ("Using Queue Reorder…"), so a tool added elsewhere in the codebase is never
 * invisible here — only terse until it earns a phrase.
 * @param {ChatEvent} event
 * @returns {string}
 */
export function activityLabel(event) {
  const phrase = toolPhrase(event);
  if (phrase !== null) return `${phrase.charAt(0).toUpperCase()}${phrase.slice(1)}…`;
  return `Using ${event.tool_name ?? event.tool_name_raw ?? 'tool'}…`;
}

// ---- Pure DOM builders ---- //

/**
 * Create an element with a class and optional text. Shared with chat.js —
 * the chat modules never write markup through this builder.
 * @param {string} tag
 * @param {string} className
 * @param {string} [text] - assigned via textContent (never innerHTML)
 * @returns {HTMLElement}
 */
export function elem(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Build an operator (user) log entry — plain text, no markdown.
 * @param {string} text
 * @returns {HTMLElement}
 */
export function buildUserEntry(text) {
  const entry = elem('div', 'op-entry operator');
  entry.appendChild(elem('div', 'op-entry-prefix', 'Operator'));
  entry.appendChild(elem('div', 'op-entry-body', text));
  return entry;
}

/**
 * Build an empty agent (assistant) log entry. The body carries both
 * `op-entry-body` (console styling) and `osprey-md-rendered` (the shared
 * markdown-content class), and is the element {@link renderMarkdownInto}
 * writes into as text streams in.
 * @returns {{ entry: HTMLElement, body: HTMLElement }}
 */
export function buildAgentEntry() {
  const entry = elem('div', 'op-entry assistant');
  entry.appendChild(elem('div', 'op-entry-prefix', 'Osprey'));
  const body = elem('div', 'op-entry-body osprey-md-rendered');
  entry.appendChild(body);
  return { entry, body };
}

/**
 * Build an error block. Rendered from error events; text only.
 * @param {string} message
 * @returns {HTMLElement}
 */
export function buildErrorEntry(message) {
  return elem('div', 'op-error-block', message);
}

/**
 * Build a session-reset divider (a centred system notice).
 * @returns {HTMLElement}
 */
export function buildSessionResetDivider() {
  return elem('div', 'op-system', 'session reset');
}

/**
 * Build the transient activity line (LED + label). Created inactive; the
 * renderer toggles `.active` (which flips its `display`) as work starts/stops.
 * @param {string} label
 * @returns {{ line: HTMLElement, labelEl: HTMLElement }}
 */
export function buildActivityLine(label) {
  const line = elem('div', 'op-processing');
  line.appendChild(elem('span', 'op-processing-led'));
  const labelEl = elem('span', 'op-processing-label', label);
  line.appendChild(labelEl);
  return { line, labelEl };
}

// ---- Stateful renderer ---- //

/**
 * @typedef {object} ChatRenderer
 * @property {(text: string) => HTMLElement} addUserMessage - append an
 *   operator entry and return it
 * @property {(event: ChatEvent) => void} handleEvent - dispatch one SSE event
 * @property {() => void} reset - clear the list and all per-conversation state
 * @property {() => number} messageCount - rendered message entries so far
 *   (excludes dividers/activity); primarily for tests
 */

/**
 * Create a chat renderer bound to a message-list container (the `op-messages`
 * element). The renderer appends entries in arrival order and keeps the
 * activity line pinned to the bottom of the log.
 *
 * @param {HTMLElement} container - the `op-messages` scroll region
 * @returns {ChatRenderer}
 */
export function createChatRenderer(container) {
  /**
   * Count of rendered *message* entries (operator / agent / error) in the
   * current conversation. Dividers and the activity line don't count. Reported
   * by {@link messageCount} (primarily for tests).
   */
  let count = 0;

  /**
   * Whether a prior agent exchange — an agent message or a finalised turn —
   * has been rendered in this conversation. Gates the session-reset divider.
   *
   * The real turn sequence is: the controller appends the operator's bubble,
   * *then* opens the stream, and a fresh server session emits `session_reset`
   * as frame 0. So a plain message count is already ≥ 1 by the time the first
   * reset arrives and can't distinguish a genuine first turn from a later one.
   * A reset is only worth a divider when a completed exchange sits above it, so
   * we track that explicitly and never let the current turn's own just-added
   * user bubble count as prior history.
   */
  let hasPriorExchange = false;

  /** The in-progress agent entry's body, or null between turns. */
  let agentBody = /** @type {HTMLElement | null} */ (null);
  /** Text accumulated across `text` events for the current agent entry. */
  let agentText = '';
  /** The activity line, created lazily on first use and then reused. */
  let activity = /** @type {{ line: HTMLElement, labelEl: HTMLElement } | null} */ (null);

  /**
   * Append an entry, keeping the activity line (if present) last so it always
   * reads as the current status beneath the latest message.
   * @param {HTMLElement} node
   */
  function append(node) {
    container.appendChild(node);
    if (activity && activity.line.parentNode === container) {
      container.appendChild(activity.line);
    }
  }

  /**
   * Show the activity line with `label`, creating it on first use.
   * @param {string} label
   */
  function setActivity(label) {
    if (!activity) activity = buildActivityLine(label);
    else activity.labelEl.textContent = label;
    activity.line.classList.add('active');
    container.appendChild(activity.line); // move/keep at the bottom
  }

  /** Hide the activity line (kept in the DOM, inert, for reuse). */
  function clearActivity() {
    if (activity) activity.line.classList.remove('active');
  }

  /**
   * Ensure an agent entry exists for the current turn and stream `chunk` into
   * it. The first `text` event of a turn creates the entry (and clears any
   * activity line); subsequent events accumulate and re-render.
   * @param {string} chunk
   */
  function appendAgentText(chunk) {
    clearActivity();
    if (!agentBody) {
      const { entry, body } = buildAgentEntry();
      agentBody = body;
      agentText = '';
      count += 1;
      hasPriorExchange = true;
      append(entry);
    }
    agentText += chunk;
    renderMarkdownInto(agentBody, agentText);
  }

  /** End the current agent turn: drop the streaming buffer and activity line. */
  function finishTurn() {
    clearActivity();
    agentBody = null;
    agentText = '';
  }

  /** @param {string} text */
  function addUserMessage(text) {
    // A new operator message starts a new turn; abandon any dangling stream.
    agentBody = null;
    agentText = '';
    const entry = buildUserEntry(text);
    count += 1;
    append(entry);
    return entry;
  }

  /** @param {ChatEvent} event */
  function handleEvent(event) {
    switch (event.type) {
      case 'text':
        appendAgentText(event.content ?? '');
        break;
      case 'thinking':
        setActivity('Thinking…');
        break;
      case 'tool_use':
        setActivity(activityLabel(event));
        break;
      case 'tool_result':
        // Stripped of its body; nothing to render. The activity line stays as
        // it is until the next text/thinking/tool_use event moves it.
        break;
      case 'result':
        // Turn boundary. Errors surface via `error` events, so a result only
        // finalises streaming state here — but it does mark a completed
        // exchange, so a later reset earns its divider.
        hasPriorExchange = true;
        finishTurn();
        break;
      case 'session_reset':
        // Show the divider only when a completed exchange already sits above
        // this reset; the current turn's own user bubble is not enough (that is
        // the fresh-session first turn). The divider is not a message, so it
        // doesn't bump `count`.
        if (hasPriorExchange) {
          finishTurn();
          append(buildSessionResetDivider());
        }
        break;
      case 'error': {
        clearActivity();
        agentBody = null;
        agentText = '';
        count += 1;
        hasPriorExchange = true;
        append(buildErrorEntry(event.message ?? 'An error occurred.'));
        break;
      }
      default:
        // `system` and any unknown types are not rendered in the operator log.
        break;
    }
  }

  function reset() {
    container.replaceChildren();
    count = 0;
    hasPriorExchange = false;
    agentBody = null;
    agentText = '';
    activity = null;
  }

  return {
    addUserMessage,
    handleEvent,
    reset,
    messageCount: () => count,
  };
}
