// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: message-list renderer.
 *
 * View-model for the operator chat log (the `op-*` console styled in
 * operator.css). A {@link createChatRenderer} instance owns the message-list
 * state a single stream of SSE-style events would otherwise scatter across the
 * controller: the agent turn being written (chat-turn.js holds its blocks), the
 * live loader, and the first-turn bookkeeping that decides whether a
 * `session_reset` is worth a divider. The blocks are built by the chat-*.js
 * modules it imports; markdown and math are rendered by chat-markdown.js.
 *
 * Trust boundary: chat renders *model* output — including verbatim tool
 * results from outside the trust boundary — so every markdown-to-HTML path
 * runs through {@link renderMarkdownInto} (chat-markdown.js), which sanitises
 * with DOMPurify before anything reaches the DOM and degrades to inert
 * `textContent` when the vendored libraries are absent. The scaffold Preview
 * renders agent-writable `.claude/` files through the same function.
 *
 * The libraries (`marked`, `DOMPurify`, `hljs`) are vendored classic-script
 * globals loaded before this module; `DOMPurify` has no ambient declaration,
 * and any of the three can be missing under a partial load, so they are read
 * off `globalThis` behind an `any` cast and every use is guarded.
 *
 * @module chat-render
 */

import {
  createErrorBlock,
  createLoader,
  createNotice,
  createSessionDivider,
  createTextBlock,
  createUserEntry,
  createAgentEntry,
} from './chat-blocks.js';
import { refusalLine } from './chat-tool-cards.js';
import { createTurn } from './chat-turn.js';
import { createApprovalCard } from './chat-approval.js';
import { createQuestionCard, createQuestionReplay } from './chat-question.js';
import { HIDDEN_TOOL_KEYS, QUESTION_TOOL_KEY, toolKey } from './chat-tool-vocab.js';
import { activityLabel } from './chat-activity.js';
import { renderMarkdownInto } from './chat-markdown.js';
import { linkArtifactIds } from './artifact-links.js';
import { emitChat } from './chat-events.js';
import { BACKGROUND_WAIT_LABEL, WORKING_LABEL, systemFrame } from './chat-system.js';
import { endsTurn } from './chat-frames.js';

export { elem } from './chat-dom.js';

/**
 * The frames that mark the agent's progress through a turn; a thinking row's
 * time runs from the last of them. Heartbeats and status frames do not count.
 */
const TIMED_FRAMES = new Set(['text', 'delta', 'thinking', 'tool_use', 'tool_result']);

/** @typedef {import('./chat-blocks.js').LoaderHandle} LoaderHandle */
/** @typedef {import('./chat-turn.js').AgentTurn} AgentTurn */
/** @typedef {import('./chat-tool-cards.js').ToolRow} ToolRow */
/** @typedef {import('./chat-tool-cards.js').Decision} Decision */
/** @typedef {import('./chat-tool-cards.js').ToolOutcome} ToolOutcome */
/** @typedef {import('./chat-permission.js').PermissionBody} PermissionBody */

/**
 * A chat event, post server-side stripping (see routes/chat.py
 * `_strip_for_chat`). Typed open on purpose: each event type carries its own
 * fields, and the renderer reads the ones its type has. The fields read most
 * are `type` (the discriminator: `text` | `delta` | `thinking` | `tool_use` |
 * `tool_result` | `permission_request` | `permission_resolved` | `result` |
 * `session_reset` | `error` | `system`), `content` (text, deltas and thinking),
 * `tool_name` and `tool_name_raw` (`tool_use`; the raw name, prefix intact, is
 * the preferred phrase key), `message` (`error`), `is_error`, `refusal` (a
 * `tool_result` the hooks refused: `{reason, phrase}`, else null), and `decision` (a
 * replayed `tool_result`: `allowed` or `denied`, when its call has one).
 *
 * @typedef {{ type: string, [key: string]: any }} ChatEvent
 */

/**
 * One conversation turn as served by `GET /api/session-chat` (see
 * routes/session.py). Without `events=1` an agent turn is its text alone; with
 * it, the turn also carries the stream the live chat received, so a replay
 * draws the same thinking, tool calls and plan.
 *
 * @typedef {object} ChatTurn
 * @property {string} role - `user` or `assistant`
 * @property {string} content - the turn's text; markdown for an agent turn
 * @property {string} [timestamp] - transcript timestamp; not rendered here
 * @property {ChatEvent[]} [events] - an agent turn's stream, when `events=1` is asked for
 */

/**
 * Render an agent's markdown into `el` and make its artifact ids clickable:
 * every agent-authored body in the log goes through here.
 * @param {HTMLElement} el
 * @param {string} text
 */
function renderAgentMarkdown(el, text) {
  renderMarkdownInto(el, text, { math: true, svg: true });
  linkArtifactIds(el);
}

// ---- Stateful renderer ---- //

/**
 * @typedef {object} ChatRenderer
 * @property {(text: string) => HTMLElement} addUserMessage - append an
 *   operator bubble and return it
 * @property {(markdown: string) => HTMLElement} addAgentMessage - append a
 *   finished agent entry rendered from markdown and return it
 * @property {(turns: ChatTurn[] | null | undefined) => number} replay - render
 *   prior conversation turns, leaving the session-reset divider state
 *   untouched; returns the number of entries rendered. A last agent turn whose
 *   history has no end is left open for the live stream to continue
 * @property {() => void} beginTurn - start a turn: shows the live loader
 * @property {(event: ChatEvent) => void} handleEvent - dispatch one SSE event
 * @property {() => void} endTurn - finish the turn: hides the loader and marks
 *   any card still waiting on the operator as cancelled
 * @property {() => void} reset - clear the list and all per-conversation state
 * @property {() => number} messageCount - rendered message entries so far
 *   (excludes dividers and the loader); primarily for tests
 * @property {() => boolean} hasOpenTurn - whether an agent turn is open: one a
 *   live stream draws, or a replayed one whose history has no end yet
 */

/**
 * @typedef {object} RendererOptions
 * @property {(requestId: string, body: PermissionBody) => Promise<unknown>} [answerPermission]
 *   delivers a card's answer; rejects with `status` 404 once the request is gone
 */

/**
 * Create a chat renderer bound to a message-list container (the `op-messages`
 * element). The renderer appends entries in arrival order, keeps the loader
 * pinned to the bottom of the log while a turn runs, and routes each event to
 * the block it belongs in.
 *
 * @param {HTMLElement} container - the `op-messages` scroll region
 * @param {RendererOptions} [options]
 * @returns {ChatRenderer}
 */
export function createChatRenderer(container, options = {}) {
  /**
   * Count of rendered *message* entries (operator / agent / error) in the
   * current conversation. Dividers and the loader don't count. Reported by
   * {@link messageCount} (primarily for tests).
   */
  let count = 0;

  /**
   * Whether a prior agent exchange — an agent message or a finished turn — has
   * been rendered in this conversation. Gates the session-reset divider.
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

  /** Whether a turn is in flight: from beginTurn until endTurn. */
  let running = false;
  /** When the turn in flight began (epoch ms); the loader and footer count from here. */
  let startedAt = 0;
  /** When the turn's last content frame landed (epoch ms): a thinking row's time starts here. */
  let lastFrameAt = 0;
  /** The loader's label: the current activity. */
  let activity = WORKING_LABEL;
  /** Whether the reply's text is streaming now; the status line says so. */
  let streamingText = false;
  /** The live loader, created on first use and then reused. */
  let loader = /** @type {LoaderHandle | null} */ (null);
  /** The agent turn in flight, or null before its first block. */
  let turn = /** @type {AgentTurn | null} */ (null);
  /** Tool rows of the conversation, by `tool_use_id`: kept across turns, so a replayed call's approval finds its row. */
  const chips = new Map();
  /** Permission cards of the turn in flight, keyed by `request_id`. */
  const cards = new Map();
  /** Subagent rows of the turn in flight, keyed by the call's `tool_use_id`. */
  const agents = new Map();
  /** `tool_use_id`s of the main agent's tool calls shown now and not yet answered. */
  const openTools = new Set();
  /** Redraws each replayed AskUserQuestion from its call and result: history holds no card. */
  const questionReplay = createQuestionReplay();

  /**
   * The agent row a record belongs to, when it is a subagent's own. A record
   * whose parent is not a row here (a subagent of a subagent) has no place
   * in the operator log and is dropped rather than shown as the main agent's.
   * @param {Record<string, any>} event
   * @returns {{ agent: any, nested: boolean }}
   */
  function ownerOf(event) {
    const parent = event.parent_tool_use_id;
    if (!parent) return { agent: null, nested: false };
    return { agent: agents.get(String(parent)) ?? null, nested: true };
  }

  const answer =
    options.answerPermission ??
    (async () => {
      throw new Error('Permission answers are not wired up.');
    });

  /**
   * Append a node to the log, keeping the loader (when it is showing) last so
   * it always reads as the current status beneath the latest message.
   * @param {Node} node
   */
  function append(node) {
    container.appendChild(node);
    if (loader?.el.isConnected) container.appendChild(loader.el);
  }

  /**
   * The current turn, created with its first block.
   * @returns {AgentTurn}
   */
  function ensureTurn() {
    if (turn) return turn;
    turn = createTurn({
      renderText: renderAgentMarkdown,
      // A live finish of the reasoning: the osprey on the status line flies once.
      onThinkingEnd: () => {
        if (!replaying) loader?.flourish();
      },
    });
    count += 1;
    hasPriorExchange = true;
    append(turn.entry);
    return turn;
  }

  /** @param {Record<string, any>} event - a `tool_use` event */
  function appendToolUse(event) {
    const t = ensureTurn();
    const raw = String(event.tool_name_raw ?? '');
    const id = String(event.tool_use_id ?? '');
    if (raw === 'Agent' || raw === 'Task') {
      const agent = t.agent(event);
      if (id) {
        agents.set(id, agent);
        chips.set(id, agent);
      }
      return;
    }
    const chip = t.tool(event);
    if (id) chips.set(id, chip);
  }

  /**
   * The tool row a permission request is filed under: the row of the same call,
   * wherever it sits, when that row has no card yet. Plans and questions have none.
   * @param {Record<string, any>} request
   * @returns {ToolRow | null}
   */
  function rowFor(request) {
    if (request.kind !== 'tool' || !request.tool_use_id) return null;
    const row = chips.get(String(request.tool_use_id)) ?? null;
    return row && !row.hasCard ? row : null;
  }

  /**
   * Draw a replayed call's decision on its row, the way an answer draws it live.
   * The row's own placement takes the decision with a placeholder that leaves at
   * once: no card is shown and nothing takes focus. A row that holds a card keeps it.
   * @param {ToolRow | undefined} row
   * @param {Decision} decision
   */
  function showDecision(row, decision) {
    if (!row || row.hasCard) return;
    row.file(document.createElement('div')).decide(decision);
  }

  /** @param {Record<string, any>} request - a `permission_request` event */
  function appendPermission(request) {
    const id = String(request.request_id ?? '');
    const row = rowFor(request);
    const hooks = {
      answer: (/** @type {PermissionBody} */ body) => answer(id, body),
      onSettled: () => syncLoader(),
      renderText: renderAgentMarkdown,
    };
    let card;
    if (row) {
      // Under its own row the card belongs to the call, not to the message it came
      // in: the text and reasoning around it close, and the row leaves its run.
      if (turn) {
        turn.closeText();
        turn.closeThinking();
        turn.countApproval();
      }
      card = createApprovalCard(request, { ...hooks, row });
    } else {
      const t = ensureTurn();
      if (request.kind !== 'question') t.countApproval();
      card =
        request.kind === 'question'
          ? createQuestionCard(request, hooks)
          : createApprovalCard(request, hooks);
      t.place(card.el);
    }
    cards.set(id, card);
    // A live card takes focus, but never from the composer's text box, and never while history is redrawn.
    const typing = document.activeElement;
    if (!replaying && !(typing instanceof HTMLTextAreaElement && !typing.disabled)) card.focus();
  }

  /** @param {Record<string, any>} event - the `result` event */
  function appendFooter(event) {
    if (!turn) return;
    const durationMs =
      typeof event.duration_ms === 'number' ? event.duration_ms : Date.now() - startedAt;
    turn.footer({ durationMs, isError: Boolean(event.is_error) });
  }

  /**
   * The status line stays at the foot of the log for the whole turn, from
   * beginTurn to endTurn, so there is always a sign of what the agent is
   * doing. Its label follows the turn: a card waiting on the operator, the
   * reply being written, or the current activity.
   */
  function syncLoader() {
    if (!running) {
      loader?.stop();
      loader?.el.remove();
      return;
    }
    const waiting = [...cards.values()].some((card) => card.pending);
    if (!loader) loader = createLoader();
    const writing = turn?.drawing() ? 'Drawing…' : 'Writing…';
    loader.setLabel(waiting ? 'Waiting for your answer' : streamingText ? writing : activity);
    loader.start(startedAt);
    if (container.lastElementChild !== loader.el) container.appendChild(loader.el);
  }

  /**
   * Close the turn in flight: settle its open blocks, cancel every card still
   * waiting (the server has moved on without those answers), and finish the
   * subagents still running. Tool rows stay, so a later answer can still reach
   * them. The running state is left alone — see {@link endTurn}.
   */
  function dropTurn() {
    turn?.close();
    turn = null;
    for (const card of cards.values()) card.cancel();
    cards.clear();
    for (const agent of agents.values()) agent.finish({ status: 'stopped' });
    agents.clear();
    openTools.clear();
    streamingText = false;
  }

  /** The turn is over: a result, a terminal error, or a closed stream. */
  function endTurn() {
    const wasRunning = running;
    dropTurn();
    running = false;
    syncLoader();
    if (wasRunning) emitChat('turn-end', {});
  }

  /** @param {string} text */
  function addUserMessage(text) {
    // A new operator message starts a new turn; abandon any dangling one.
    endTurn();
    const entry = createUserEntry(text);
    count += 1;
    append(entry);
    emitChat('user-message', { text, replay: replaying });
    return entry;
  }

  /**
   * Append a finished agent entry rendered from `markdown` in one go — the
   * whole-message counterpart to the streamed text path, through the same
   * sanitising {@link renderMarkdownInto} boundary.
   * @param {string} markdown
   * @returns {HTMLElement}
   */
  function addAgentMessage(markdown) {
    dropTurn();
    const entry = createAgentEntry();
    const body = createTextBlock();
    renderAgentMarkdown(body, markdown);
    entry.appendChild(body);
    count += 1;
    hasPriorExchange = true;
    append(entry);
    syncLoader();
    emitChat('agent-message', { markdown, replay: replaying });
    return entry;
  }

  /** True while {@link replay} redraws history; listeners on chat-events read it. */
  let replaying = false;

  /**
   * Render prior conversation turns into the log, oldest first.
   *
   * Replayed turns are the same conversation continued in another window, not
   * a completed exchange in this one, so the divider bookkeeping is restored
   * afterwards: the next stream's frame-0 `session_reset` still draws no
   * divider above replayed history. Entries are appended — a caller wanting a
   * clean log calls {@link reset} first.
   *
   * The history's last agent turn may have no end yet — no `result` and no
   * error that ends a turn — because the agent is still on it. That turn
   * stays open, so the live frames that follow continue it in the same entry.
   *
   * @param {ChatTurn[] | null | undefined} turns - as served by
   *   `GET /api/session-chat`; a missing list renders nothing
   * @returns {number} entries rendered; a turn with no content is skipped
   */
  function replay(turns) {
    const list = turns ?? [];
    const priorExchange = hasPriorExchange;
    let rendered = 0;
    replaying = true;
    try {
      for (const [index, turnItem] of list.entries()) {
        const content = turnItem?.content ?? '';
        const events = Array.isArray(turnItem?.events) ? turnItem.events : null;
        if ((turnItem.role ?? '').toLowerCase() === 'user') {
          if (!content) continue;
          addUserMessage(content);
        } else if (events?.length) {
          // The same path a live stream takes, so thinking, tool rows and the
          // plan come back exactly as they were drawn; `result` closes the turn.
          for (const event of events) handleEvent(event);
          const unfinished = index === list.length - 1 && !events.some(endsTurn);
          if (!unfinished) endTurn();
        } else if (content) {
          addAgentMessage(content);
        } else {
          continue;
        }
        rendered += 1;
      }
    } finally {
      replaying = false;
    }
    hasPriorExchange = priorExchange;
    return rendered;
  }

  function beginTurn() {
    running = true;
    startedAt = Date.now();
    lastFrameAt = startedAt;
    activity = WORKING_LABEL;
    streamingText = false;
    syncLoader();
  }

  /**
   * When a thinking row's time starts: live, the turn's last frame before it;
   * replay, the duration the history recorded, or null when it has none.
   * @param {ChatEvent} event
   */
  function thinkingTiming(event) {
    if (!replaying) return { since: lastFrameAt };
    return { durationMs: typeof event.duration_ms === 'number' ? event.duration_ms : null };
  }

  /** @param {ChatEvent} event */
  function handleEvent(event) {
    switch (event.type) {
      case 'text': {
        const { agent, nested } = ownerOf(event);
        if (nested) {
          agent?.addText(event.content ?? '');
          break;
        }
        streamingText = true;
        ensureTurn().text(event.content ?? '');
        break;
      }
      case 'delta': {
        // A streamed piece of the main agent's reply or reasoning; a
        // subagent's pieces are not drawn — its row gets each whole block.
        if (event.parent_tool_use_id) break;
        if (event.kind === 'thinking') {
          streamingText = false;
          activity = 'Thinking…';
          ensureTurn().think(event.content ?? '', true, thinkingTiming(event));
        } else {
          streamingText = true;
          ensureTurn().textDelta(event.content ?? '');
        }
        break;
      }
      case 'thinking':
        // A subagent's reasoning stays inside it; its row shows its work.
        if (ownerOf(event).nested) break;
        streamingText = false;
        activity = 'Thinking…';
        ensureTurn().think(event.content ?? '', false, thinkingTiming(event));
        break;
      case 'tool_use': {
        const { agent, nested } = ownerOf(event);
        if (nested) {
          if (agent) {
            const chip = agent.addTool(event);
            if (event.tool_use_id) chips.set(String(event.tool_use_id), chip);
            activity = `${agent.name} · ${activityLabel(event)}`;
          }
          break;
        }
        streamingText = false;
        const key = toolKey(event.tool_name_raw ?? event.tool_name);
        activity = key === QUESTION_TOOL_KEY ? 'Preparing a question…' : activityLabel(event);
        if (key === QUESTION_TOOL_KEY) {
          if (replaying) questionReplay.ask(event);
        } else if (!HIDDEN_TOOL_KEYS.has(key)) {
          appendToolUse(event);
          if (event.tool_use_id) openTools.add(String(event.tool_use_id));
        }
        break;
      }
      case 'tool_start': {
        // The agent began writing a call; name it now, as its input can take
        // seconds to arrive. The row waits for the whole `tool_use`.
        const { agent, nested } = ownerOf(event);
        if (nested) {
          if (agent) activity = `${agent.name} · ${activityLabel(event)}`;
          break;
        }
        streamingText = false;
        const key = toolKey(event.tool_name_raw ?? event.tool_name);
        activity = key === QUESTION_TOOL_KEY ? 'Preparing a question…' : activityLabel(event);
        break;
      }
      case 'tool_result': {
        streamingText = false;
        // The call's phrase is over once its result is in; with no other call of
        // the turn still open the line says the agent is working, as beginTurn does.
        if (!event.parent_tool_use_id && openTools.delete(String(event.tool_use_id ?? ''))) {
          if (openTools.size === 0) activity = WORKING_LABEL;
        }
        const trace = questionReplay.answer(event);
        if (trace) {
          ensureTurn().place(trace);
          break;
        }
        const row = chips.get(String(event.tool_use_id ?? ''));
        // Only a replayed call carries a decision; a live frame never does.
        if (event.decision) showDecision(row, event.decision);
        if (row) {
          row.finish(event);
        } else {
          // A refused write with no row here (a hidden tool, a subagent's subagent)
          // still says who can lift it.
          const line = refusalLine(/** @type {ToolOutcome} */ (event));
          if (line) ensureTurn().place(line);
        }
        break;
      }
      case 'permission_request':
        streamingText = false;
        appendPermission(event);
        break;
      case 'permission_resolved':
        // Answered somewhere — another tab, or this one; a card still waiting
        // here settles to the same outcome.
        cards.get(String(event.request_id ?? ''))?.resolved(event);
        break;
      case 'result':
        // Turn boundary. A result finalises the turn and earns the footer; it
        // also marks a completed exchange, so a later reset earns its divider.
        streamingText = false;
        hasPriorExchange = true;
        appendFooter(event);
        endTurn();
        break;
      case 'session_reset':
        // The divider shows only when a completed exchange already sits above
        // this reset; the current turn's own user bubble is not enough (that is
        // the fresh-session first turn). The divider is not a message.
        if (hasPriorExchange) {
          dropTurn();
          append(createSessionDivider());
        }
        break;
      case 'error': {
        streamingText = false;
        count += 1;
        hasPriorExchange = true;
        const block = createErrorBlock(event.message ?? 'An error occurred.');
        if (turn) turn.append(block);
        else append(block);
        // An assistant-message error leaves the turn streaming; anything else ends it.
        if (endsTurn(event)) endTurn();
        break;
      }
      case 'system': {
        // Most system frames draw nothing; chat-system.js says what the rest do.
        const effect = systemFrame(event);
        if (!effect) break;
        if (effect.stopStreaming) streamingText = false;
        if (effect.activity) activity = effect.activity;
        if (effect.closeText && turn) turn.closeText();
        if (effect.agentDone) {
          agents.get(effect.agentDone.id)?.finish(effect.agentDone.outcome);
          // The wait is over when no background agent is left running.
          const waiting = activity === BACKGROUND_WAIT_LABEL;
          if (waiting && ![...agents.values()].some((a) => a.running())) activity = WORKING_LABEL;
        }
        if (effect.notice) {
          const notice = createNotice(effect.notice);
          if (turn) turn.append(notice);
          else append(notice);
        }
        break;
      }
      default:
        // Unknown types are not rendered in the operator log.
        break;
    }
    if (TIMED_FRAMES.has(event.type)) lastFrameAt = Date.now();
    syncLoader();
    emitChat('frame', { event, replay: replaying });
  }

  function reset() {
    dropTurn();
    running = false;
    loader?.stop();
    loader = null;
    container.replaceChildren();
    count = 0;
    hasPriorExchange = false;
    chips.clear();
    emitChat('reset', {});
  }

  return {
    addUserMessage,
    addAgentMessage,
    replay,
    beginTurn,
    handleEvent,
    endTurn,
    reset,
    messageCount: () => count,
    hasOpenTurn: () => running || turn !== null,
  };
}
