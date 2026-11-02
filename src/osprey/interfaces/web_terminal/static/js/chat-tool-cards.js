// @ts-check
/**
 * OSPREY Web Terminal — operator chat: tool calls.
 *
 * The rows a tool call is drawn as: one row per tool call, and one row per
 * subagent call with its own trace nested inside. Both are action rows
 * (chat-row.js): the category glyph in the gutter, the tool's name, its
 * one-line input, and the status at the right edge. An approval card waiting on
 * a call sits under that call's row, which carries the operator's answer once
 * it is given. The turn's rows are placed and folded into runs by
 * chat-tool-runs.js.
 *
 * @module chat-tool-cards
 */

import { chatIcon } from './chat-icons.js';
import { formatToolResult, inputView } from './chat-format.js';
import { elem } from './chat-dom.js';
import { createRow } from './chat-row.js';
import {
  CATEGORY_ICON,
  foldKey,
  oneLine,
  summarizeInput,
  toolCategory,
  toolLabel,
} from './chat-tool-vocab.js';

/** @typedef {import('./chat-dom.js').WireEvent} WireEvent */

/**
 * What a refused write's result frame carries: the closed reason and the
 * phrase, which says who can lift the refusal (C-FRAME; set only on an error).
 * @typedef {{ reason: string, phrase: string }} Refusal
 */

/**
 * @typedef {{ is_error?: boolean, preview?: string, truncated?: boolean, refusal?: Refusal | null }} ToolOutcome
 */

/**
 * What was decided on a call's approval, as the call's row states it: the
 * operator's answer, or `expired` when the server declined an unanswered ask.
 * @typedef {'allowed' | 'session' | 'denied' | 'expired'} Decision
 */

/**
 * The approval card filed under a row. The row shows the decision and the card
 * leaves; or the turn ends first and the card leaves with no decision.
 * @typedef {object} CardPlacement
 * @property {(decision: Decision) => void} decide
 * @property {() => void} drop
 */

/**
 * A tool call's row, a plain call or a subagent's, as an approval card needs it.
 * A row carries one card at a time, filed directly under itself.
 * @typedef {object} ToolRow
 * @property {boolean} hasCard - a card is filed under the row now
 * @property {(card: HTMLElement) => CardPlacement} file
 */

/**
 * `refusal` is the line a refused write drew ({@link refusalLine}), or null:
 * the chip puts it in its own row, after the row's body, so it reads without
 * opening anything. A row that holds the chip folded away (a run, a subagent)
 * moves the line under itself.
 * @typedef {ToolRow & { el: HTMLElement, finish: (outcome: ToolOutcome) => void, refusal: HTMLElement | null }} ToolChip
 */

/**
 * How a subagent call ends: its own tool result (a foreground agent), or the
 * report a background agent sends later (`status`, `summary`).
 * @typedef {ToolOutcome & { status?: string, summary?: string }} AgentOutcome
 */

/**
 * @typedef {object} AgentChip
 * @property {HTMLElement} el
 * @property {string} name - the subagent type, e.g. `facility-knowledge`
 * @property {(event: WireEvent) => ToolChip} addTool - file one of its tool calls
 * @property {(text: string) => void} addText - file one of its messages
 * @property {(outcome: AgentOutcome) => void} finish
 * @property {() => boolean} running
 * @property {boolean} hasCard - a card is filed under the agent's own row now
 * @property {(card: HTMLElement) => CardPlacement} file
 */

/**
 * The line a refused write puts in the conversation: its phrase, in view
 * without expanding anything. Null when the outcome carries no refusal.
 * @param {ToolOutcome} outcome
 * @returns {HTMLElement | null}
 */
export function refusalLine(outcome) {
  const refusal = outcome.is_error ? outcome.refusal : null;
  const phrase = typeof refusal?.phrase === 'string' ? refusal.phrase.trim() : '';
  if (!refusal || !phrase) return null;
  const line = elem('p', 'op-tool-refusal', phrase);
  line.dataset.reason = String(refusal.reason ?? '');
  line.setAttribute('role', 'note');
  return line;
}

/** What a row's decision badge says. */
const DECISION_LABEL = Object.freeze({
  allowed: 'Allowed by you',
  session: 'Allowed for this session',
  denied: 'Denied by you',
  expired: 'No answer · denied',
});

/**
 * Where a row's approval card goes: directly under the row, and the decision
 * badge goes in the row's head, before its status. `openFor` takes whatever
 * holds the row in view while the card waits (a subagent's trace) and returns
 * the release.
 * @param {HTMLElement} el - the row
 * @param {HTMLElement} status - the row's status; the badge goes before it
 * @param {() => () => void} openFor
 * @returns {ToolRow}
 */
function cardSlot(el, status, openFor) {
  /** The card filed under the row, as the way to take it away again. */
  let filed = /** @type {(() => void) | null} */ (null);
  return {
    get hasCard() {
      return filed !== null;
    },
    file(card) {
      el.after(card);
      const release = openFor();
      const unfile = () => {
        card.remove();
        release();
        filed = null;
      };
      filed = unfile;
      return {
        decide(decision) {
          if (filed !== unfile) return;
          const badge = elem('span', 'op-tool-decision', DECISION_LABEL[decision]);
          badge.dataset.decision = decision;
          status.before(badge);
          unfile();
        },
        drop() {
          if (filed === unfile) unfile();
        },
      };
    },
  };
}

/** Nothing holds a top-level row in view: it is never folded away while a card waits. */
export const NO_HOLD = () => () => {};

/**
 * One tool call: category glyph, display name, a one-line input summary, and a
 * status that is a spinner until its result lands and then a check or a red x.
 * A click opens the full input and the result preview. A control-system write
 * is marked as one (`data-kind="control-write"`), so it reads heavier. A write
 * the hooks refused also shows its phrase under the row, outside what opens.
 * @param {WireEvent} event
 * @param {() => () => void} [openFor] - takes what holds the row in view while its card waits
 * @returns {ToolChip}
 */
export function createToolChip(event, openFor = NO_HOLD) {
  const raw = event.tool_name_raw ?? event.tool_name;
  const row = createRow({
    className: 'op-tool',
    kind: foldKey(raw),
    glyph: chatIcon(CATEGORY_ICON[toolCategory(raw)]),
    label: toolLabel(event),
    detail: summarizeInput(event.input),
  });
  const result = resultSection('Waiting for the result…');
  const inputBox = elem('div', 'op-tool-pre');
  inputBox.append(inputView(event.input));
  row.body.append(section('Input', inputBox), result.el);
  row.head.addEventListener('click', () => row.setOpen(!row.isOpen()));

  const slot = cardSlot(row.el, row.status, openFor);
  let refusal = /** @type {HTMLElement | null} */ (null);
  return {
    el: row.el,
    finish(outcome) {
      row.setState(outcome.is_error ? 'error' : 'done');
      const preview = outcome.preview ?? '';
      result.set(preview ? formatToolResult(preview) : '(no output)', {
        shortened: Boolean(preview && outcome.truncated),
        failed: Boolean(outcome.is_error),
      });
      refusal = refusalLine(outcome);
      if (refusal) row.el.append(refusal);
    },
    get refusal() {
      return refusal;
    },
    get hasCard() {
      return slot.hasCard;
    },
    file: slot.file,
  };
}

/**
 * A labelled block inside an opened row: a rounded box whose header cell
 * carries the title, and *meta* at its right edge when given.
 * @param {string} title
 * @param {HTMLElement} content
 * @param {HTMLElement} [meta]
 * @returns {HTMLElement}
 */
function section(title, content, meta) {
  const head = elem('div', 'op-tool-section-head');
  head.append(elem('span', 'op-tool-label', title));
  if (meta) head.append(meta);
  const body = elem('div', 'op-tool-section-body');
  body.append(content);
  const node = elem('div', 'op-tool-section');
  node.append(head, body);
  return node;
}

/** @typedef {{ shortened: boolean, failed: boolean }} ResultState */

/**
 * A call's Result section: its text in a preformatted block, with a quiet note
 * in the header when the server shortened the text. A failed call marks the
 * whole section.
 * @param {string} waiting - the text shown until the result lands
 * @returns {{ el: HTMLElement, set: (text: string, state: ResultState) => void }}
 */
function resultSection(waiting) {
  const pre = elem('pre', 'op-tool-pre', waiting);
  const note = elem('span', 'op-tool-note', 'shortened');
  note.hidden = true;
  const el = section('Result', pre, note);
  el.classList.add('op-tool-result');
  return {
    el,
    set(text, { shortened, failed }) {
      pre.textContent = text;
      pre.classList.toggle('op-tool-error', failed);
      if (failed) el.dataset.state = 'error';
      else delete el.dataset.state;
      note.hidden = !shortened;
    },
  };
}

/** The text a subagent call's result opens with when the agent went to the background. */
const ASYNC_LAUNCH_MARK = 'Async agent launched';

/**
 * One subagent call as a row: the agent glyph, the agent type, its task, and a
 * live note of what it is doing. Opened, it shows the agent's own tool calls
 * and messages, nested, so they never read as the main agent's. A background
 * agent stays running past its launch result until its report lands. A write
 * the hooks refused inside it shows its phrase under the agent's row, in call
 * order, so the closed trace does not hide it.
 * @param {WireEvent} event - the `Agent`/`Task` tool_use
 * @param {(el: HTMLElement, text: string) => void} renderText - markdown renderer
 * @param {() => void} onSettled - called once, when the agent finishes
 * @returns {AgentChip}
 */
export function createAgentChip(event, renderText, onSettled) {
  const input = event.input && typeof event.input === 'object' ? event.input : {};
  const name = String(input.subagent_type || 'agent');
  const row = createRow({
    className: 'op-tool op-agent',
    kind: 'agent',
    glyph: chatIcon('agent'),
    label: name,
    detail: oneLine(String(input.description || '')),
  });
  const note = elem('span', 'op-agent-note');
  row.status.before(note);

  const trace = elem('div', 'op-agent-trace');
  const empty = elem('span', 'op-agent-empty', 'Nothing yet.');
  trace.append(empty);
  const result = resultSection('Waiting for the agent…');
  row.body.append(section('Activity', trace), result.el);
  // The trace opens on the operator's word, and by itself while a card in it waits.
  let chosen = false;
  let holds = 0;
  function showTrace() {
    row.setOpen(holds > 0 || chosen);
  }
  /** Keep the trace open while a card under one of its rows waits; returns the release. */
  function holdTrace() {
    holds += 1;
    showTrace();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      holds -= 1;
      showTrace();
    };
  }
  row.head.addEventListener('click', () => {
    if (holds > 0) return;
    chosen = !chosen;
    showTrace();
  });

  /** The agent's own last message: its report, when the outcome carries none. */
  let lastText = '';
  let tools = 0;
  let latest = '';
  let phase = 'Working';
  let settled = false;
  /** The agent's own tool calls, in the order it made them. */
  const calls = /** @type {ToolChip[]} */ ([]);

  /** Its refused writes' lines, under the agent's row and outside its trace. */
  function showRefusals() {
    row.el.append(...calls.flatMap((call) => (call.refusal ? [call.refusal] : [])));
  }

  function renderNote() {
    const count = tools > 0 ? `${tools} tool call${tools === 1 ? '' : 's'}` : '';
    const live = settled ? '' : latest;
    note.textContent = [settled ? '' : phase, count, live].filter(Boolean).join(' · ');
  }
  renderNote();

  /** @param {AgentOutcome} outcome */
  function settle(outcome) {
    settled = true;
    const failed = Boolean(outcome.is_error) || outcome.status === 'failed';
    const stopped = outcome.status === 'stopped';
    row.setState(failed ? 'error' : stopped ? 'stopped' : 'done');
    const preview = outcome.preview ? formatToolResult(outcome.preview) : '';
    const text = outcome.summary || preview || lastText;
    result.set(text || (stopped ? 'Stopped.' : '(no report)'), {
      shortened: !outcome.summary && Boolean(outcome.truncated) && Boolean(outcome.preview),
      failed,
    });
    renderNote();
    onSettled();
  }

  const slot = cardSlot(row.el, row.status, NO_HOLD);

  return {
    el: row.el,
    name,
    addTool(toolEvent) {
      empty.remove();
      tools += 1;
      latest = String(toolEvent.tool_name || toolEvent.tool_name_raw || '');
      renderNote();
      const chip = createToolChip(toolEvent, holdTrace);
      trace.append(chip.el);
      calls.push(chip);
      return {
        el: chip.el,
        finish(outcome) {
          chip.finish(outcome);
          showRefusals();
        },
        get refusal() {
          return chip.refusal;
        },
        get hasCard() {
          return chip.hasCard;
        },
        file: chip.file,
      };
    },
    addText(text) {
      if (!text.trim()) return;
      empty.remove();
      lastText = text;
      const block = elem('div', 'op-agent-text osprey-md-rendered');
      renderText(block, text);
      trace.append(block);
    },
    finish(outcome) {
      if (settled) return;
      // A background agent's launch result is not its end: it reports later.
      if (!outcome.status && (outcome.preview ?? '').includes(ASYNC_LAUNCH_MARK)) {
        phase = 'In background';
        row.setState('background');
        renderNote();
        return;
      }
      settle(outcome);
    },
    running: () => !settled,
    get hasCard() {
      return slot.hasCard;
    },
    file: slot.file,
  };
}
