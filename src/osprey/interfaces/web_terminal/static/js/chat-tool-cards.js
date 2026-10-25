// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: tool calls.
 *
 * The tool-call group and what it holds: one chip per tool call, one chip per
 * subagent call with its own trace nested inside, and the group that folds the
 * chips away once the agent moves on. An approval card waiting on a call sits
 * under that call's chip, which carries the operator's answer once it is given.
 *
 * @module chat-tool-cards
 */

import { chatIcon } from './chat-icons.js';
import { formatToolResult, inputView } from './chat-format.js';
import { button, elem } from './chat-dom.js';
import { CATEGORY_ICON, oneLine, summarizeInput, toolCategory } from './chat-tool-vocab.js';

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
 * @typedef {ToolRow & { el: HTMLElement, finish: (outcome: ToolOutcome) => void }} ToolChip
 */

/**
 * How a subagent call ends: its own tool result (a foreground agent), or the
 * report a background agent sends later (`status`, `summary`).
 * @typedef {ToolOutcome & { status?: string, summary?: string }} AgentOutcome
 */

/**
 * @typedef {object} AgentChip
 * @property {string} name - the subagent type, e.g. `facility-knowledge`
 * @property {(event: WireEvent) => ToolChip} addTool - file one of its tool calls
 * @property {(text: string) => void} addText - file one of its messages
 * @property {(outcome: AgentOutcome) => void} finish
 * @property {() => boolean} running
 * @property {boolean} hasCard - a card is filed under the agent's own row now
 * @property {(card: HTMLElement) => CardPlacement} file
 */

/**
 * @typedef {object} ToolGroup
 * @property {HTMLElement} el
 * @property {(event: WireEvent) => ToolChip} addTool
 * @property {(event: WireEvent, renderText: (el: HTMLElement, text: string) => void) => AgentChip} addAgent
 * @property {() => void} settle - fold away, unless the operator chose its state
 *   or an agent or a waiting card in it is still working (it folds when the last one is done)
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

/**
 * A chip whose finish also hands a refused write's line to `show`.
 * @param {ToolChip} chip
 * @param {(line: HTMLElement) => void} show
 * @returns {ToolChip}
 */
function showingRefusal(chip, show) {
  return {
    el: chip.el,
    finish(outcome) {
      chip.finish(outcome);
      const line = refusalLine(outcome);
      if (line) show(line);
    },
    get hasCard() {
      return chip.hasCard;
    },
    file: chip.file,
  };
}

/** @returns {HTMLElement} */
function spinner() {
  return elem('span', 'op-spin');
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
 * badge goes in the row's header, before its status. `openFor` takes whatever
 * holds the row open while the card waits (its group, a subagent's trace) and
 * returns the release.
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

/**
 * One tool call: category icon, display name, a one-line input summary, and a
 * status that is a spinner until its result lands and then a check or a red x.
 * A click opens the full input and the result preview.
 * @param {WireEvent} event
 * @param {() => () => void} openFor - takes what holds the row open while its card waits
 * @returns {ToolChip}
 */
function createToolChip(event, openFor) {
  const name = String(event.tool_name || event.tool_name_raw || 'tool');
  const el = elem('div', 'op-tool');
  el.dataset.state = 'running';

  const head = button('op-tool-head');
  head.setAttribute('aria-expanded', 'false');
  const icon = chatIcon(CATEGORY_ICON[toolCategory(event.tool_name_raw ?? event.tool_name)]);
  icon.classList.add('op-tool-icon');
  const status = elem('span', 'op-tool-status');
  status.append(spinner());
  head.append(
    icon,
    elem('span', 'op-tool-name', name),
    elem('span', 'op-tool-input', summarizeInput(event.input)),
    status,
  );

  const detail = elem('div', 'op-tool-detail');
  detail.hidden = true;
  const result = resultBlock('Waiting for the result…');
  const inputBox = elem('div', 'op-tool-pre');
  inputBox.append(inputView(event.input));
  detail.append(section('Input', inputBox), section('Result', result.el));
  head.addEventListener('click', () => {
    const open = detail.hidden;
    detail.hidden = !open;
    el.classList.toggle('open', Boolean(open));
    head.setAttribute('aria-expanded', String(open));
  });
  el.append(head, detail);

  const slot = cardSlot(el, status, openFor);
  return {
    el,
    finish(outcome) {
      el.dataset.state = outcome.is_error ? 'error' : 'done';
      status.replaceChildren(chatIcon(outcome.is_error ? 'x' : 'check'));
      const preview = outcome.preview ?? '';
      result.set(preview ? formatToolResult(preview) : '(no output)', {
        shortened: Boolean(preview && outcome.truncated),
        failed: Boolean(outcome.is_error),
      });
    },
    get hasCard() {
      return slot.hasCard;
    },
    file: slot.file,
  };
}

/**
 * A labelled block inside an expanded chip.
 * @param {string} title
 * @param {HTMLElement} content
 * @returns {HTMLElement}
 */
function section(title, content) {
  const node = elem('div', 'op-tool-section');
  node.append(elem('span', 'op-tool-label', title), content);
  return node;
}

/** @typedef {{ shortened: boolean, failed: boolean }} ResultState */

/**
 * A call's RESULT: its text in a preformatted block, with a quiet note under it when the
 * server shortened the text.
 * @param {string} waiting - the text shown until the result lands
 * @returns {{ el: HTMLElement, set: (text: string, state: ResultState) => void }}
 */
function resultBlock(waiting) {
  const pre = elem('pre', 'op-tool-pre', waiting);
  const note = elem('span', 'op-tool-note', 'shortened');
  note.hidden = true;
  const el = elem('div', 'op-tool-result');
  el.append(pre, note);
  return {
    el,
    set(text, { shortened, failed }) {
      pre.textContent = text;
      pre.classList.toggle('op-tool-error', failed);
      note.hidden = !shortened;
    },
  };
}

/** The text a subagent call's result opens with when the agent went to the background. */
const ASYNC_LAUNCH_MARK = 'Async agent launched';

/**
 * One subagent call as a chip: the agent type, its task, and a live note of
 * what it is doing. Opened, it shows the agent's own tool calls and messages,
 * nested, so they never read as the main agent's. A background agent stays
 * running past its launch result until its report lands.
 * @param {WireEvent} event - the `Agent`/`Task` tool_use
 * @param {(el: HTMLElement, text: string) => void} renderText - markdown renderer
 * @param {() => void} onSettled - called once, when the agent finishes
 * @param {() => () => void} openGroup - takes the group open while a card in the agent waits; returns the release
 * @returns {{ el: HTMLElement } & AgentChip}
 */
function createAgentChip(event, renderText, onSettled, openGroup) {
  const input = event.input && typeof event.input === 'object' ? event.input : {};
  const name = String(input.subagent_type || 'agent');
  const el = elem('div', 'op-tool op-agent');
  el.dataset.state = 'running';

  const head = button('op-tool-head');
  head.setAttribute('aria-expanded', 'false');
  const icon = chatIcon('agent');
  icon.classList.add('op-tool-icon');
  const note = elem('span', 'op-agent-note');
  const status = elem('span', 'op-tool-status');
  status.append(spinner());
  head.append(
    icon,
    elem('span', 'op-tool-name', name),
    elem('span', 'op-tool-input', oneLine(String(input.description || ''))),
    note,
    status,
  );

  const trace = elem('div', 'op-agent-trace');
  const empty = elem('span', 'op-agent-empty', 'Nothing yet.');
  trace.append(empty);
  const result = resultBlock('Waiting for the agent…');
  const detail = elem('div', 'op-tool-detail');
  detail.hidden = true;
  detail.append(section('Activity', trace), section('Result', result.el));
  // The trace opens on the operator's word, and by itself while a card in it waits.
  let chosen = false;
  let holds = 0;
  function showTrace() {
    const open = holds > 0 || chosen;
    detail.hidden = !open;
    el.classList.toggle('open', open);
    head.setAttribute('aria-expanded', String(open));
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
  head.addEventListener('click', () => {
    if (holds > 0) return;
    chosen = !chosen;
    showTrace();
  });
  el.append(head, detail);

  /** The agent's own last message: its report, when the outcome carries none. */
  let lastText = '';
  let tools = 0;
  let latest = '';
  let phase = 'Working';
  let settled = false;

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
    el.dataset.state = failed ? 'error' : stopped ? 'stopped' : 'done';
    status.replaceChildren(chatIcon(failed ? 'x' : stopped ? 'stop' : 'check'));
    const preview = outcome.preview ? formatToolResult(outcome.preview) : '';
    const text = outcome.summary || preview || lastText;
    result.set(text || (stopped ? 'Stopped.' : '(no report)'), {
      shortened: !outcome.summary && Boolean(outcome.truncated) && Boolean(outcome.preview),
      failed,
    });
    renderNote();
    onSettled();
  }

  const slot = cardSlot(el, status, openGroup);

  return {
    el,
    name,
    addTool(toolEvent) {
      empty.remove();
      tools += 1;
      latest = String(toolEvent.tool_name || toolEvent.tool_name_raw || '');
      renderNote();
      const chip = createToolChip(toolEvent, () => {
        const releaseTrace = holdTrace();
        const releaseGroup = openGroup();
        return () => {
          releaseTrace();
          releaseGroup();
        };
      });
      trace.append(chip.el);
      return chip;
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
        el.dataset.state = 'background';
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

/**
 * A run of consecutive tool calls as one collapsible row. It stays open while
 * the turn works through it and folds itself away once the agent moves on —
 * unless the operator opened or closed it, and then the state is theirs. An
 * approval card waiting in it holds it open. A subagent counts on its own, so
 * the folded row still says what it did: "1 agent finished · 3 tool calls".
 * A write the hooks refused, its own or a subagent's, puts its phrase under
 * the group's list, where folding does not hide it.
 * @returns {ToolGroup}
 */
export function createToolGroup() {
  const el = elem('div', 'op-tools open');
  const head = button('op-tools-head');
  head.setAttribute('aria-expanded', 'true');
  const chevron = chatIcon('chevron');
  chevron.classList.add('op-tools-chevron');
  const label = elem('span', 'op-tools-label');
  const status = elem('span', 'op-tools-status');
  head.append(chevron, label, status);
  const list = elem('div', 'op-tools-list');
  const refusals = elem('div', 'op-tools-refusals');
  refusals.hidden = true;
  el.append(head, list, refusals);

  /** @param {HTMLElement} line - a refused write's phrase */
  function showRefusal(line) {
    refusals.hidden = false;
    refusals.append(line);
  }

  let total = 0;
  let running = 0;
  let failed = 0;
  let agents = 0;
  let agentsRunning = 0;
  let open = true;
  /** Whether the turn has moved on from this group. */
  let settled = false;
  /** Whether the operator opened or closed it; then `chosen` is their state. */
  let operatorChose = false;
  let chosen = true;
  /** Cards waiting in the group, and whatever else holds it open while they wait. */
  let holds = 0;

  function render() {
    const parts = [];
    if (agents > 0) {
      const noun = `agent${agents === 1 ? '' : 's'}`;
      parts.push(agentsRunning > 0 ? `${agents} ${noun} working` : `${agents} ${noun} finished`);
    }
    if (total > 0 || agents === 0) parts.push(`${total} tool call${total === 1 ? '' : 's'}`);
    if (failed > 0) parts.push(`${failed} failed`);
    label.textContent = parts.join(' · ');
    const busy = running + agentsRunning > 0;
    status.dataset.state = busy ? 'running' : failed > 0 ? 'failed' : 'done';
    status.replaceChildren(busy ? spinner() : chatIcon(failed > 0 ? 'x' : 'check'));
  }

  /** @param {boolean} on */
  function setOpen(on) {
    open = on;
    el.classList.toggle('open', on);
    head.setAttribute('aria-expanded', String(on));
    list.hidden = !on;
  }

  /**
   * The group's state: open while a card in it waits; the operator's own choice
   * once they made one; otherwise folded once the turn has moved on and nothing
   * in it is still running.
   */
  function apply() {
    if (holds > 0) setOpen(true);
    else if (operatorChose) setOpen(chosen);
    else setOpen(!(settled && agentsRunning === 0));
  }

  /** Keep the group open while a card in it waits; returns the release. */
  function holdOpen() {
    holds += 1;
    apply();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      holds -= 1;
      apply();
    };
  }

  head.addEventListener('click', () => {
    if (holds > 0) return;
    operatorChose = true;
    chosen = !open;
    apply();
  });
  render();

  return {
    el,
    addTool(event) {
      total += 1;
      running += 1;
      render();
      const chip = showingRefusal(createToolChip(event, holdOpen), showRefusal);
      list.append(chip.el);
      let done = false;
      return {
        el: chip.el,
        finish(outcome) {
          if (done) return;
          done = true;
          running -= 1;
          if (outcome.is_error) failed += 1;
          chip.finish(outcome);
          render();
        },
        get hasCard() {
          return chip.hasCard;
        },
        file: chip.file,
      };
    },
    addAgent(event, renderText) {
      agents += 1;
      agentsRunning += 1;
      render();
      const chip = createAgentChip(event, renderText, () => {
        agentsRunning -= 1;
        if (chip.el.dataset.state === 'error') failed += 1;
        render();
        apply();
      }, holdOpen);
      // The agent's trace is collapsed; its refused writes surface on the group.
      const addNested = chip.addTool;
      chip.addTool = (toolEvent) => showingRefusal(addNested(toolEvent), showRefusal);
      list.append(chip.el);
      return chip;
    },
    settle() {
      // An agent or a card still waiting keeps the group in view; it folds once they are done.
      settled = true;
      apply();
    },
  };
}
