// @ts-check
/**
 * OSPREY Web Terminal — operator chat: the turn's actions.
 *
 * Where a turn's tool and subagent rows go, and how consecutive tool calls of
 * one kind fold into a run: "Read 4 channels", one row with a short stack of
 * glyphs, that opens to the calls themselves. A run grows while calls of its
 * kind keep coming and breaks on anything else: a call of another kind, a
 * subagent, reasoning, text, the plan, a question — or an approval card, which
 * takes its call's row out of the run so the card, and the answer the row then
 * carries, stay in view. A run of one is the call's own row. A refused write's
 * line (`refusalLine` in chat-tool-cards.js) sits under the row
 * that shows the call: its own, or the run's while the call is folded into it,
 * one line per refused call in call order, so folding never hides it.
 *
 * The module also keeps the turn's one pulse: of the rows still running, the
 * one that started last breathes, and the rest hold the live colour. And it
 * counts what the turn did, for the footer.
 *
 * @module chat-tool-runs
 */

import { createAgentChip, createToolChip } from './chat-tool-cards.js';
import { createRow, iconStack } from './chat-row.js';
import { chatIcon } from './chat-icons.js';
import { elem } from './chat-dom.js';
import {
  CATEGORY_ICON,
  foldKey,
  oneLine,
  ranByOperator,
  runPhrase,
  summarizeInput,
  toolCategory,
  toolKey,
} from './chat-tool-vocab.js';

/** @typedef {import('./chat-dom.js').WireEvent} WireEvent */
/** @typedef {import('./chat-tool-cards.js').ToolChip} ToolChip */
/** @typedef {import('./chat-tool-cards.js').AgentChip} AgentChip */
/** @typedef {import('./chat-tool-vocab.js').FoldKey} FoldKey */
/** @typedef {import('./chat-icons.js').ChatIconName} ChatIconName */
/** @typedef {import('./chat-row.js').RowState} RowState */

/**
 * One tool call, as the runs see it.
 * @typedef {object} Call
 * @property {HTMLElement} el - the call's row
 * @property {FoldKey} key
 * @property {ChatIconName} icon
 * @property {string} tool - the call's {@link toolKey}
 * @property {string} summary - its one-line input
 * @property {boolean} byOperator - the operator ran it by a slash command
 * @property {RowState} state
 * @property {Run | null} run - the run row holding it, once folded
 * @property {HTMLElement | null} refusal - its refused write's line, once its result says so
 */

/**
 * A run row and the calls folded into it, in order.
 * @typedef {object} Run
 * @property {HTMLElement} el
 * @property {FoldKey} key
 * @property {Call[]} calls
 * @property {(call: Call) => void} adopt - move a call's row into the run
 * @property {() => void} render
 */

/**
 * What the turn did, for the footer.
 * @typedef {object} TurnStats
 * @property {number} actions - tool and subagent calls
 * @property {number} approvals - approvals asked of the operator
 * @property {ChatIconName[]} icons - the actions' glyphs, in order of first use
 */

/**
 * @typedef {object} TurnActions
 * @property {(event: WireEvent) => ToolChip} addTool - place a tool call's row,
 *   folded into the open run when it is of the same kind
 * @property {(event: WireEvent, renderText: (el: HTMLElement, text: string) => void) => AgentChip} addAgent
 *   - place a subagent's row; it breaks the run and stands on its own
 * @property {() => void} breakRun - something else joined the turn: the next
 *   call starts a run of its own
 * @property {(el: HTMLElement) => void} live - a row outside the runs (reasoning) started
 * @property {(el: HTMLElement) => void} settled - … and ended
 * @property {() => void} countApproval
 * @property {() => TurnStats} stats
 */

/**
 * The actions of one agent turn, placed in its entry.
 * @param {HTMLElement} entry - the turn's agent entry; rows are appended to it
 * @returns {TurnActions}
 */
export function createTurnActions(entry) {
  /** The run the next call may join; with one call it is still a bare row. */
  let open = /** @type {{ key: FoldKey, calls: Call[], run: Run | null } | null} */ (null);
  /** What is running now, in start order, each with the row that shows it. */
  const running = /** @type {{ id: unknown, row: () => HTMLElement }[]} */ ([]);
  /** The row carrying `pulse`. */
  let pulsing = /** @type {HTMLElement | null} */ (null);
  let actions = 0;
  let approvals = 0;
  /** @type {ChatIconName[]} */
  const icons = [];

  function pulse() {
    const last = running[running.length - 1];
    const target = last ? last.row() : null;
    if (pulsing === target) return;
    pulsing?.classList.remove('pulse');
    target?.classList.add('pulse');
    pulsing = target;
  }

  /**
   * @param {unknown} id
   * @param {() => HTMLElement} row
   */
  function start(id, row) {
    running.push({ id, row });
    pulse();
  }

  /** @param {unknown} id */
  function end(id) {
    const i = running.findIndex((item) => item.id === id);
    if (i >= 0) running.splice(i, 1);
    pulse();
  }

  /** @param {ChatIconName} icon */
  function noteIcon(icon) {
    if (!icons.includes(icon)) icons.push(icon);
  }

  /**
   * A run row: a stack of its calls' glyphs, the phrase for the run, the calls'
   * inputs dimmed, and the status of the run as a whole. It opens to the calls.
   * @param {FoldKey} key
   * @returns {Run}
   */
  function createRun(key) {
    const row = createRow({ className: 'op-run', kind: 'run', glyph: [], label: '' });
    const chevron = chatIcon('chevron');
    chevron.classList.add('op-row-chevron');
    row.status.before(chevron);
    const failedNote = elem('span', 'op-run-failed');
    failedNote.hidden = true;
    row.label.after(failedNote);
    const list = elem('div', 'op-run-list');
    row.body.append(list);
    row.head.addEventListener('click', () => row.setOpen(!row.isOpen()));
    /** @type {Call[]} */
    const calls = [];
    /** @type {Run} */
    const run = {
      el: row.el,
      key,
      calls,
      adopt(call) {
        calls.push(call);
        call.run = run;
        list.append(call.el);
        run.render();
      },
      render() {
        row.label.textContent = runPhrase(
          key,
          calls.map((call) => call.tool),
          calls.every((call) => call.byOperator)
        );
        row.detail.textContent = oneLine(calls.map((call) => call.summary).join(', '));
        row.glyph.replaceChildren(...iconStack(calls[0]?.icon ?? 'wrench', calls.length));
        row.glyph.classList.toggle('op-row-stack', calls.length > 1);
        const failed = calls.filter((call) => call.state === 'error').length;
        failedNote.hidden = failed === 0;
        failedNote.textContent = failed > 0 ? `${failed} failed` : '';
        const busy = calls.some((call) => call.state === 'running');
        row.setState(busy ? 'running' : failed > 0 ? 'error' : 'done');
        // Its calls' refused writes, under the run row rather than in its folded list.
        row.el.append(...calls.flatMap((call) => (call.refusal ? [call.refusal] : [])));
      },
    };
    return run;
  }

  /**
   * A call standing as its own row again: its refused write's line goes back
   * under that row.
   * @param {Call} call
   * @returns {HTMLElement}
   */
  function alone(call) {
    if (call.refusal) call.el.append(call.refusal);
    return call.el;
  }

  /**
   * Draw calls again after a split: two or more fold into a new run, one
   * stands as its own row.
   * @param {Call[]} calls
   * @returns {{ nodes: HTMLElement[], run: Run | null }}
   */
  function regroup(calls) {
    if (calls.length < 2) return { nodes: calls.map(alone), run: null };
    const run = createRun(calls[0].key);
    for (const call of calls) run.adopt(call);
    return { nodes: [run.el], run };
  }

  /**
   * Take a call out of its run, for the approval card the call is about to
   * carry: the calls before it keep their run, and the calls after it keep
   * theirs. Either side with one call left is that call's own row again. The
   * run is broken at the call: a later call starts a new run.
   * @param {Call} call
   */
  function eject(call) {
    const run = call.run;
    if (!run) {
      if (open?.calls.includes(call)) open = null;
      return;
    }
    const at = run.calls.indexOf(call);
    const wasOpen = open?.run === run;
    for (const member of run.calls) member.run = null;
    const before = regroup(run.calls.slice(0, at));
    const after = regroup(run.calls.slice(at + 1));
    run.el.replaceWith(...before.nodes, alone(call), ...after.nodes);
    open =
      wasOpen && after.nodes.length > 0
        ? { key: run.key, calls: run.calls.slice(at + 1), run: after.run }
        : null;
    pulse();
  }

  /**
   * Place a call's row: into the open run when it is of the same kind, else
   * as a row of its own that opens a run. A control-system write never folds:
   * it stands alone, and the next call starts afresh.
   * @param {Call} call
   */
  function place(call) {
    if (call.key === 'control-write') {
      open = null;
      entry.appendChild(call.el);
      return;
    }
    if (open && open.key === call.key) {
      if (!open.run) {
        const first = open.calls[0];
        open.run = createRun(call.key);
        first.el.replaceWith(open.run.el);
        open.run.adopt(first);
      }
      open.calls.push(call);
      open.run.adopt(call);
    } else {
      open = { key: call.key, calls: [call], run: null };
      entry.appendChild(call.el);
    }
  }

  return {
    addTool(event) {
      const raw = event.tool_name_raw ?? event.tool_name;
      const chip = createToolChip(event);
      /** @type {Call} */
      const call = {
        el: chip.el,
        key: foldKey(raw),
        icon: CATEGORY_ICON[toolCategory(raw)],
        tool: toolKey(raw),
        summary: summarizeInput(event.input),
        byOperator: ranByOperator(event.input),
        state: 'running',
        run: null,
        refusal: null,
      };
      actions += 1;
      noteIcon(call.icon);
      place(call);
      start(call, () => call.run?.el ?? call.el);
      let done = false;
      return {
        el: chip.el,
        finish(outcome) {
          if (done) return;
          done = true;
          chip.finish(outcome);
          call.state = outcome.is_error ? 'error' : 'done';
          call.refusal = chip.refusal;
          end(call);
          call.run?.render();
        },
        get refusal() {
          return chip.refusal;
        },
        get hasCard() {
          return chip.hasCard;
        },
        file(card) {
          eject(call);
          return chip.file(card);
        },
      };
    },
    addAgent(event, renderText) {
      open = null;
      /** @type {AgentChip} */
      const chip = createAgentChip(event, renderText, () => end(chip));
      actions += 1;
      noteIcon('agent');
      entry.appendChild(chip.el);
      start(chip, () => chip.el);
      return chip;
    },
    breakRun() {
      open = null;
    },
    live(el) {
      start(el, () => el);
    },
    settled(el) {
      end(el);
    },
    countApproval() {
      approvals += 1;
    },
    stats: () => ({ actions, approvals, icons: [...icons] }),
  };
}
