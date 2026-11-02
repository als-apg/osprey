// @ts-check
/**
 * OSPREY Web Terminal — operator chat: the task bar.
 *
 * The agent's task list for the conversation, pinned above the composer. It is
 * folded to one line by default: the count, the task in hand (or the next
 * one), and a progress bar. A click opens the whole checklist.
 *
 * The list is the conversation's, not a turn's: it carries over to the next
 * prompt, and it is redrawn from history on a reload, because a replay sends
 * the same frames a live turn does. It goes away when the log is cleared, or
 * when the operator sends a prompt after every task was done.
 *
 * Two tool styles feed it. `TodoWrite` sends the whole list each time.
 * `TaskCreate` adds one task and `TaskUpdate` changes one by id; both take
 * effect when their result lands, and only if the call succeeded. A created
 * task's id is read from the result (`Task #3 created successfully: …`), never
 * counted, so a lost frame cannot shift which task an update names. An update
 * names its task as `taskId` or `task_id` (the CLI takes both), or, failing
 * both, by the id its result reports (`Updated task #3 status`). Calls made
 * by a subagent are its own bookkeeping and are left out.
 *
 * @module chat-tasks
 */

import { chatIcon } from './chat-icons.js';
import { button, elem } from './chat-dom.js';
import { onChat } from './chat-events.js';
import { toolKey } from './chat-tool-vocab.js';

/** @typedef {'pending' | 'in_progress' | 'completed'} TaskStatus */

/**
 * @typedef {object} Task
 * @property {string} id
 * @property {string} subject - what the task is, as a list item
 * @property {string} activeForm - what the agent is doing while it is in hand
 * @property {TaskStatus} status
 */

/** The id a successful task call reports: `Task #<id> created …`, `Updated task #<id> …`. */
const REPORTED_ID = /#([A-Za-z0-9_-]+)/;

/**
 * @param {unknown} status
 * @returns {TaskStatus}
 */
function normaliseStatus(status) {
  return status === 'in_progress' || status === 'completed' ? status : 'pending';
}

/** @param {unknown} value */
function text(value) {
  return typeof value === 'string' ? value.trim() : '';
}

/**
 * The conversation's task list, built from the chat's frames. Pure: no DOM.
 */
export function createTaskList() {
  /** @type {Map<string, Task>} */
  const tasks = new Map();
  /** Task calls whose result has not landed yet, by call id. */
  const calls = /** @type {Map<string, { key: string, input: Record<string, any> }>} */ (new Map());
  /** Fallback ids for created tasks whose result named none. */
  let unnamed = 0;

  /** @param {unknown} todos - TodoWrite's whole list */
  function replace(todos) {
    tasks.clear();
    if (!Array.isArray(todos)) return;
    todos.forEach((item, i) => {
      if (!item || typeof item !== 'object') return;
      const subject = text(item.content) || text(item.activeForm);
      if (!subject) return;
      const id = `todo-${i}`;
      tasks.set(id, {
        id,
        subject,
        activeForm: text(item.activeForm),
        status: normaliseStatus(item.status),
      });
    });
  }

  /**
   * @param {Record<string, any>} input
   * @param {string} result
   */
  function create(input, result) {
    const subject = text(input.subject);
    if (!subject) return false;
    const id = REPORTED_ID.exec(result)?.[1] ?? `unnamed-${(unnamed += 1)}`;
    tasks.set(id, { id, subject, activeForm: text(input.activeForm), status: 'pending' });
    return true;
  }

  /**
   * @param {Record<string, any>} input
   * @param {string} result
   */
  function update(input, result) {
    const named = input.taskId ?? input.task_id;
    const id =
      named === undefined || named === null ? (REPORTED_ID.exec(result)?.[1] ?? '') : String(named);
    const task = tasks.get(id);
    if (!task) return false;
    if (input.status === 'deleted') {
      tasks.delete(id);
      return true;
    }
    if (input.status !== undefined) task.status = normaliseStatus(input.status);
    if (text(input.subject)) task.subject = text(input.subject);
    if (text(input.activeForm)) task.activeForm = text(input.activeForm);
    return true;
  }

  return {
    /**
     * Read one chat frame.
     * @param {Record<string, any>} event
     * @returns {boolean} whether the list changed
     */
    frame(event) {
      if (event.parent_tool_use_id) return false;
      const callId = event.tool_use_id === undefined ? '' : String(event.tool_use_id);
      if (event.type === 'tool_use') {
        const key = toolKey(event.tool_name_raw ?? event.tool_name);
        if (key === 'todowrite') {
          replace(event.input?.todos);
          return true;
        }
        if ((key === 'taskcreate' || key === 'taskupdate') && callId) {
          calls.set(callId, { key, input: event.input ?? {} });
        }
        return false;
      }
      if (event.type === 'tool_result') {
        const call = calls.get(callId);
        if (!call) return false;
        calls.delete(callId);
        if (event.is_error) return false;
        const result = String(event.preview ?? '');
        return call.key === 'taskcreate' ? create(call.input, result) : update(call.input, result);
      }
      return false;
    },
    /**
     * Drop the list if every task in it is done: a new prompt starts afresh.
     * @returns {boolean} whether the list changed
     */
    clearIfDone() {
      if (tasks.size === 0) return false;
      for (const task of tasks.values()) if (task.status !== 'completed') return false;
      tasks.clear();
      return true;
    },
    clear() {
      tasks.clear();
      calls.clear();
    },
    /** @returns {Task[]} the tasks, in the order they were made */
    items() {
      return [...tasks.values()].map((task) => ({ ...task }));
    },
  };
}

/**
 * The line the folded bar shows: the task in hand, else the next one, else
 * that everything is done.
 * @param {Task[]} tasks
 * @returns {string}
 */
export function taskSummary(tasks) {
  const current = tasks.find((task) => task.status === 'in_progress');
  if (current) return current.activeForm || current.subject;
  const next = tasks.find((task) => task.status === 'pending');
  if (next) return `Next: ${next.subject}`;
  return 'All done';
}

/** @param {TaskStatus} status */
function statusGlyph(status) {
  const glyph = elem('span', 'op-taskbar-glyph');
  glyph.append(chatIcon(status === 'completed' ? 'check' : 'circle'));
  return glyph;
}

let bars = 0;

/**
 * The pinned bar: hidden while the list is empty.
 */
export function createTaskBar() {
  const el = elem('section', 'op-taskbar');
  el.setAttribute('aria-label', 'Tasks');
  el.hidden = true;

  const listId = `op-taskbar-list-${(bars += 1)}`;
  const head = button('op-taskbar-head');
  head.setAttribute('aria-expanded', 'false');
  head.setAttribute('aria-controls', listId);
  const icon = elem('span', 'op-taskbar-icon');
  icon.append(chatIcon('tasks'));
  const count = elem('span', 'op-taskbar-count');
  const summary = elem('span', 'op-taskbar-summary');
  const track = elem('span', 'op-taskbar-track');
  const fill = elem('span', 'op-taskbar-fill');
  track.append(fill);
  const chevron = chatIcon('chevron');
  chevron.classList.add('op-taskbar-chevron');
  head.append(icon, elem('span', 'op-taskbar-title', 'Tasks'), count, summary, track, chevron);

  const list = elem('ul', 'op-taskbar-list');
  list.id = listId;
  list.hidden = true;
  el.append(head, list);

  /** @param {boolean} open */
  function setOpen(open) {
    el.classList.toggle('open', open);
    head.setAttribute('aria-expanded', String(open));
    list.hidden = !open;
  }
  head.addEventListener('click', () => setOpen(head.getAttribute('aria-expanded') !== 'true'));

  return {
    el,
    /** @param {Task[]} tasks */
    update(tasks) {
      if (tasks.length === 0) {
        el.hidden = true;
        setOpen(false);
        list.replaceChildren();
        return;
      }
      el.hidden = false;
      const done = tasks.filter((task) => task.status === 'completed').length;
      el.dataset.state =
        done === tasks.length
          ? 'done'
          : tasks.some((t) => t.status === 'in_progress')
            ? 'running'
            : 'pending';
      count.textContent = `${done}/${tasks.length}`;
      summary.textContent = taskSummary(tasks);
      fill.style.width = `${Math.round((done / tasks.length) * 100)}%`;
      list.replaceChildren(
        ...tasks.map((task) => {
          const row = elem('li', 'op-taskbar-item');
          row.dataset.status = task.status;
          const label =
            task.status === 'in_progress' ? task.activeForm || task.subject : task.subject;
          row.append(statusGlyph(task.status), elem('span', 'op-taskbar-text', label));
          return row;
        })
      );
    },
  };
}

/**
 * Pin the task bar above the composer and keep it on the chat's frames.
 * @param {import('./chat-features.js').ChatFeatureContext} ctx
 * @returns {() => void} teardown
 */
export function initChatTasks(ctx) {
  const tasks = createTaskList();
  const bar = createTaskBar();
  ctx.inputArea.before(bar.el);
  const draw = () => bar.update(tasks.items());
  const stops = [
    onChat('frame', ({ event }) => {
      if (tasks.frame(event)) draw();
    }),
    onChat('user-message', () => {
      if (tasks.clearIfDone()) draw();
    }),
    onChat('reset', () => {
      tasks.clear();
      draw();
    }),
  ];
  return () => {
    for (const stop of stops) stop();
    bar.el.remove();
  };
}
