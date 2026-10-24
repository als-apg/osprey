// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: plan card.
 *
 * The turn's plan: one card, rewritten in place each time the agent updates its
 * todo list.
 *
 * @module chat-plan
 */

import { chatIcon } from './chat-icons.js';
import { elem } from './chat-dom.js';

/**
 * @typedef {{ content?: string, activeForm?: string, status?: string }} PlanItem
 */

/**
 * @typedef {object} TaskCard
 * @property {HTMLElement} el
 * @property {(todos: PlanItem[]) => void} update
 */

/**
 * The turn's plan: one card, rewritten in place each time the agent updates its
 * todo list. A k/n counter sits in the header.
 * @returns {TaskCard}
 */
export function createTaskCard() {
  const el = elem('section', 'op-tasks');
  el.setAttribute('aria-label', 'Tasks');
  const head = elem('header', 'op-tasks-head');
  head.append(elem('span', 'op-tasks-title', 'Tasks'), elem('span', 'op-tasks-count'));
  const count = /** @type {HTMLElement} */ (head.querySelector('.op-tasks-count'));
  const list = elem('ul', 'op-task-list');
  el.append(head, list);

  return {
    el,
    update(todos) {
      let done = 0;
      const rows = todos.map((item) => {
        const status = normaliseStatus(item.status);
        if (status === 'completed') done += 1;
        const row = elem('li', 'op-task');
        row.dataset.status = status;
        const glyph = elem('span', 'op-task-glyph');
        if (status === 'completed') glyph.append(chatIcon('check'));
        const text =
          status === 'in_progress'
            ? item.activeForm || item.content || ''
            : item.content || item.activeForm || '';
        row.append(glyph, elem('span', 'op-task-text', text));
        return row;
      });
      list.replaceChildren(...rows);
      count.textContent = `${done}/${todos.length}`;
    },
  };
}

/**
 * @param {string | undefined} status
 * @returns {'pending' | 'in_progress' | 'completed'}
 */
function normaliseStatus(status) {
  return status === 'in_progress' || status === 'completed' ? status : 'pending';
}
