// @ts-check
/* OSPREY Web Terminal — the Settings drawer's Review tab
 *
 * Shows the code review section as the next review reads it
 * (`GET /api/code-review`): on or off, the time limit, and the reviewer's
 * rules file. The tab is present whether review is on or off. The server
 * spells the warning for a section or rules file a review cannot use, so the
 * tab states the consequence the reviewer actually has rather than a guess.
 * Where the profile turns review on, its bar is code review's runtime safety
 * switch, drawn by the Safety tab's own row.
 */

import { fetchJSON } from './api.js';
import { ADMIN_LOCK_TEXT, switchRow } from './safety-switches.js';

/** Drawer-tab id of the Review tab (index.html: the tab button + panel). */
export const REVIEW_TAB_ID = 'tab-review';

/** The palette row's name for the reviewer's rules. */
export const RULES_ENTRY_NAME = 'code-review/rules';

/**
 * `GET /api/code-review`.
 * @typedef {object} CodeReviewState
 * @property {boolean | null} enabled  null when the section cannot be read
 * @property {string | null} level
 * @property {number | null} time_limit  seconds
 * @property {string | null} prompt  the rules file's path, as configured
 * @property {string | null} rules  the file's text, null when it cannot be used
 * @property {string | null} warning  what a review does meanwhile, and the fix
 * @property {boolean} can_change
 * @property {import('./safety-switches.js').SafetySwitch | null} switch  null while the profile leaves review off
 */

/** @type {HTMLElement | null} */
let section = null;

/**
 * Mount the Review tab: it re-reads the section each time it is opened, since
 * the reviewer reads it fresh for every review.
 */
export function initCodeReviewTab() {
  section = document.getElementById('code-review-section');
  const panel = document.getElementById(REVIEW_TAB_ID);
  if (!section || !panel) return;
  panel.addEventListener('drawer:tab-activate', () => void refreshCodeReviewTab());
}

/**
 * Re-read the section and re-render the tab.
 * @returns {Promise<void>}
 */
export async function refreshCodeReviewTab() {
  if (!section) return;
  /** @type {CodeReviewState} */
  let data;
  try {
    data = await fetchJSON('/api/code-review');
  } catch {
    section.replaceChildren(textEl('p', 'code-review-foot', 'Code review settings unavailable.'));
    return;
  }
  render(section, data);
}

/**
 * The palette rows for this tab: one, for the reviewer's rules, while the tab
 * is in the drawer.
 * @returns {import('./scaffold-gallery.js').GalleryEntry[]}
 */
export function reviewPaletteEntries() {
  const tab = document.querySelector(`#settings-drawer .drawer-tab[data-tab="${REVIEW_TAB_ID}"]`);
  if (!tab) return [];
  return [
    {
      name: RULES_ENTRY_NAME,
      label: 'Code review rules',
      category: '',
      tabId: REVIEW_TAB_ID,
      tabLabel: (tab.textContent || '').trim(),
      description: "The rules the code reviewer judges the agent's code against",
    },
  ];
}

/**
 * @param {string} tag
 * @param {string} className
 * @param {string} text
 * @returns {HTMLElement}
 */
function textEl(tag, className, text) {
  const el = document.createElement(tag);
  el.className = className;
  el.textContent = text;
  return el;
}

/**
 * The state line under the bar's label.
 * @param {boolean | null} enabled
 * @returns {string}
 */
export function stateText(enabled) {
  if (enabled === null) return 'Cannot be read';
  return enabled ? 'On' : 'Off';
}

/**
 * @param {HTMLElement} container
 * @param {CodeReviewState} data
 */
function render(container, data) {
  container.replaceChildren();
  if (data.warning) {
    const warning = textEl('p', 'drawer-notice code-review-warning', data.warning);
    warning.setAttribute('role', 'status');
    container.appendChild(warning);
  }
  const head = data.switch
    ? switchRow(data.switch, data.can_change, refreshCodeReviewTab)
    : bar(data);
  container.append(head, limitRow(data), ...rulesBlock(data));
}

/**
 * The bar where there is no switch: review is off in the profile, or the
 * section cannot be read. Nobody can move it here.
 * @param {CodeReviewState} data
 * @returns {HTMLElement}
 */
function bar(data) {
  const el = document.createElement('div');
  el.className = 'hook-debug-bar safety-switch-row';
  const text = document.createElement('div');
  text.className = 'safety-switch-text';
  text.append(
    textEl('span', 'hook-debug-label', 'Code review'),
    textEl(
      'div',
      'safety-switch-state code-review-state' + (data.enabled ? '' : ' off'),
      stateText(data.enabled)
    )
  );

  const right = document.createElement('span');
  right.className = 'safety-switch-lock';
  right.appendChild(textEl('span', '', lockText(data)));
  const toggleLabel = document.createElement('label');
  toggleLabel.className = 'toggle-switch';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = data.enabled === true;
  input.disabled = true;
  input.setAttribute('aria-label', 'Code review');
  const slider = document.createElement('span');
  slider.className = 'toggle-slider';
  toggleLabel.append(input, slider);
  right.appendChild(toggleLabel);

  el.append(text, right);
  return el;
}

/**
 * Who sets the section, beside the locked toggle.
 * @param {CodeReviewState} data
 * @returns {string}
 */
export function lockText(data) {
  return data.can_change ? '🔒 Set in the profile; rebuild to apply' : ADMIN_LOCK_TEXT;
}

/**
 * @param {CodeReviewState} data
 * @returns {HTMLElement}
 */
function limitRow(data) {
  const rows = document.createElement('div');
  rows.className = 'code-review-rows';
  const row = document.createElement('div');
  row.className = 'code-review-row';
  const value = data.time_limit === null ? '—' : `${data.time_limit} s`;
  row.append(
    textEl('span', 'code-review-key', 'Time limit per review'),
    textEl('span', 'code-review-value', value)
  );
  rows.appendChild(row);
  return rows;
}

/**
 * @param {CodeReviewState} data
 * @returns {HTMLElement[]}
 */
function rulesBlock(data) {
  const head = document.createElement('div');
  head.className = 'code-review-section-head';
  head.append(
    textEl('span', '', 'Reviewer rules'),
    textEl('span', 'prompts-badge read-only', 'READ-ONLY')
  );
  /** @type {HTMLElement[]} */
  const blocks = [head];
  if (data.prompt) blocks.push(textEl('div', 'code-review-path', data.prompt));

  const box = document.createElement('pre');
  box.className = 'code-review-rules';
  if (data.rules === null) {
    box.classList.add('missing');
    box.textContent = 'The rules file cannot be used.';
  } else {
    for (const [className, text] of rulesSpans(data.rules)) {
      box.append(textEl('span', className, text), '\n');
    }
  }
  blocks.push(box);
  blocks.push(textEl('p', 'code-review-foot', footText(data)));
  return blocks;
}

/**
 * The rules file as it is written, cut into spans: a heading line, a list item
 * with the indented lines that continue it, or any other line.
 * @param {string} rules
 * @returns {Array<[string, string]>}  [class, text] pairs
 */
export function rulesSpans(rules) {
  /** @type {Array<[string, string]>} */
  const spans = [];
  for (const line of rules.replace(/\n$/, '').split('\n')) {
    const last = spans[spans.length - 1];
    if (last && last[0] === 'code-review-rule' && /^\s+\S/.test(line)) {
      last[1] += `\n${line}`;
    } else if (line.startsWith('#')) {
      spans.push(['code-review-rules-heading', line]);
    } else {
      spans.push([/^[-*] /.test(line) ? 'code-review-rule' : '', line]);
    }
  }
  return spans;
}

/**
 * @param {CodeReviewState} data
 * @returns {string}
 */
export function footText(data) {
  if (!data.can_change) return 'Ask your deployment admin to change these rules.';
  const moves = data.switch
    ? 'Turning review off or back on is recorded in the audit trail with your name. '
    : '';
  return `${moves}Edit the rules file in the profile; the next review reads it.`;
}

/** Forget the mounted section; for tests. */
export function _resetCodeReviewTabForTest() {
  section = null;
}
