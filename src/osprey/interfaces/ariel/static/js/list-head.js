// @ts-check
/**
 * ARIEL List Pane head
 *
 * The list pane's head, built once by list.js: the standalone search box and
 * compose button, the mode chip, and the body-level Filters & Options popover
 * shell. A search's diagnostics paint on the chip as a dot, its tooltip and
 * accessible description, and the top of the popover. list.js fills the chip's
 * label and the popover's controls.
 */

import { renderDiagnosticsBar, diagnosticsLevel, diagnosticsText } from './components.js';
import { setState, getState } from './store.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { icon } from '/design-system/js/icons.js';

/** The mode chip's own tooltip, before any search issue is appended. */
const CHIP_TITLE = 'Search mode, filters and options';

/**
 * A search's diagnostics, off the result list: a dot on the mode chip, the
 * issue text as the chip's tooltip and accessible description, and the full
 * list at the top of the chip's popover, which the keyboard reaches through
 * the chip. An empty list clears all three.
 * @param {HTMLButtonElement} chip
 * @param {HTMLElement} popover - The Filters & Options popover
 * @param {import('./components.js').Diagnostic[]} diagnostics
 */
export function paintChipIssues(chip, popover, diagnostics) {
  const issues = /** @type {HTMLElement|null} */ (popover.querySelector('[data-search-issues]'));
  chip.querySelector('.ariel-mode-chip-dot')?.remove();
  if (diagnostics.length === 0) {
    chip.title = CHIP_TITLE;
    chip.removeAttribute('aria-description');
    if (issues) {
      issues.hidden = true;
      issues.innerHTML = '';
    }
    return;
  }
  const text = diagnosticsText(diagnostics);
  const dot = document.createElement('span');
  dot.className = 'ariel-mode-chip-dot';
  dot.dataset.level = diagnosticsLevel(diagnostics);
  dot.setAttribute('aria-hidden', 'true');
  chip.append(dot);
  chip.title = `${CHIP_TITLE}\n${text}`;
  chip.setAttribute('aria-description', text);
  if (issues) {
    issues.innerHTML = renderDiagnosticsBar(diagnostics, { open: true });
    issues.hidden = false;
  }
}

/**
 * The pane head: search box and compose button standalone, the mode chip always.
 * `paintChip` fills the chip's label.
 * @returns {{head: HTMLElement, input: HTMLInputElement|null, chip: HTMLButtonElement}}
 */
export function buildHead() {
  const embedded = isEmbedded();
  const head = document.createElement('div');
  head.className = 'osprey-pane-head ariel-list-head';

  /** @type {HTMLInputElement|null} */
  let input = null;
  if (!embedded) {
    const search = document.createElement('label');
    search.className = 'osprey-search ariel-list-search';
    search.append(icon('search'));
    input = document.createElement('input');
    input.type = 'search';
    input.id = 'ariel-search-input';
    input.placeholder = 'Search entries';
    input.setAttribute('aria-label', 'Search entries');
    input.autocomplete = 'off';
    input.value = getState().query ?? '';
    search.append(input);
    head.append(search);
  }

  const chip = document.createElement('button');
  chip.type = 'button';
  chip.className = 'osprey-btn ariel-mode-chip';
  chip.id = 'ariel-mode-chip';
  chip.title = CHIP_TITLE;
  chip.setAttribute('aria-haspopup', 'dialog');
  const chipLabel = document.createElement('span');
  chipLabel.className = 'osprey-btn-label ariel-mode-chip-label';
  chip.append(chipLabel, icon('chevron-down'));
  head.append(chip);

  if (!embedded) {
    const compose = document.createElement('button');
    compose.type = 'button';
    compose.className = 'osprey-icon-btn ariel-compose-btn';
    compose.id = 'ariel-compose-btn';
    compose.setAttribute('aria-label', 'New entry');
    compose.title = 'New entry';
    compose.append(icon('plus'));
    compose.addEventListener('click', () => setState({ compose: { draft: null } }));
    head.append(compose);
  }
  return { head, input, chip };
}

/**
 * Replace the body-level Filters & Options popover with a fresh, hidden one.
 * @returns {HTMLElement}
 */
export function buildFiltersPopover() {
  document.getElementById('ariel-filters-popover')?.remove();
  const popover = document.createElement('div');
  popover.className = 'osprey-popover ariel-filters-popover';
  popover.id = 'ariel-filters-popover';
  popover.setAttribute('role', 'dialog');
  popover.setAttribute('aria-label', 'Search mode, filters and options');
  popover.hidden = true;
  popover.innerHTML = `
    <div class="ariel-search-issues" data-search-issues hidden></div>
    <div class="ariel-mode-choices" data-mode-choices></div>
    <div class="advanced-sections ariel-filters" data-filters></div>
  `;
  document.body.append(popover);
  return popover;
}
