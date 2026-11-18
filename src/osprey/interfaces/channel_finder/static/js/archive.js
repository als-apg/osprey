// @ts-check
/**
 * OSPREY Channel Finder — Archive
 *
 * The view of `#/archive` (Panel view contract): what the current target's
 * archive holds, read from the archive route alone. It claims nothing in the
 * toolbar. Read once on mount and on Refresh; the newest-sample age is the
 * clock's at that read. Nothing here writes: not the archive, not its
 * configuration. The markup is archive-render.js.
 */

import { copyText } from '/design-system/js/clipboard.js';
import { icon } from '/design-system/js/icons.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { renderState } from '/design-system/js/panel-state.js';
import { showToast } from '/design-system/js/toast.js';
import { configurationText, pageHtml, skeletonHtml } from './archive-render.js';
import { fetchArchiveSummary } from './directory-api.js';
import { NO_TARGET } from './pv-page-sections.js';
import { messageOf } from './utils.js';

/** @typedef {import('./directory-api.js').ArchiveSummary} ArchiveSummary */

/**
 * The mounted view; null while nothing is mounted.
 * @typedef {{
 *   root: HTMLElement,
 *   summary: ArchiveSummary|null,
 *   seq: number,
 *   controller: AbortController,
 * }} View
 */

/** @type {View|null} */
let view = null;

/**
 * @param {string} selector
 * @returns {HTMLElement}
 */
function part(selector) {
  return /** @type {HTMLElement} */ (view?.root.querySelector(selector));
}

/**
 * @param {HTMLElement} container
 * @returns {void}
 */
export function mountArchive(container) {
  unmountArchive();
  container.innerHTML = skeletonHtml();
  const root = /** @type {HTMLElement} */ (container.querySelector('.archive'));
  const controller = new AbortController();
  view = { root, summary: null, seq: 0, controller };
  root.addEventListener('click', onClick, { signal: controller.signal });
  void read();
}

/** @returns {void} */
export function unmountArchive() {
  if (!view) return;
  const { controller } = view;
  view = null;
  controller.abort();
}

/** Read the summary; a late answer for a view that is gone, or an older read, is dropped. */
async function read() {
  if (!view) return;
  const current = view;
  const seq = ++current.seq;
  if (!current.summary) showState({ icon: 'database', message: 'Reading the archive' });
  /** @type {ArchiveSummary} */
  let summary;
  try {
    summary = await fetchArchiveSummary(current.controller.signal);
  } catch (err) {
    if (view !== current || seq !== current.seq) return;
    current.summary = null;
    showState({
      icon: 'triangle-alert',
      message: 'Archive summary not available',
      detail: messageOf(err),
      action: { label: 'Retry', onClick: () => void read() },
    });
    return;
  }
  if (view !== current || seq !== current.seq) return;
  if (summary.outcome === 'unavailable') {
    current.summary = null;
    const message =
      summary.reason === NO_TARGET ? 'Not available for this target' : 'Archive not available';
    showState({
      icon: 'database',
      message,
      detail: summary.reason === NO_TARGET ? undefined : summary.reason,
      action: { label: 'Refresh', onClick: () => void read() },
    });
    return;
  }
  current.summary = summary;
  render(summary);
}

/**
 * Show a centred state in place of the page.
 * @param {import('/design-system/js/panel-state.js').PanelState} state
 */
function showState(state) {
  const page = part('.archive-page');
  page.hidden = true;
  page.replaceChildren();
  const host = part('.archive-state');
  host.hidden = false;
  renderState(host, state);
}

/**
 * Draw the page of `summary`, its icons placed after the markup.
 * @param {ArchiveSummary} summary
 */
function render(summary) {
  const host = part('.archive-state');
  host.hidden = true;
  host.replaceChildren();
  const page = part('.archive-page');
  page.hidden = false;
  page.innerHTML = pageHtml(summary, Date.now());
  page.querySelector('[data-action="refresh"]')?.prepend(icon('refresh'));
  page.querySelector('[data-action="copy"]')?.append(icon('copy'));
  page.querySelector('[data-action="source"]')?.append(icon('info'));
}

/** @param {MouseEvent} event */
function onClick(event) {
  const target = /** @type {HTMLElement|null} */ (
    /** @type {HTMLElement} */ (event.target).closest('[data-action]')
  );
  if (!target || !view) return;
  const summary = view.summary;
  switch (target.dataset.action) {
    case 'refresh':
      void read();
      return;
    case 'copy':
      if (!summary) return;
      void copyText(configurationText(summary.configuration)).then((copied) =>
        showToast(copied ? 'Configuration copied' : 'Copy failed', {
          kind: copied ? 'success' : 'error',
        })
      );
      return;
    case 'source':
      if (!summary) return;
      openInfoPopover(target, {
        title: 'Archive',
        rows: [
          { label: 'Archiver', value: summary.source.label },
          { label: 'Target', value: summary.source.target },
          { label: 'Rehearsal', value: summary.source.rehearsal ? 'yes' : 'no' },
        ],
      });
      return;
    default:
  }
}
