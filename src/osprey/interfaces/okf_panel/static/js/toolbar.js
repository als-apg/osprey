// @ts-check
/*
 * OKF Knowledge Panel — the panel toolbar: the breadcrumb and the
 * bundle-health chip.
 *
 * The toolbar (index.html) is the panel's one header: the contents toggle
 * (contents.js), the breadcrumb, the bundle-health chip and the search field
 * (app.js runs the search). The tile's ⋯ lists the panel's menu under "This
 * panel".
 *
 * The breadcrumb says where the reader is: "Browse" alone on the overview,
 * "Browse › group › concept" on a page. Browse opens the overview; the group
 * opens it at that group's section.
 *
 * The chip is the bundle check (/api/bundle_health): "Checking bundle" while
 * it runs, then "Bundle healthy", "N issues" or "Check failed". It is a
 * button that opens what is behind its label in an info popover: one row per
 * issue (concept, kind, detail), or why the check failed.
 */

import { icon } from '/design-system/js/icons.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { el, STRUCTURE_MARKER } from './helpers.js';

/** @typedef {{group: string | null, title: string}} Trail  a page: its group's label, and its title */
/** @typedef {{concept_id: string, kind: string, detail: string}} BundleWarning */
/**
 * The `/api/bundle_health` body.
 * @typedef {{ok?: boolean, total?: number, warnings?: BundleWarning[]}} BundleHealth
 */
/**
 * @typedef {{state: 'checking'}
 *   | {state: 'checked', health: BundleHealth}
 *   | {state: 'failed', reason: string}} HealthCheck
 */
/**
 * @typedef {{tone: 'connecting' | 'ok' | 'warn' | 'err', label: string,
 *   detail: import('/design-system/js/info-popover.js').InfoContent | null}} ChipView
 */

const TONES = /** @type {const} */ (['connecting', 'ok', 'warn', 'err']);
const POPOVER_TITLE = 'Bundle health';

/** @type {HTMLElement | null} */
let crumbEl = null;
/** @type {HTMLButtonElement | null} */
let chipEl = null;
/** @type {ChipView | null} */
let chipView = null;
/** @type {(group: string | null) => void} */
let onBrowse = () => {};

/**
 * Wire the toolbar. Called once at boot.
 * @param {{onBrowse: (group: string | null) => void}} handles `onBrowse` opens
 *   the overview, at the named group's section when one is given
 */
export function initToolbar(handles) {
  crumbEl = document.getElementById('crumb');
  chipEl = /** @type {HTMLButtonElement | null} */ (document.getElementById('bundle-health'));
  onBrowse = handles.onBrowse;
  const field = document.querySelector('.osprey-search-field');
  if (field && !field.querySelector('.osprey-ic')) field.prepend(icon('search'));
  chipEl?.addEventListener('click', () => {
    if (chipEl && chipView?.detail) openInfoPopover(chipEl, chipView.detail);
  });
}

/**
 * A crumb that opens the overview.
 * @param {string} label
 * @param {string | null} group the group whose section to show; null for the top
 * @returns {HTMLElement}
 */
function crumbLink(label, group) {
  const link = el('a', { class: 'osprey-crumb-root', href: '#' + STRUCTURE_MARKER, text: label });
  link.addEventListener('click', (ev) => {
    ev.preventDefault();
    onBrowse(group);
  });
  return link;
}

/** @returns {HTMLElement} */
function crumbSep() {
  return el('span', { class: 'osprey-crumb-sep', 'aria-hidden': 'true' }, [icon('chevron-right')]);
}

/**
 * @param {string} text
 * @returns {HTMLElement}
 */
function crumbLeaf(text) {
  return el('span', { class: 'osprey-crumb-leaf', 'aria-current': 'page', title: text, text });
}

/**
 * Show where the reader is.
 * @param {Trail | null} trail the open page; null on the overview
 */
export function showCrumb(trail) {
  if (!crumbEl) return;
  if (!trail) {
    crumbEl.replaceChildren(crumbLeaf('Browse'));
    return;
  }
  const parts = [crumbLink('Browse', null)];
  if (trail.group) parts.push(crumbSep(), crumbLink(trail.group, trail.group));
  parts.push(crumbSep(), crumbLeaf(trail.title));
  crumbEl.replaceChildren(...parts);
}

/**
 * "broken-link" reads "Broken link".
 * @param {string} kind
 * @returns {string}
 */
function kindLabel(kind) {
  const words = String(kind || 'issue').replace(/[-_]+/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * What the chip shows for a bundle check, and the detail it opens.
 * @param {HealthCheck} check
 * @returns {ChipView}
 */
export function healthChipView(check) {
  if (check.state === 'checking') {
    return { tone: 'connecting', label: 'Checking bundle', detail: null };
  }
  if (check.state === 'failed') {
    return {
      tone: 'err',
      label: 'Check failed',
      detail: { title: POPOVER_TITLE, text: 'The bundle check failed: ' + check.reason + '.' },
    };
  }
  const warnings = check.health.warnings || [];
  if (check.health.ok) {
    return {
      tone: 'ok',
      label: 'Bundle healthy',
      detail: {
        title: POPOVER_TITLE,
        text: 'No issues: every page has complete frontmatter, and every link between pages resolves.',
      },
    };
  }
  const total = check.health.total ?? warnings.length;
  return {
    tone: 'warn',
    label: total + (total === 1 ? ' issue' : ' issues'),
    detail: {
      title: POPOVER_TITLE,
      rows: warnings.map((w) => ({
        label: w.concept_id,
        value: kindLabel(w.kind) + ': ' + w.detail,
      })),
    },
  };
}

/**
 * Show a bundle check on the chip. Until the check has an answer the chip
 * has no detail, so it is disabled.
 * @param {HealthCheck} check
 */
export function showBundleHealth(check) {
  if (!chipEl) return;
  const view = healthChipView(check);
  chipView = view;
  for (const tone of TONES) chipEl.classList.toggle('osprey-chip--' + tone, tone === view.tone);
  const label = chipEl.querySelector('.osprey-chip-label') ?? chipEl;
  label.textContent = view.label;
  chipEl.disabled = view.detail === null;
}
