// @ts-check
/**
 * ARIEL reader "Enhancements" tab: the enhancement inspector.
 *
 * A reader-seam tab beside "Entry", opened by its button or the `e` key. It
 * fetches `/entries/{id}/enhancements` only when the tab opens, and shows one
 * card per enabled module in execution order: the server's label, what the
 * module feeds, a state chip, its times, marker, error and attempts, and a
 * detail block for a module this file knows. Any other module gets the
 * status-only card, whatever its `detail` holds.
 *
 * A card with a `command` (a failed or stale module) shows it with a copy
 * button; a stale `text_embedding` card adds that `reembed` rebuilds the
 * whole table. The inspector never re-runs anything itself, and no embedding
 * vector is rendered: the detail blocks name the fields they show.
 */

import { showToast } from '/design-system/js/toast.js';
import { entriesApi } from './api.js';
import { escapeHtml, formatTimestamp, stateLabel, stateToken } from './components.js';
import { registerReaderTab } from './reader.js';

/** @typedef {import('./api.js').ModuleCard} ModuleCard */
/** @typedef {import('./api.js').EnhancementsResponse} EnhancementsResponse */

const DASH = '—';

/**
 * A value, escaped, or an em dash when there is none.
 * @param {unknown} value
 * @returns {string}
 */
function show(value) {
  if (value === null || value === undefined || value === '') return DASH;
  return escapeHtml(String(value));
}

/**
 * A timestamp, formatted and escaped, or an em dash.
 * @param {?string|undefined} value
 * @returns {string}
 */
function time(value) {
  return value ? escapeHtml(formatTimestamp(value)) : DASH;
}

/**
 * One `<dt>`/`<dd>` pair; `html` is already escaped.
 * @param {string} term
 * @param {string} html
 * @returns {string}
 */
function field(term, html) {
  return `<dt>${escapeHtml(term)}</dt><dd>${html}</dd>`;
}

/**
 * @param {Record<string, any>} d
 * @returns {string}
 */
function semanticDetail(d) {
  const keywords = Array.isArray(d.keywords) ? d.keywords.map(String).join(', ') : '';
  return `<dl class="inspector-fields">${field('Summary', show(d.summary))}${field('Keywords', show(keywords))}</dl>`;
}

/**
 * Stored embedding rows, one per model table. Only the named fields are read.
 * @param {Record<string, any>} d
 * @returns {string}
 */
function textEmbeddingDetail(d) {
  const tables = Array.isArray(d.tables) ? d.tables : [];
  return tables
    .map((/** @type {any} */ t) => {
      const now = t?.would_embed_now ?? {};
      return `<dl class="inspector-fields" data-inspector-row>${field('Table', show(t?.table))}${field('Model', show(t?.model))}${field('Dimensions', show(t?.dimensions))}${field('Embedded', time(t?.created_at))}${field('Input key', show(t?.input_key))}${field('Input key now', show(now.input_key))}</dl>
        <pre class="inspector-text">${show(now.text)}</pre>`;
    })
    .join('');
}

/**
 * The image table and one row per picture. Only the named fields are read.
 * @param {Record<string, any>} d
 * @returns {string}
 */
function imageEmbeddingDetail(d) {
  const head = `<dl class="inspector-fields">${field('Table', show(d.table))}${field('Model', show(d.model))}${field('Dimensions', show(d.dimensions))}${field('Table exists', d.table_exists ? 'yes' : 'no')}</dl>`;
  const rows = (Array.isArray(d.rows) ? d.rows : [])
    .map((/** @type {any} */ r) => {
      const state = r?.stored
        ? `${show(r.dimensions)} dims${r.skip_reason ? `, skipped: ${escapeHtml(String(r.skip_reason))}` : ''}, ${show(r.model_ref)}, ${time(r.created_at)}`
        : 'not stored';
      return `<li data-inspector-row><code>${show(r?.attachment_id)}</code> ${state}</li>`;
    })
    .join('');
  return `${head}<ul class="inspector-rows">${rows}</ul>`;
}

/**
 * Captions per picture per model, then the composed attachment text.
 * @param {Record<string, any>} d
 * @returns {string}
 */
function imageCaptionDetail(d) {
  const pictures = (Array.isArray(d.pictures) ? d.pictures : [])
    .map((/** @type {any} */ p) => {
      const captions = p?.captions && typeof p.captions === 'object' ? p.captions : {};
      const lines = Object.entries(captions)
        .map(
          ([model, text]) =>
            `<div><span class="inspector-muted">${escapeHtml(model)}:</span> ${show(text)}</div>`
        )
        .join('');
      return `<li data-inspector-row><code>${show(p?.attachment_id)}</code>${lines || ` ${DASH}`}</li>`;
    })
    .join('');
  return `<dl class="inspector-fields">${field('Model', show(d.model))}</dl>
    <ul class="inspector-rows">${pictures}</ul>
    <pre class="inspector-text">${show(d.attachment_text)}</pre>`;
}

/**
 * @param {Record<string, any>} d
 * @returns {string}
 */
function qmdDetail(d) {
  return `<dl class="inspector-fields">${field('Mirror file', show(d.mirror_path))}${field('Exists', d.exists ? 'yes' : 'no')}</dl>`;
}

/** The modules with a detail block; any other module's card is status-only. */
const DETAIL_RENDERERS =
  /** @type {Record<string, ((d: Record<string, any>) => string) | undefined>} */ ({
    semantic_processor: semanticDetail,
    text_embedding: textEmbeddingDetail,
    image_embedding: imageEmbeddingDetail,
    image_caption: imageCaptionDetail,
    qmd_export: qmdDetail,
  });

/**
 * One module's card.
 * @param {ModuleCard} card
 * @returns {string} HTML string
 */
export function renderModuleCard(card) {
  const state = String(card.state ?? 'n/a');
  const token = stateToken(state);
  const feeds = Array.isArray(card.feeds) ? card.feeds.map(String).join(', ') : '';
  const fields = [
    field('Feeds', show(feeds)),
    field('Completed', time(card.completed_at)),
    field('Failed', time(card.failed_at)),
    field('Marker', show(card.marker)),
    field('Error', show(card.error)),
    field('Attempts', show(card.attempts)),
  ].join('');

  let command = '';
  if (card.command) {
    command = `<div class="inspector-command" data-inspector-command>
        <code>${escapeHtml(card.command)}</code>
        <button type="button" class="btn btn-secondary btn-sm" data-inspector-copy>Copy</button>
      </div>`;
    if (card.module === 'text_embedding' && state === 'stale') {
      command += `<div class="inspector-note">${escapeHtml('`reembed` rebuilds the whole table, not just this entry.')}</div>`;
    }
  }

  const render = Object.hasOwn(DETAIL_RENDERERS, card.module)
    ? DETAIL_RENDERERS[card.module]
    : undefined;
  const detail =
    render && card.detail && typeof card.detail === 'object'
      ? `<div class="inspector-detail" data-inspector-detail>${render(card.detail)}</div>`
      : '';

  return `<article class="inspector-card" data-inspector-card data-module="${escapeHtml(card.module)}" data-state="${escapeHtml(state)}">
      <header class="inspector-card-head">
        <h4 class="inspector-card-title">${escapeHtml(card.label || card.module)}</h4>
        <span class="inspector-chip ariel-strip-cell-${token}" data-inspector-chip>${escapeHtml(stateLabel(state))}</span>
      </header>
      <dl class="inspector-fields">${fields}</dl>
      ${command}${detail}
    </article>`;
}

/**
 * Copy a command and say so.
 * @param {string} command
 */
async function copyCommand(command) {
  try {
    await navigator.clipboard.writeText(command);
    showToast('Command copied to the clipboard.', { kind: 'success' });
  } catch (error) {
    console.error('Copy failed:', error);
    showToast('Could not copy the command; select it and copy by hand.', { kind: 'warning' });
  }
}

/**
 * The tab's content for an `/enhancements` response: the cards in execution
 * order, with each copy button wired.
 * @param {EnhancementsResponse} response
 * @returns {HTMLElement}
 */
export function renderInspector(response) {
  const root = document.createElement('div');
  root.className = 'entry-inspector';
  root.setAttribute('data-entry-inspector', '');
  const cards = [...(Array.isArray(response?.modules) ? response.modules : [])].sort(
    (a, b) => (a.order ?? 0) - (b.order ?? 0)
  );
  root.innerHTML =
    cards.length === 0
      ? `<div class="inspector-empty">${escapeHtml('No enhancement modules are enabled.')}</div>`
      : cards.map(renderModuleCard).join('');
  root.addEventListener('click', (event) => {
    const button = /** @type {HTMLElement|null} */ (event.target)?.closest?.(
      '[data-inspector-copy]'
    );
    if (!button) return;
    const code = button.closest('[data-inspector-command]')?.querySelector('code');
    if (code?.textContent) void copyCommand(code.textContent);
  });
  return root;
}

/** @type {import('./reader-seam.js').ReaderTab} */
export const inspectorTab = {
  id: 'enhancements',
  label: 'Enhancements',
  key: 'e',
  async render({ entryId }) {
    try {
      return renderInspector(await entriesApi.enhancements(entryId));
    } catch (error) {
      console.error('Enhancements failed to load:', error);
      return `<div class="inspector-empty">${escapeHtml('The enhancements could not be loaded.')}</div>`;
    }
  },
};

/** Register the "Enhancements" tab. */
export function registerInspector() {
  registerReaderTab(inspectorTab);
}
