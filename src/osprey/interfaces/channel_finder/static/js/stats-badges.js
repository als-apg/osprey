// @ts-check
/**
 * OSPREY Channel Finder — Stats Badges
 *
 * Compact header badges showing channel count, system count, etc.
 * Fetched from /api/statistics and rendered into #stats-badges.
 */

import { fetchJSON } from './api.js';
import { esc } from './utils.js';
import { state } from './state.js';

/**
 * Every statistic the header can show, as ``[payload key, badge label]``, in
 * render order. A paradigm reports the subset it can count and omits the rest,
 * so the order here — not the payload's — is what the reader sees; a key that
 * is absent or null simply renders no badge. A new statistic adds one row.
 * @type {ReadonlyArray<readonly [string, string]>}
 */
const BADGE_KEYS = [
  ['total_devices', 'devices'],
  ['total_channels', 'channels'],
  ['total_classes', 'classes'],
  ['total_signals', 'signals'],
  ['total_sections', 'sections'],
  ['total_systems', 'systems'],
  ['total_families', 'families'],
  ['total_templates', 'templates'],
  ['total_standalone', 'standalone'],
  ['total_chunks_at_50', 'chunks'],
];

/**
 * Fetch statistics and render compact badges into the header.
 * Call on init and after every CRUD mutation.
 */
export async function refreshStatsBadges() {
  const container = document.getElementById('stats-badges');
  if (!container) return;

  try {
    const stats = await fetchJSON('/api/statistics');
    /** @type {Array<{value: string, label: string}>} */
    const badges = [];

    for (const [key, label] of BADGE_KEYS) {
      const value = stats[key];
      if (value === null || value === undefined) continue;
      badges.push({ value: value.toLocaleString(), label });
    }

    container.innerHTML = badges
      .map(
        (b) =>
          `<span class="stats-badge"><span class="badge-value">${esc(b.value)}</span> <span class="badge-label">${b.label}</span></span>`
      )
      .join('');
  } catch {
    container.innerHTML = '';
  }
}

/**
 * Name the graph store the corpus strip's numbers came from, beside them: the
 * store's URI (and the TTL file it was seeded from, when known), and an info
 * tip listing the tools the assistant reads that same store with — so the
 * panel and the agent are visibly looking at one corpus. Empty on the
 * file-backed paradigms, which report neither.
 * @returns {void}
 */
export function renderProvenance() {
  const slot = document.getElementById('corpus-prov');
  if (!slot) return;
  const store = state.graphStore;
  const uri = store && store.uri ? store.uri : '';
  const file = store && store.ttl_filename ? store.ttl_filename : '';
  const label = file && uri ? `${file} @ ${uri}` : file || uri;
  if (state.pipelineType !== 'graph' || !label) {
    slot.innerHTML = '';
    return;
  }
  const tools = state.tools || [];
  const tip = tools.length
    ? ` · ${tools.length} assistant ${tools.length === 1 ? 'tool' : 'tools'}` +
      '<button type="button" class="osprey-info" data-place="above"' +
      ' aria-label="Tools the assistant queries this store with">?' +
      '<span class="osprey-tip" role="tooltip"><span>The assistant reads this same store with</span>' +
      `<span>${tools.map((tool) => `<b class="graph-tool-chip">${esc(tool)}</b>`).join(', ')}</span>` +
      '</span></button>'
    : '';
  slot.innerHTML = `<span class="graph-store-badge" title="Graph store">store <code>${esc(label)}</code></span>${tip}`;
}
