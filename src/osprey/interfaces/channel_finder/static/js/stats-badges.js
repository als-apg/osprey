/**
 * OSPREY Channel Finder — Stats Badges
 *
 * Compact header badges showing channel count, system count, etc.
 * Fetched from /api/statistics and rendered into #stats-badges.
 */

import { fetchJSON } from './api.js';
import { state } from './state.js';

/**
 * Fetch statistics and render compact badges into the header.
 * Call on init and after every CRUD mutation.
 */
export async function refreshStatsBadges() {
  const container = document.getElementById('stats-badges');
  if (!container) return;

  try {
    const stats = await fetchJSON('/api/statistics');
    const badges = [];

    if (stats.total_channels != null) {
      badges.push({ value: stats.total_channels.toLocaleString(), label: 'channels' });
    }
    if (stats.total_systems != null) {
      badges.push({ value: stats.total_systems, label: 'systems' });
    }
    if (stats.total_families != null) {
      badges.push({ value: stats.total_families, label: 'families' });
    }
    if (stats.total_templates != null) {
      badges.push({ value: stats.total_templates, label: 'templates' });
    }
    if (stats.total_standalone != null) {
      badges.push({ value: stats.total_standalone, label: 'standalone' });
    }
    if (stats.total_chunks_at_50 != null) {
      badges.push({ value: stats.total_chunks_at_50, label: 'chunks' });
    }

    container.innerHTML = badges.map(b =>
      `<span class="stats-badge"><span class="badge-value">${b.value}</span> <span class="badge-label">${b.label}</span></span>`
    ).join('');
  } catch {
    container.innerHTML = '';
  }
}
