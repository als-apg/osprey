// @ts-check
/**
 * OSPREY Channel Finder — Middle-Layer channel grouping (pure logic).
 *
 * Extracted from explore-middle-layer.js so the sector-grouping / positional
 * device-alignment / `_unknown`-last ordering / display-cap logic has no DOM or
 * module-state dependencies and can be unit-tested under Vitest
 * (see tests/interfaces/channel_finder/explore-grouping.test.mjs).
 *
 * The renderer passes in the field's channels plus the parallel `device_list` /
 * `common_names` arrays (positionally aligned to the channels) and the active
 * sector/device filter sets; this returns the resolved, ordered sector groups it
 * turns into markup — output is identical to the previous inline logic.
 */

/** Max items rendered per sector group before a "... and N more" summary. */
export const SECTOR_DISPLAY_CAP = 50;

/**
 * @typedef {object} GroupedChannel
 * @property {string} name - Channel name (PV).
 * @property {string|null} device - Device number/id (positional), or null.
 * @property {string|null} commonName - Common name (positional), or null.
 */

/**
 * @typedef {object} SectorGroup
 * @property {string} key - Sector key ('_unknown' for channels with no sector).
 * @property {string} label - Display label ('Sector 3' or 'Unknown').
 * @property {GroupedChannel[]} shown - Items to render (capped at `cap`).
 * @property {number} total - Total items in this sector before the cap.
 * @property {number} hidden - Items beyond the cap (total - shown.length).
 */

/**
 * @typedef {object} GroupedField
 * @property {SectorGroup[]} sectors - Sector groups, `_unknown` ordered last.
 * @property {number} visibleCount - Total items passing the active filters.
 */

/**
 * Group a field's channels by sector using positional device alignment,
 * applying the active sector/device filters, ordering `_unknown` last, and
 * truncating each sector to `cap`. Pure: no DOM, no module state.
 *
 * @param {any[]} channels - Field channels (strings or {name|channel} objects).
 * @param {any[]} deviceList - `deviceInfo.device_list`: positional [sector, device] tuples.
 * @param {any[]|null|undefined} commonNames - `deviceInfo.common_names`: positional labels.
 * @param {Set<string>} activeSectors - Active sector filter (empty = no filter).
 * @param {Set<string>} activeDevices - Active device filter (empty = no filter).
 * @param {number} [cap] - Per-sector display cap.
 * @returns {GroupedField}
 */
export function groupFieldChannels(channels, deviceList, commonNames, activeSectors, activeDevices, cap = SECTOR_DISPLAY_CAP) {
  /** @type {Record<string, GroupedChannel[]>} */
  const grouped = {};
  let visibleCount = 0;

  channels.forEach((ch, i) => {
    const name = typeof ch === 'string' ? ch : (ch.name || ch.channel || '');
    const entry = i < deviceList.length ? deviceList[i] : null;
    const sector = entry && entry.length >= 2 ? String(entry[0]) : null;
    const device = entry && entry.length >= 2 ? String(entry[1]) : null;
    const commonName = commonNames && i < commonNames.length ? commonNames[i] : null;

    if (activeSectors.size > 0 && (sector === null || !activeSectors.has(sector))) return;
    if (activeDevices.size > 0 && (device === null || !activeDevices.has(device))) return;

    const key = sector || '_unknown';
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push({ name, device, commonName });
    visibleCount++;
  });

  // Sort sectors, keeping the "_unknown" bucket last.
  const sectorKeys = Object.keys(grouped).sort((a, b) => {
    if (a === '_unknown') return 1;
    if (b === '_unknown') return -1;
    return a < b ? -1 : a > b ? 1 : 0;
  });

  const sectors = sectorKeys.map(key => {
    const items = grouped[key];
    return {
      key,
      label: key === '_unknown' ? 'Unknown' : `Sector ${key}`,
      shown: items.slice(0, cap),
      total: items.length,
      hidden: Math.max(0, items.length - cap),
    };
  });

  return { sectors, visibleCount };
}
