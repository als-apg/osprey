// @ts-check
/**
 * OSPREY Channel Finder — Middle-Layer channel grouping (pure logic).
 *
 * Place grouping, positional device alignment, group ordering and the display
 * cap, with no DOM or module-state dependencies, so they are unit-tested under
 * Vitest (see tests/interfaces/channel_finder/explore-grouping.test.mjs).
 *
 * The renderer passes in the field's channels plus the parallel `device_list` /
 * `place_list` / `common_names` arrays (positionally aligned to the channels)
 * and the active place/device filter sets; this returns the resolved, ordered
 * place groups it turns into markup. A group is a place: it is keyed by the
 * place id and labelled by the id's last segment. A database that states no
 * places is grouped by the bare place index its rows carry.
 */

/** Max items rendered per place group before a "... and N more" summary. */
export const GROUP_DISPLAY_CAP = 50;

/** The group key of a channel with no device row. */
const UNKNOWN = '_unknown';

/** The group key and label of a device the database states no place for. */
const NO_PLACE = '-';

/**
 * @typedef {object} GroupedChannel
 * @property {string} name - Channel name (PV).
 * @property {string|null} device - Device ordinal inside its place (positional), or null.
 * @property {string|null} commonName - Common name (positional), or null.
 */

/**
 * @typedef {object} PlaceChip
 * @property {string} key - The place id; without a place list, the row's place index; '_unknown' for a channel with no row.
 * @property {string} label - The key's last `/` segment; the bare index without a place list; '-' for no place; 'Unknown' for '_unknown'.
 */

/**
 * @typedef {object} PlaceGroup
 * @property {string} key - See {@link PlaceChip}.
 * @property {string} label - See {@link PlaceChip}.
 * @property {GroupedChannel[]} shown - Items to render (capped at `cap`).
 * @property {number} total - Total items in this group before the cap.
 * @property {number} hidden - Items beyond the cap (total - shown.length).
 */

/**
 * @typedef {object} GroupedField
 * @property {PlaceGroup[]} places - Place groups, `_unknown` ordered last.
 * @property {number} visibleCount - Total items passing the active filters.
 */

/**
 * @typedef {object} RowGroup
 * @property {string} key
 * @property {string} label
 * @property {number} index - The row's place index; Infinity for no row.
 */

/**
 * `text` as natural-order parts: digit runs compare as numbers.
 * @param {string} text
 * @returns {(string|number)[]}
 */
function naturalParts(text) {
  return text.split(/(\d+)/).filter(part => part !== '').map(part => (/^\d+$/.test(part) ? Number(part) : part));
}

/**
 * @param {string} a
 * @param {string} b
 * @returns {number}
 */
function compareNatural(a, b) {
  const pa = naturalParts(a);
  const pb = naturalParts(b);
  for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
    const x = pa[i];
    const y = pb[i];
    if (x === y) continue;
    if (typeof x === 'number' && typeof y === 'number') return x - y;
    if (typeof x === 'number') return -1;
    if (typeof y === 'number') return 1;
    return x < y ? -1 : 1;
  }
  return pa.length - pb.length;
}

/**
 * The group the row at position `i` belongs to.
 * @param {number} i
 * @param {any[]} deviceList
 * @param {any[]|null|undefined} placeList
 * @returns {RowGroup}
 */
function rowGroup(i, deviceList, placeList) {
  const entry = i < deviceList.length ? deviceList[i] : null;
  if (!entry || entry.length < 2) return { key: UNKNOWN, label: 'Unknown', index: Infinity };
  const index = Number(entry[0]);
  if (!Array.isArray(placeList)) {
    const bare = String(entry[0]);
    return { key: bare, label: bare, index };
  }
  const place = i < placeList.length ? placeList[i] : null;
  if (place === null || place === undefined || place === '') {
    return { key: NO_PLACE, label: NO_PLACE, index };
  }
  const id = String(place);
  return { key: id, label: id.slice(id.lastIndexOf('/') + 1), index };
}

/**
 * Groups by place index (numerically, index 0 first), then label in natural
 * order, then key; `_unknown` last.
 * @param {RowGroup} a
 * @param {RowGroup} b
 * @returns {number}
 */
function compareGroups(a, b) {
  if (a.key === UNKNOWN || b.key === UNKNOWN) return (a.key === UNKNOWN ? 1 : 0) - (b.key === UNKNOWN ? 1 : 0);
  if (a.index !== b.index) return a.index < b.index ? -1 : 1;
  return compareNatural(a.label, b.label) || compareNatural(a.key, b.key);
}

/**
 * The filter chips of a family: one per place its devices sit in, in the
 * order {@link groupFieldChannels} lists its groups. Pure.
 *
 * @param {any[]} deviceList - `deviceInfo.device_list`: positional [place index, ordinal] rows.
 * @param {any[]|null|undefined} placeList - `deviceInfo.place_list`: positional place ids, or null.
 * @returns {PlaceChip[]}
 */
export function placeChips(deviceList, placeList) {
  /** @type {Map<string, RowGroup>} */
  const seen = new Map();
  deviceList.forEach((_entry, i) => {
    const group = rowGroup(i, deviceList, placeList);
    if (!seen.has(group.key)) seen.set(group.key, group);
  });
  return [...seen.values()].sort(compareGroups).map(({ key, label }) => ({ key, label }));
}

/**
 * Group a field's channels by place using positional device alignment,
 * applying the active place/device filters, ordering the groups, and
 * truncating each group to `cap`. Pure: no DOM, no module state.
 *
 * @param {any[]} channels - Field channels (strings or {name|channel} objects).
 * @param {any[]} deviceList - `deviceInfo.device_list`: positional [place index, ordinal] rows.
 * @param {any[]|null|undefined} placeList - `deviceInfo.place_list`: positional place ids, or null.
 * @param {any[]|null|undefined} commonNames - `deviceInfo.common_names`: positional labels.
 * @param {Set<string>} activePlaces - Active place filter, by group key (empty = no filter).
 * @param {Set<string>} activeDevices - Active device filter (empty = no filter).
 * @param {number} [cap] - Per-group display cap.
 * @returns {GroupedField}
 */
export function groupFieldChannels(channels, deviceList, placeList, commonNames, activePlaces, activeDevices, cap = GROUP_DISPLAY_CAP) {
  /** @type {Map<string, {group: RowGroup, items: GroupedChannel[]}>} */
  const grouped = new Map();
  let visibleCount = 0;

  channels.forEach((ch, i) => {
    const name = typeof ch === 'string' ? ch : (ch.name || ch.channel || '');
    const entry = i < deviceList.length ? deviceList[i] : null;
    const device = entry && entry.length >= 2 ? String(entry[1]) : null;
    const commonName = commonNames && i < commonNames.length ? commonNames[i] : null;
    const group = rowGroup(i, deviceList, placeList);

    if (activePlaces.size > 0 && (group.key === UNKNOWN || !activePlaces.has(group.key))) return;
    if (activeDevices.size > 0 && (device === null || !activeDevices.has(device))) return;

    let bucket = grouped.get(group.key);
    if (!bucket) {
      bucket = { group, items: [] };
      grouped.set(group.key, bucket);
    }
    bucket.items.push({ name, device, commonName });
    visibleCount++;
  });

  const places = [...grouped.values()]
    .sort((a, b) => compareGroups(a.group, b.group))
    .map(({ group, items }) => ({
      key: group.key,
      label: group.label,
      shown: items.slice(0, cap),
      total: items.length,
      hidden: Math.max(0, items.length - cap),
    }));

  return { places, visibleCount };
}
