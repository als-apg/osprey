// @ts-check
/**
 * What the design system says about one facility reference: its hover text and
 * whether the facility file holds no record for it. A knowledge page and a
 * logbook entry say the same thing about the same record.
 */

/**
 * @param {string} target - The id the reference names
 * @param {Array<{kind?: string, unit?: string, description?: string}>|null} [records] -
 *   The records it resolves to; not an array when no facility file was read
 * @returns {{title: string, dangling: boolean}}
 */
export function facilityReferenceView(target, records) {
  if (!Array.isArray(records)) {
    return { title: target + ' · facility file not loaded', dangling: false };
  }
  if (records.length === 0) {
    return { title: target + ' · not in the facility file', dangling: true };
  }
  const title = records
    .map(function (record) {
      return [record.kind, record.unit, record.description].filter(Boolean).join(' · ');
    })
    .join('\n');
  return { title, dangling: false };
}
