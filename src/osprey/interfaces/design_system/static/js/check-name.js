// @ts-check
/* OSPREY Design System — health-check name humaniser
 *
 * A health check is identified by a free-form name and carries the category it
 * belongs to beside it. Every surface that lists checks — the System Health
 * dashboard's own bundle, the web terminal's health bar item — shows the same
 * name the same way, and each one used to carry its own copy of the rule. The
 * design system is the only mount both pages hold, so the rule lives here and
 * both read it.
 */

/**
 * Humanize a check name for display: drop the row's OWN `"<category>."` prefix
 * and title-case the remaining underscore/space separated words.
 * `fmtName("control_system.beam_current", "control_system")` → `"Beam Current"`.
 *
 * A check name is free-form, so a dot in it is not a category marker by
 * itself — only the category the row actually carries is stripped, or a name
 * that merely spells a dot loses its leading word. Existing capitalisation
 * survives: the regex only ever raises a lowercase letter.
 *
 * @param {string} name
 * @param {string} [category]
 * @returns {string}
 */
export function fmtName(name, category) {
  const prefix = category ? `${category}.` : "";
  const s = prefix && name.startsWith(prefix) ? name.slice(prefix.length) : name;
  return s.replace(/_/g, " ").replace(/\b[a-z]/g, (c) => c.toUpperCase());
}
