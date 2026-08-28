// @ts-check
/* OSPREY Web Terminal — Agent Activity Vocabulary
 *
 * How an agent_activity frame is WORDED. Two surfaces render one — the live
 * line in activity-strip.js and the history rows in activity-history.js — and
 * both word it through here, so a hide reads the same wherever it appears.
 *
 * Pure by construction: no DOM, no fetch, no module state. Callers own the
 * rendering, and every agent-supplied string they get back goes in as a text
 * node (createElement + textContent, never innerHTML).
 */

/** @typedef {import('./panel-manager.js').AgentActivityEvent} AgentActivityFrame */

/**
 * The `type` discriminator an agent_activity SSE frame carries. Spelled once
 * here so the session page's forwarder, panel-sse's dispatcher and the panel
 * frames it synthesizes cannot drift apart on a rename; the AgentActivityEvent
 * typedef in panel-manager.js derives its literal from this constant too.
 */
export const AGENT_ACTIVITY_FRAME = 'agent_activity';

/**
 * The tool name the panel routes mirror an agent arrange under. Its subject is
 * the workspace itself rather than a single panel, so it is worded apart from
 * the PANEL_VERBS table — and the strip keys its burst coalescing on it.
 */
export const ARRANGE_TOOL = 'arrange_workspace';

/**
 * Verb per panel action, keyed on the synthetic tool names the panel routes
 * record for agent-origin gestures (see routes/panels.py).
 * @type {Record<string, string>}
 */
const PANEL_VERBS = {
  remove_panel_from_rail: 'agent removed',
  add_panel_to_rail: 'agent made available',
  open_panel: 'agent opened',
  close_panel: 'agent closed',
  register_panel: 'agent added',
};

/**
 * Human-readable two-part label for a frame. The subject carries the
 * agent-supplied string and is rendered as a text node by the caller.
 * @param {AgentActivityFrame} frame
 * @param {{ labelOf?: (id: string) => string, count?: number }} [opts]
 *   `labelOf` resolves a panel id to its catalog label; without it (or for an
 *   id the catalog does not know) the raw id is shown. `count` is how many
 *   coalesced arrange frames this one line stands for.
 * @returns {{ verb: string, subject: string }}
 */
export function formatActivity(frame, opts = {}) {
  const t = frame.target;
  /** Catalog label for a panel id, falling back to the id, then the tool.
   *  @param {string | undefined} id @returns {string} */
  const label = (id) => (id ? opts.labelOf?.(id) || id : frame.tool);
  switch (t.kind) {
    case 'channel':
      // Neutral on purpose: a channel detail is the display/channel the tool
      // was ASKED to drive, never a read-back of what the machine confirmed.
      return { verb: 'agent wrote', subject: t.detail || frame.tool };
    case 'run':
      return t.detail
        ? { verb: 'agent launched run', subject: t.detail }
        : { verb: 'agent launched a run', subject: '' };
    case 'artifact':
      return { verb: 'agent focused', subject: t.detail || 'an artifact' };
    case 'config':
      return { verb: 'agent changed config', subject: t.detail || frame.tool };
    case 'ui':
      return { verb: 'agent moved window', subject: t.detail || frame.tool };
    case 'panel': {
      if (frame.tool === ARRANGE_TOOL) {
        // A lone arrange row (in history, or the first frame of a burst) has
        // no count to report; only a coalesced run names how many tiles moved.
        const n = opts.count ?? 1;
        return { verb: 'agent arranged', subject: n > 1 ? `workspace (${n} tiles)` : 'workspace' };
      }
      const verb = PANEL_VERBS[frame.tool];
      if (verb) return { verb, subject: label(t.panel) };
      // Generic fallback: a panel-kind frame from some other tool.
      return { verb: 'agent touched', subject: t.panel || frame.tool };
    }
    default:
      // Unknown future kind from a newer server — still show something.
      return { verb: 'agent activity', subject: frame.tool };
  }
}

/**
 * Coarse "how long ago" label for a frame's server timestamp.
 *
 * Deliberately coarse: history rows answer "what has the agent been doing",
 * not "at exactly which instant". A missing ts (older server, or a frame
 * synthesised client-side) yields an empty string, which the caller skips.
 * Browser and server clocks can disagree by a little, so any non-positive
 * age reads as "just now" rather than a negative number.
 * @param {number | undefined} ts epoch seconds, as the server stamps them
 * @param {number} nowMs current wall clock in ms (Date.now())
 * @returns {string}
 */
export function formatRelativeTime(ts, nowMs) {
  if (typeof ts !== 'number' || !Number.isFinite(ts)) return '';
  const secs = Math.round(nowMs / 1000 - ts);
  if (secs < 1) return 'just now';
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
