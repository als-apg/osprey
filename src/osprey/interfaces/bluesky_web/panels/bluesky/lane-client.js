// @ts-check
/**
 * BLUESKY panel — plan-lane addressing.
 *
 * A PLAN LANE is a whole bridge stack bound at render time to one
 * control-system target; a two-lane deployment runs two of them about two
 * different machines, and the sidecar addresses them by a `?lane=` query
 * parameter on every route (reads, draft, queue, both SSE streams). This
 * module owns the panel's half of that contract as pure functions — parsing
 * the sidecar's `GET /lanes` roster, resolving which lane this document
 * addresses, and spelling the lane onto a request path — so the whole
 * contract is unit-testable with plain strings.
 *
 * The panel binds to exactly ONE lane per document: the lane is read from the
 * document's own URL (`?lane=`), every request the panel makes carries it,
 * and switching lanes navigates to a new document. A lane is a different
 * MACHINE, and an in-place switch would leave a window where the queue view
 * shows one machine's plans beside the other machine's results — the exact
 * wrong-machine confusion the lane axis exists to remove. A fresh document
 * per lane makes that state unrepresentable.
 *
 * The panel never decides what a lane may do. An unknown lane is pinned, not
 * silently rerouted to lane 1 — the sidecar answers every request about it
 * with its 404 and the panel says so — and whether a lane can execute is the
 * bridge's own capability record, read per lane through the existing
 * `/bridge/health` fetch (which this module's addressing scopes like every
 * other request).
 */

/**
 * Service key of the plan lane every deployment has. Mirrors
 * `bluesky_bridge_connection.LANE_ONE`; requests addressed to it stay bare
 * (no `lane` parameter), so a single-lane deployment's requests are
 * byte-identical to what the panel has always sent.
 * @type {string}
 */
export const LANE_ONE = 'bluesky';

/**
 * The sidecar's lane query parameter (`read_proxy.LANE_QUERY_PARAM`).
 * @type {string}
 */
export const LANE_QUERY_PARAM = 'lane';

/** @typedef {{lane: string, laneTarget: string|null}} LaneEntry */

/** The roster a single-lane deployment has, and the fallback when the
 *  sidecar's answer cannot be read: one lane, no declared target.
 * @returns {LaneEntry[]}
 */
function singleLaneRoster() {
  return [{ lane: LANE_ONE, laneTarget: null }];
}

/**
 * Parse the sidecar's `GET /lanes` body into a roster.
 *
 * Malformed or empty answers collapse to the single-lane roster rather than
 * to an empty one: the panel must render regardless, and "one lane" is the
 * shape every deployment had before the lane axis existed. Entries without a
 * usable `lane` key are dropped, not guessed at.
 *
 * @param {any} body
 * @returns {LaneEntry[]}
 */
export function parseLaneRoster(body) {
  if (!body || typeof body !== 'object' || !Array.isArray(body.lanes)) return singleLaneRoster();
  const lanes = body.lanes
    .filter(
      (/** @type {any} */ entry) =>
        entry && typeof entry === 'object' && typeof entry.lane === 'string' && entry.lane !== ''
    )
    .map((/** @type {{lane: string, lane_target?: unknown}} */ entry) => ({
      lane: entry.lane,
      laneTarget:
        typeof entry.lane_target === 'string' && entry.lane_target ? entry.lane_target : null,
    }));
  return lanes.length ? lanes : singleLaneRoster();
}

/**
 * The lane this document's own URL addresses.
 *
 * No parameter (or an empty one) is lane 1 — the only lane a single-lane
 * deployment has, so nothing about those documents changes. A named lane is
 * returned VERBATIM, whether or not the roster knows it: rerouting an unknown
 * lane to lane 1 silently would point the operator at a different machine
 * than the URL names, which is worse than the honest alternative — every
 * request 404s at the sidecar and `laneIsKnown` lets the shell say why.
 *
 * @param {string} search  the document's `location.search`
 * @returns {string}
 */
export function resolveLaneFromSearch(search) {
  return new URLSearchParams(search).get(LANE_QUERY_PARAM) || LANE_ONE;
}

/**
 * Whether the roster renders `lane`.
 *
 * @param {LaneEntry[]} roster
 * @param {string} lane
 * @returns {boolean}
 */
export function laneIsKnown(roster, lane) {
  return roster.some((entry) => entry.lane === lane);
}

/**
 * Spell the lane axis onto a sidecar-relative request path.
 *
 * Lane 1 stays bare — the parameter defaults there sidecar-side, and keeping
 * it off the wire makes a single-lane deployment's requests byte-identical
 * to what the panel has always sent. Composes with paths that already carry
 * a query string (`/draft?client_id=…`).
 *
 * @param {string} path
 * @param {string} lane
 * @returns {string}
 */
export function withLane(path, lane) {
  if (!lane || lane === LANE_ONE) return path;
  const separator = path.includes('?') ? '&' : '?';
  return `${path}${separator}${LANE_QUERY_PARAM}=${encodeURIComponent(lane)}`;
}

/**
 * The picker label for one lane: the control target it drives.
 *
 * A lane is named for its target, never for its index — "va" / "live" is
 * what tells an operator which machine a click arms. The service key is the
 * fallback for a roster entry that declared none (lane 1 on rosters written
 * before targets were declared), so the picker never renders a blank button.
 *
 * @param {LaneEntry} entry
 * @returns {string}
 */
export function laneLabel(entry) {
  return entry.laneTarget || entry.lane;
}

/**
 * The `location.search` string addressing `lane`, with every other parameter
 * (the host's `?embedded=`, `?mode=`, `?theme=`) preserved — dropping those
 * on a lane switch would un-embed the panel or flash it into the wrong
 * theme. Lane 1 removes the parameter rather than naming it, mirroring
 * `withLane`.
 *
 * @param {string} search  the document's current `location.search`
 * @param {string} lane
 * @returns {string}  a search string (`?…`), or `''` for no query at all
 */
export function laneSearch(search, lane) {
  const params = new URLSearchParams(search);
  if (!lane || lane === LANE_ONE) {
    params.delete(LANE_QUERY_PARAM);
  } else {
    params.set(LANE_QUERY_PARAM, lane);
  }
  const spelled = params.toString();
  return spelled ? `?${spelled}` : '';
}
