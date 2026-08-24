// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: write posture
 *
 * The client half of `web.scaffold_gallery.write_enabled`. The server side is
 * the real enforcement: with the key off, every write and delete verb under
 * `/api/scaffold` refuses with 403 before the gallery service is even
 * constructed (routes/scaffold.py). This module is what stops the browser from
 * painting controls that reach for that refusing surface.
 *
 * Why the posture rides `GET /api/panels`: the gallery's own payloads describe
 * ARTIFACTS (what each one is, whether it is reserved), and the deployment's
 * tier is not a property of any artifact. `/api/panels` is the payload every
 * boot consumer already reads for exactly this class of fact — `ui_mode`,
 * `allow_runtime_panels`, `config_panel_enabled` — so gating off it keeps one
 * source of truth and costs the page no extra round trip.
 *
 * Why a module-level posture rather than a per-gallery flag: the three
 * ArtifactGallery instances are built at DOMContentLoaded and their controls
 * are rendered lazily, when a drawer tab is first activated. The posture
 * arrives once, at boot, and every gallery that renders afterwards must agree
 * with it — including ones constructed before it landed.
 *
 * Absence of the field means ENABLED, matching the server's own
 * `getattr(app.state, "scaffold_write_enabled", True)` default: only an
 * explicit `false` withdraws the controls. A null payload — a failed or hung
 * `/api/panels` — leaves the posture as it stands; a read that never landed is
 * not a statement about the deployment.
 *
 * Timing, stated plainly: the payload arrives one round trip after boot, and
 * the drawer is opened by a human some time after that, so in practice the
 * posture is resolved long before a control is painted. Even if it were not,
 * the gap is cosmetic — the routes behind those controls answer 403 from the
 * first request.
 *
 * @module scaffold/write-gate
 */

/**
 * Sentence shown on the one control that stays rendered-but-disabled (the Edit
 * mode tab), and available to any other affordance that needs to say why.
 * Phrased as the deployment's posture rather than the operator's fault.
 */
export const WRITES_DISABLED_REASON =
  'This deployment does not allow editing the agent’s setup from the browser.';

/** Resolved posture. Enabled until a payload says otherwise. */
let writesEnabled = true;

/**
 * Record the deployment's gallery-write posture from a `/api/panels` payload.
 *
 * @param {any} panelsPayload - the `GET /api/panels` response, or null.
 * @returns {boolean} the posture in force after this call.
 */
export function applyScaffoldWriteGate(panelsPayload) {
  if (panelsPayload) writesEnabled = panelsPayload.scaffold_write_enabled !== false;
  return writesEnabled;
}

/**
 * Whether this deployment's gallery may write.
 *
 * Read at RENDER time, never cached by callers: the posture lands one round
 * trip after boot, and a control built before it must not outlive it.
 *
 * @returns {boolean}
 */
export function scaffoldWritesEnabled() {
  return writesEnabled;
}
