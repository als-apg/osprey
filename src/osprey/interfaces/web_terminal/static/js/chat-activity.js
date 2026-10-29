// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode chat: the activity line's phrases.
 *
 * The operator phrase for each tool the agent is likely to use, and the label
 * the activity line shows for a `tool_use` event. A mapped tool reads as a
 * sentence. An unmapped one keeps its raw name, so it is never invisible here.
 *
 * The vocabulary is exported for its own unit suite; the chat itself reaches
 * it only through the renderer's activity line.
 *
 * @module chat-activity
 */

/** @typedef {import('./chat-render.js').ChatEvent} ChatEvent */

/**
 * Operator phrases for the tools a turn is likely to use, keyed by normalised
 * tool name (see {@link normaliseToolName}).
 *
 * In Simple mode the chat is the whole interface, so this line is the only
 * account an operator gets of what the agent is doing. Phrases say it in
 * control-room terms and share the activity strip's vocabulary (open/close/
 * focus/arrange for panels, "wrote" for channels).
 *
 * They are lower-case gerunds — "writing control channels" — so a caller can
 * drop one mid-sentence; {@link activityLabel} capitalises for the activity
 * line. Coverage is deliberately partial: a tool whose formatted name already
 * reads as plain English ("List Panels", "Session Summary") falls through to
 * the raw-name fallback instead of earning a row. Adding one is a single line.
 *
 * Phrases cannot name the specific panel, channel, or file involved: they are
 * fixed text per tool, whatever the call's arguments. The chat route
 * (`_strip_for_chat` in routes/chat.py) does keep a `tool_use` event's `input`,
 * clipped, and {@link activityLabel} reads it for one thing only: the name of
 * a skill.
 *
 * The keyspace is server-agnostic by design: keys are short tool names with
 * the `mcp__<server>__` prefix stripped, so a name owned by two servers
 * (e.g. `capabilities` on both ariel and osprey_facility_knowledge) would
 * share one phrase. Keep phrases generic enough to be true for every owner.
 *
 * @type {Readonly<Record<string, string>>}
 */
export const TOOL_PHRASES = Object.freeze({
  // Control system — the writes an operator most needs to see coming.
  channel_write: 'writing control channels',
  channel_read: 'reading control channels',
  channel_limits: 'checking channel limits',
  archiver_read: 'reading archived data',
  archiver_downsample: 'thinning archived data',

  // Python executor.
  execute: 'running Python',
  execute_file: 'running Python',

  // Workspace panels. The synthetic panel activity uses these same names.
  add_panel_to_rail: 'making a panel available',
  remove_panel_from_rail: 'removing a panel',
  open_panel: 'opening a panel',
  close_panel: 'closing a panel',
  arrange_workspace: 'arranging the workspace',
  register_panel: 'adding a panel',
  manage_window: 'arranging a window',
  screenshot_capture: 'taking a screenshot',

  // Workspace artifacts (plots, documents and stored datasets alike).
  artifact_register: 'registering an artifact',
  artifact_get: 'opening an artifact',
  artifact_list: 'listing saved artifacts',
  artifact_read: 'reading a saved artifact',
  artifact_focus: 'showing an artifact',
  artifact_pin: 'pinning an artifact',
  artifact_export: 'exporting an artifact',
  artifact_delete: 'deleting an artifact',
  artifact_delete_all: 'deleting artifacts',
  create_static_plot: 'drawing a plot',
  create_interactive_plot: 'drawing an interactive plot',
  create_dashboard: 'building a dashboard',
  create_document: 'writing a document',

  // Workspace project setup and session record.
  setup_inspect: 'inspecting the project setup',
  setup_patch: 'changing the project setup',
  session_log: 'writing to the session log',

  // Plan queue (bluesky).
  queue_status: 'checking the plan queue',
  queue_list: 'listing the plan queue',
  queue_add: 'queueing a plan',
  queue_start: 'starting the plan queue',
  queue_stop: 'stopping the plan queue',
  stop_run: 'stopping the running plan',
  write_plan: 'drafting a plan',
  validate_plan: 'checking the plan',
  get_draft: 'reading the plan draft',
  set_draft: 'editing the plan draft',
  clear_draft: 'clearing the plan draft',
  get_run: 'reading a plan run',
  get_run_data: 'reading run data',
  get_run_figure: 'reading a run figure',

  // Phoebus displays.
  phoebus_open_panel: 'opening a Phoebus display',
  phoebus_open_databrowser: 'opening the data browser',
  phoebus_list_displays: 'listing Phoebus displays',
  phoebus_perceive: 'reading a Phoebus display',
  phoebus_perceive_region: 'reading part of a Phoebus display',
  phoebus_snapshot: 'capturing a Phoebus display',
  phoebus_drive: 'operating a Phoebus display',

  // Logbook (ARIEL).
  browse: 'browsing the logbook',
  keyword_search: 'searching the logbook',
  semantic_search: 'searching the logbook',
  hybrid_search: 'searching the logbook',
  sql_query: 'querying the logbook',
  filter_options: 'listing logbook filters',
  entry_get: 'reading a logbook entry',
  entries_by_ids: 'reading logbook entries',
  attachment_view: 'viewing a logbook picture',
  entry_open: 'showing a logbook entry',
  attachment_to_artifact: 'saving a logbook picture',
  entry_create: 'drafting a logbook entry',
  entry_publish: 'publishing a logbook entry',

  // Channel finder.
  list_channels: 'looking up channels',
  ask_channels: 'looking up channels',
  run_sql: 'querying the channel database',
  build_channels: 'building a channel list',
  list_families: 'looking up channel families',
  list_systems: 'looking up systems',
  get_common_names: 'looking up channel names',
  inspect_fields: 'inspecting channel fields',

  // Facility knowledge.
  list_concepts: 'browsing facility knowledge',
  read_concept: 'reading facility knowledge',
  draft_concept: 'drafting a facility note',

  // Lattice model. The mutators are phrased; the getters fall back.
  lattice_state: 'reading the lattice',
  lattice_set_param: 'changing a lattice parameter',
  lattice_set_baseline: 'setting the lattice baseline',
  lattice_clear_baseline: 'clearing the lattice baseline',
  lattice_update_settings: 'changing lattice settings',
  lattice_refresh: 'refreshing the lattice view',

  // Health.
  health_check: 'checking system health',
  health_check_full: 'checking system health',

  // Built-in agent tools.
  read: 'reading a file',
  write: 'writing a file',
  edit: 'editing a file',
  bash: 'running a shell command',
  glob: 'looking for files',
  grep: 'searching files',
  skill: 'loading a skill',
  task: 'delegating to a helper agent',
  todowrite: 'updating its plan',
  taskcreate: 'updating its plan',
  taskupdate: 'updating its plan',
  tasklist: 'reading its plan',
  taskget: 'reading its plan',
  enterplanmode: 'planning',
  exitplanmode: 'finishing its plan',
  webfetch: 'fetching a web page',
  websearch: 'searching the web',
});

/**
 * Fold a tool name to a {@link TOOL_PHRASES} key: drop the `mcp__<server>__`
 * prefix, collapse whitespace and hyphens to underscores, lower-case.
 *
 * A `tool_use` event carries the name twice — `tool_name_raw`
 * (`mcp__osprey__channel_write`) and the server-formatted `tool_name`
 * (`Channel Write`, from operator_session `format_tool_name`) — and both fold
 * to the same key, so the table works whichever spelling an event carries.
 * (Caveat: a tool name containing a literal `__` folds differently per
 * spelling — no framework tool is named that way.)
 *
 * @param {string} name
 * @returns {string}
 */
export function normaliseToolName(name) {
  return (
    name
      // Non-greedy: the FIRST `__` after the server name ends the prefix — the
      // server name may itself contain single underscores (osprey_workspace,
      // osprey_facility_knowledge, or a facility-declared server named that way).
      .replace(/^mcp__.+?__/, '')
      .trim()
      .replace(/[\s-]+/g, '_')
      .toLowerCase()
  );
}

/**
 * The operator phrase for a `tool_use` event, or null when the tool has no
 * table entry. Lower-case and sentence-fragment shaped; capitalise at the
 * point of display.
 * @param {ChatEvent} event
 * @returns {string | null}
 */
export function toolPhrase(event) {
  for (const name of [event.tool_name_raw, event.tool_name]) {
    if (!name) continue;
    const key = normaliseToolName(name);
    const phrase = Object.hasOwn(TOOL_PHRASES, key) ? TOOL_PHRASES[key] : null;
    if (phrase) return phrase;
  }
  return null;
}

/** Longest skill name the status line shows before it is cut. */
const SKILL_NAME_MAX = 60;

/**
 * The activity-line label for a `tool_use` event. A mapped tool reads as a
 * sentence ("Writing control channels…"); anything unmapped keeps its raw name
 * ("Using Queue Reorder…"), so a tool added elsewhere in the codebase is never
 * invisible here — only terse until it earns a phrase. A skill is named once
 * its input says which one ("Using skill diagnose…").
 * @param {ChatEvent} event
 * @returns {string}
 */
export function activityLabel(event) {
  const input = /** @type {{ skill?: unknown } | undefined} */ (event.input);
  const skill = typeof input?.skill === 'string' ? input.skill.trim() : '';
  if (skill && normaliseToolName(event.tool_name_raw ?? event.tool_name ?? '') === 'skill') {
    const shown = skill.slice(0, SKILL_NAME_MAX);
    return `Using skill ${shown}…`;
  }
  const phrase = toolPhrase(event);
  if (phrase !== null) return `${phrase.charAt(0).toUpperCase()}${phrase.slice(1)}…`;
  return `Using ${event.tool_name ?? event.tool_name_raw ?? 'tool'}…`;
}
