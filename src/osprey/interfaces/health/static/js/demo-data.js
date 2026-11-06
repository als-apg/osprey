// @ts-check
/*
 * System Health dashboard — `?demo` fixture rows and component builder.
 *
 * The result rows of the demo envelope (one per check, already carrying the
 * panel's layer / component / label / method) and the helper that folds row
 * indices into a component in the /checks `layers` shape.
 */

export const NOW = () => Date.now() / 1000;

/**
 * [component, label, status, value, method, extra] → one result row.
 *
 * @param {string} layer
 * @param {string} component
 * @param {string} name
 * @param {string} label
 * @param {string} status
 * @param {string} value
 * @param {string} method
 * @param {Record<string, unknown>} [extra]
 */
export function row(layer, component, name, label, status, value, method, extra) {
  return {
    name,
    category: name.split('.')[0],
    status,
    message: value,
    value,
    layer,
    component,
    label,
    method,
    group: null,
    probed_at: NOW() - 14,
    ...extra,
  };
}

/** @returns {any[]} */
export function demoRows() {
  return [
    /* 0 */ row(
      'op',
      'web-terminal',
      'web_terminals.alice',
      'alice',
      'ok',
      'runs as osprey',
      'config parse web_terminals'
    ),
    /* 1 */ row(
      'op',
      'web-terminal',
      'web_terminals.bob',
      'bob',
      'ok',
      'runs as osprey',
      'config parse web_terminals'
    ),
    /* 2 */ row(
      'op',
      'panels',
      'web_panels.ariel',
      'HTTP /health',
      'ok',
      'up · 480 ms',
      'http GET /health',
      { group: 'logbook' }
    ),
    /* 3 */ row(
      'op',
      'panels',
      'reach.postgresql',
      'backend: ARIEL database',
      'ok',
      'up · 424 ms',
      'tcp connect',
      { group: 'logbook' }
    ),
    /* 4 */ row(
      'op',
      'panels',
      'web_panels.okf',
      'HTTP /health',
      'ok',
      'up · 444 ms',
      'http GET /health',
      { group: 'knowledge' }
    ),
    /* 5 */ row(
      'agent',
      'claude-code',
      'claude_cli_version',
      'CLI version',
      'ok',
      '2.1.300',
      'exec claude --version'
    ),
    /* 6 */ row(
      'agent',
      'model-provider',
      'als-apg',
      'als-apg',
      'ok',
      'authenticated · 1662 ms',
      'http GET /v1/models with key',
      { category: 'providers' }
    ),
    /* 7 */ row(
      'agent',
      'model-provider',
      'cborg',
      'cborg',
      'skip',
      'not used by this deployment',
      'env lookup',
      { category: 'providers' }
    ),
    /* 8 */ row(
      'agent',
      'model-provider',
      'openai',
      'openai',
      'skip',
      'not used by this deployment',
      'env lookup',
      { category: 'providers' }
    ),
    /* 9 */ row(
      'agent',
      'embed-vision',
      'ollama',
      'ollama',
      'ok',
      'up · 469 ms',
      'http GET /api/tags',
      { category: 'providers' }
    ),
    /* 10 */ row(
      'chain',
      'hooks',
      'hooks.osprey_writes_check',
      'osprey_writes_check',
      'ok',
      'registered · sha = render',
      'JSON parse settings.json + sha256'
    ),
    /* 11 */ row(
      'chain',
      'hooks',
      'hooks.osprey_limits',
      'osprey_limits',
      'ok',
      'registered · sha = render',
      'JSON parse settings.json + sha256'
    ),
    /* 12 */ row(
      'chain',
      'hooks',
      'hooks.osprey_approval',
      'osprey_approval',
      'ok',
      'registered · sha = render',
      'JSON parse settings.json + sha256'
    ),
    /* 13 */ row(
      'chain',
      'write-switch',
      'write_switch.alice',
      'alice on virtual_accelerator',
      'ok',
      'asked',
      'state record read'
    ),
    /* 14 */ row(
      'chain',
      'limits',
      'limits.parse',
      'parse',
      'ok',
      '1,284 channels · 0 errors',
      'YAML parse limits file'
    ),
    /* 15 */ row(
      'data',
      'graph-store',
      'graphdb_connection',
      'bolt connection',
      'ok',
      'open · 11 ms',
      'bolt connect + RETURN 1',
      { used_by: ['channel finder', 'graph MCP server'] }
    ),
    /* 16 */ row(
      'data',
      'graph-store',
      'graphdb_resources',
      'Resource nodes',
      'ok',
      '5,889',
      'bolt count(Resource)'
    ),
    /* 17 */ row(
      'data',
      'telemetry-store',
      'agent_record_backend',
      'backend',
      'ok',
      'clickhouse · 76 entries',
      'sql SELECT 1',
      { group: 'store' }
    ),
    /* 18 */ row(
      'data',
      'telemetry-store',
      'agent_record.retention',
      'retention',
      'ok',
      '90 d',
      'config parse agent_record.retention_days',
      { group: 'store' }
    ),
    /* 19 */ row(
      'data',
      'telemetry-store',
      'agent_record_transcripts.alice',
      'alice',
      'ok',
      'readable',
      'dir stat /data/transcripts/alice',
      { group: 'transcripts' }
    ),
    /* 20 */ row(
      'data',
      'telemetry-store',
      'agent_record_transcripts.logbook',
      'logbook',
      'warning',
      'root unreadable (ENOENT)',
      'dir stat /data/transcripts/logbook',
      { group: 'transcripts', fix: 'docker compose exec agent-record ls /data/transcripts' }
    ),
    /* 21 */ row(
      'out',
      'cs-connector',
      'reach.virtual_accelerator',
      'virtual_accelerator',
      'ok',
      'up · 423 ms',
      'tcp connect'
    ),
    /* 22 */ row('cfg', 'config-yml', 'yaml_valid', 'syntax', 'ok', '0 errors', 'YAML parse'),
    /* 23 */ row(
      'cfg',
      'config-yml',
      'model_configs',
      'models section',
      'skip',
      'absent (optional)',
      'config parse models:'
    ),
    /* 24 */ row(
      'host',
      'disk',
      'disk_space',
      'project volume',
      'ok',
      '30 % used · 1308.4 GB free',
      'statvfs'
    ),
    /* 25 */ row(
      'host',
      'container-runtime',
      'container_runtime',
      'container runtime',
      'skip',
      'n/a in container',
      'docker ps'
    ),
  ];
}

/**
 * @param {any[]} rows
 * @param {number[]} idx
 */
function worst(rows, idx) {
  const rank = { error: 3, warning: 2, ok: 1, skip: 0 };
  let w = 'skip';
  for (const i of idx) {
    const s = rows[i].status;
    if (rank[/** @type {'ok'} */ (s)] > rank[/** @type {'ok'} */ (w)]) w = s;
  }
  return w;
}

/**
 * @param {any[]} rows
 * @param {string} id
 * @param {string} name
 * @param {string} fact
 * @param {Record<string, any>} [o]
 */
export function comp(rows, id, name, fact, o) {
  const opts = o || {};
  /** @type {number[]} */
  const own = opts.rows || [];
  /** @type {any[]} */
  const groups = (opts.groups || []).map((/** @type {any} */ g) => ({
    fix: null,
    summary: '',
    ...g,
    status: worst(rows, g.rows),
  }));
  /** @type {number[]} */
  const unused = opts.unused || [];
  const all = own.concat(...groups.map((g) => g.rows), unused);
  const status = worst(rows, all);
  return {
    id,
    name,
    status,
    fact,
    issue: opts.issue || null,
    header: opts.header || null,
    used_by: opts.used_by || [],
    rows: own,
    groups,
    unused,
  };
}
