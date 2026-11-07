// @ts-check
/*
 * System Health dashboard — the `?demo` fixtures, for local review of the
 * dashboard without a backend. dashboard.js serves them in place of GET
 * /checks when the page URL carries `?demo`.
 */

import { countsText } from './chrome.js';

/** @typedef {import('./dashboard.js').CheckResult} CheckResult */
/** @typedef {import('./dashboard.js').Envelope} Envelope */

/** The poll interval the fixtures report, as the engine's default. */
const DEFAULT_INTERVAL_S = 60;

/** @type {CheckResult[]} */
const DEMO_RESULTS = [
  {
    name: 'configuration.required_settings',
    category: 'configuration',
    status: 'ok',
    message: 'All required settings present',
  },
  {
    name: 'configuration.connector_selected',
    category: 'configuration',
    status: 'ok',
    message: 'Connector selected',
    value: 'epics',
  },
  {
    name: 'control_system',
    category: 'configuration',
    status: 'warning',
    message: 'control_system config changed; restart the web terminal to apply',
  },
  {
    name: 'file_system.workspace_writable',
    category: 'file_system',
    status: 'ok',
    message: 'Workspace writable',
    value: '12 GB free',
    latency_ms: 4,
  },
  {
    name: 'file_system.scratch_writable',
    category: 'file_system',
    status: 'ok',
    message: 'Scratch dir writable',
    latency_ms: 3,
  },
  {
    name: 'providers.anthropic',
    category: 'providers',
    status: 'ok',
    message: 'Anthropic reachable',
    latency_ms: 88,
  },
  {
    name: 'providers.cborg',
    category: 'providers',
    status: 'warning',
    message: 'CBORG slow response',
    latency_ms: 2140,
  },
  {
    name: 'control_system.beam_current',
    category: 'control_system',
    status: 'ok',
    message: 'Beam current',
    value: '401.2 mA',
    latency_ms: 12,
  },
  {
    name: 'control_system.rf_frequency',
    category: 'control_system',
    status: 'ok',
    message: 'RF frequency',
    value: '499.654 MHz',
    latency_ms: 9,
  },
  {
    name: 'control_system.corrector_write',
    category: 'control_system',
    status: 'error',
    message: 'Corrector setpoint unreachable',
    details: 'caput SR:C01:HCM:SP timed out after 5s\nIOC may be offline',
  },
  {
    name: 'web.dashboard_http',
    category: 'web',
    status: 'ok',
    message: 'Dashboard endpoint reachable',
    value: '200',
    latency_ms: 15,
  },
  {
    name: 'claude_cli_pinned',
    category: 'claude_cli_pinned',
    status: 'skip',
    message:
      'on_demand category; run with --full (e.g. `osprey health --full --category claude_cli_pinned`)',
  },
  {
    name: 'model_chat',
    category: 'model_chat',
    status: 'skip',
    message:
      'on_demand category; run with --full (e.g. `osprey health --full --category model_chat`)',
  },
];

/**
 * @param {CheckResult[]} results
 * @param {Partial<Envelope>} [extra]
 * @returns {Envelope}
 */
function demoEnvelope(results, extra) {
  const count = (/** @type {string} */ s) => results.filter((r) => r.status === s).length;
  const ok = count('ok');
  const warnings = count('warning');
  const errors = count('error');
  const skips = count('skip');
  const total = results.length;

  const parts = countsText({ total, ok, warnings, errors, skips });
  const summary = ok + '/' + total + ' checks passed' + (parts ? ' (' + parts + ')' : '');

  return {
    summary,
    ok,
    warnings,
    errors,
    skips,
    total,
    elapsed_ms: 1240,
    deadline_hit: false,
    results,
    stale: false,
    warming: false,
    interval_s: DEFAULT_INTERVAL_S,
    title: 'System Health',
    ...extra,
  };
}

/**
 * Fixtures for local review without a backend. `?demo` is the healthy-ish
 * snapshot; `?demo=stale` and `?demo=warming` exercise the new envelope states.
 *
 * @param {string} variant
 * @returns {Envelope}
 */
export function demoData(variant) {
  if (variant === 'warming') {
    return demoEnvelope([], { warming: true, stale: true });
  }
  if (variant === 'stale') {
    return demoEnvelope(DEMO_RESULTS, { stale: true, deadline_hit: true });
  }
  return demoEnvelope(DEMO_RESULTS);
}
