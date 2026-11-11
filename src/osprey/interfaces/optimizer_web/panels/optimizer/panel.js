// @ts-check
/**
 * OPTIMIZER panel entry: wires the header pickers, the draft form, the
 * approval surface and the live run view to the sidecar, and runs the poll.
 *
 * The poll follows `jobs/current`, so a draft the agent creates over MCP (the
 * service points the current job at every new draft) appears here by itself.
 * Each tick reads the current job; while it is a CREATED draft it re-reads the
 * stored config, and the revision header of THAT response is the one Start
 * sends. The Start button carries the job and revision it was rendered for,
 * and a click refuses when the state has moved on since; it never fetches a
 * revision of its own. The approval card reads the draft's own environment
 * (not the form's pickers) and holds Start until it has; while the draft
 * waits, the poll refreshes that card's readbacks in place (approval-env.js),
 * never its review state or revision. At load it asks
 * `relay/session` which account this session's writes carry, and warns
 * before Start when it names none.
 *
 * No auto-restore anywhere: a cancelled or failed run holds the machine where
 * it is, and restoring is the operator's call (results restore, PR 5).
 *
 * @module panel
 */

import { panelApiPrefix } from '/design-system/js/dom.js';
import { subscribe } from '/design-system/js/theme-manager.js';
import { markPageReady } from '/design-system/js/page-ready.js';
import { createApi, readbackValues, saveDraft, seg } from './api-client.js';
import { createPoller, isActive } from './poller.js';
import {
  approvalEnvKey,
  approvalInit,
  emptyApprovalEnv,
  isDraftChangedConflict,
  plannedEvaluations,
  reduceApproval,
} from './approval-model.js';
import { approvalGateFor, fetchApprovalEnv, readApprovalReadbacks } from './approval-env.js';
import { renderApproval, updateApprovalReadbacks } from './approval-view.js';
import { renderDraftForm } from './draft-form.js';
import { buildConfig, formFromConfig, formFromEnv, savedMessage } from './draft-model.js';
import { renderLivePlot, rethemeLivePlot } from './live-plot.js';
import { renderRunStatus } from './run-view.js';
import { h, replaceChildren } from './hyperscript.js';

/** Every `ControlTarget`; the service 403s one this deployment may not drive. */
export const TARGETS = Object.freeze(['va', 'standin', 'live']);
const LED_MS = 10000;

/**
 * @param {Document} doc
 * @param {string} id
 * @returns {HTMLElement}
 */
function byId(doc, id) {
  const node = doc.getElementById(id);
  if (!node) throw new Error(`optimizer panel: missing #${id}`);
  return node;
}

/**
 * @param {{ api: import('./api-client.js').Api, doc?: Document, setTimer?: any, clearTimer?: any, now?: () => number }} options
 */
export function createPanel({ api, doc = document, setTimer, clearTimer, now = Date.now }) {
  const els = {
    target: /** @type {HTMLSelectElement} */ (byId(doc, 'target-select')),
    env: /** @type {HTMLSelectElement} */ (byId(doc, 'env-select')),
    chip: byId(doc, 'status-chip'),
    chipLabel: byId(doc, 'status-label'),
    banner: byId(doc, 'panel-banner'),
    approval: byId(doc, 'approval-card'),
    form: byId(doc, 'form-card'),
    run: byId(doc, 'run-card'),
    plot: byId(doc, 'live-plot'),
  };

  const state = {
    environment: '',
    target: '',
    /** @type {import('./approval-model.js').EnvironmentDetails|null} */
    env: null,
    /** @type {Record<string, any>} */
    presets: {},
    /** @type {any[]} */
    models: [],
    /** @type {Record<string, number|null>} */
    readbacks: {},
    /** @type {import('./draft-model.js').FormState|null} */
    form: null,
    /** @type {{ tone: string, text: string }|null} */
    formMessage: null,
    /** The draft the form was last loaded from, so a poll never clobbers edits. */
    formJobId: /** @type {string|null} */ (null),
    /** @type {any} */
    job: null,
    approval: approvalInit(),
    starting: false,
    /** @type {string|null} */
    startError: null,
    /** @type {string|null} */
    runError: null,
    cancelArmed: false,
    /** `job_id:data_version` of the figure on screen. */
    plotKey: '',
    renderedKey: '',
    runKey: '',
    /** The approval card's own read of the draft's environment. */
    approvalEnv: emptyApprovalEnv(''),
    /** When the approval card's readbacks were last read (ms). */
    approvalReadAt: 0,
    /** Whether a readback read is in flight: at most one at a time. */
    approvalReading: false,
    /** The account `relay/session` named; `ownerKnown` is null until it answers. */
    owner: /** @type {string|null} */ (null),
    ownerKnown: /** @type {boolean|null} */ (null),
  };

  /** @param {string|null} text */
  function setBanner(text) {
    els.banner.hidden = !text;
    els.banner.textContent = text || '';
  }

  /**
   * @param {'ok'|'warn'|'err'|'connecting'} tone
   * @param {string} text
   */
  function setChip(tone, text) {
    els.chip.className = `osprey-chip osprey-chip--${tone}`;
    els.chipLabel.textContent = text;
    els.chip.setAttribute('aria-label', `Optimizer status: ${text}`);
  }

  async function refreshLed() {
    const health = await api.get('service/health');
    if (!health.ok) return setChip('err', health.error || 'service unreachable');
    if (!state.environment || !state.target)
      return setChip('warn', 'pick a target and environment');
    const res = await api.get(`environments/${seg(state.environment)}/status`, {
      target: state.target,
    });
    if (!res.ok) return setChip('err', res.error || 'status unavailable');
    const { connected, message } = res.data || {};
    if (connected === true) setChip('ok', `${state.environment} @ ${state.target}`);
    else if (connected === false) setChip('err', message || 'environment not connected');
    else setChip('warn', message || 'connection unknown');
  }

  async function loadEnvironments() {
    const res = await api.get('environments');
    if (!res.ok) return setBanner(res.error);
    replaceChildren(
      els.env,
      h('option', { value: '', text: 'Environment…' }),
      ...(res.data || []).map((/** @type {any} */ e) =>
        h('option', {
          value: e.name,
          text:
            e.valid === false ? `${e.display_name || e.name} (invalid)` : e.display_name || e.name,
          disabled: e.valid === false,
        })
      )
    );
    els.env.value = state.environment;
  }

  async function refreshReadbacks() {
    if (!state.env || !state.target) return;
    const names = Object.keys(state.env.variables || {});
    if (!names.length) return;
    const res = await api.get(`environments/${seg(state.environment)}/variables`, {
      names,
      target: state.target,
    });
    if (!res.ok) {
      state.formMessage = { tone: 'warn', text: `Readbacks unavailable: ${res.error}` };
    } else {
      state.readbacks = readbackValues(res.data);
    }
    renderForm();
  }

  /**
   * Seed the form from the preset (or the safety ranges) with the service's
   * resolved initial bounds on the chosen target.
   *
   * @param {string} preset
   */
  async function seedForm(preset) {
    if (!state.env) return;
    const base = state.form?.base || {};
    let bounds = null;
    if (state.target) {
      const res = await api.get(`environments/${seg(state.environment)}/initial-bounds`, {
        target: state.target,
        preset: preset || null,
      });
      // Its current_values are setpoints; the Readback column comes from
      // environments/{name}/variables (refreshReadbacks) instead.
      if (res.ok) {
        bounds = res.data?.bounds || null;
      } else {
        state.formMessage = {
          tone: 'warn',
          text: `Initial bounds unavailable (safety ranges shown): ${res.error}`,
        };
      }
    }
    state.form = {
      ...formFromEnv(state.env, { name: preset, preset: state.presets[preset] || null, bounds }),
      base,
    };
    renderForm();
  }

  /**
   * @param {string} name
   * @param {import('./approval-model.js').OptimizationConfig|null} [fromConfig]
   */
  async function selectEnvironment(name, fromConfig = null) {
    state.environment = name;
    state.env = null;
    state.form = null;
    state.readbacks = {};
    state.formMessage = null;
    els.env.value = name;
    if (!name) return renderForm();
    const [details, presets, models] = await Promise.all([
      api.get(`environments/${seg(name)}`),
      api.get(`environments/${seg(name)}/presets`),
      api.get('models'),
    ]);
    if (!details.ok) {
      state.formMessage = { tone: 'err', text: details.error || 'environment unavailable' };
      return renderForm();
    }
    state.env = details.data;
    state.presets = presets.ok ? presets.data || {} : {};
    state.models = models.ok ? models.data || [] : [];
    if (fromConfig) state.form = formFromConfig(fromConfig, /** @type {any} */ (state.env));
    else await seedForm('');
    void refreshReadbacks();
    void refreshLed();
  }

  /**
   * Save the form as a new draft or over the current one, then re-seed the
   * form from what the service stored. The job and environment are captured
   * before the write: a poll during it may replace or clear `state.job`, and
   * the form is re-seeded only while it still belongs to that same draft.
   *
   * @param {'create'|'update'} kind
   */
  async function submitDraft(kind) {
    if (!state.form) return;
    const { config, errors } = buildConfig(state.form, {
      environment: state.environment,
      target: state.target,
    });
    if (!config) {
      state.formMessage = { tone: 'err', text: errors.join(' ') };
      return renderForm();
    }
    const env = state.env;
    const jobId = kind === 'update' ? state.job?.job_id || null : null;
    try {
      if (kind === 'update' && !jobId) throw new Error('the draft is gone; nothing was saved');
      const { error, id, stored } = await saveDraft(api, jobId, config);
      if (kind === 'create' && id) state.formJobId = id;
      // Re-seed only while the form still belongs to that draft and environment.
      const ours = state.env === env && (jobId ? state.job?.job_id : state.formJobId) === id;
      if (stored && env && ours) state.form = formFromConfig(stored, env);
      state.formMessage = error ? { tone: 'err', text: error } : savedMessage(kind, config, stored);
    } catch (err) {
      state.formMessage = {
        tone: 'err',
        text: `Saving failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
    renderForm();
    await poller.now();
  }

  /**
   * Read the stored draft's environment (details, models, and readbacks on
   * the draft's own target) for the approval card, then re-render it. A
   * failed read is retried on the next tick; until one succeeds the direction
   * reads unverified and Start is held.
   */
  async function loadApprovalEnv() {
    const config = state.approval.config;
    const key = approvalEnvKey(config);
    if (key === state.approvalEnv.key && !state.approvalEnv.error) return;
    state.approvalEnv = emptyApprovalEnv(key);
    if (!config || !key) return;
    state.approvalReadAt = now();
    const loaded = await fetchApprovalEnv(api, config, key);
    if (state.approvalEnv.key !== key) return; // the draft moved on meanwhile
    state.approvalEnv = loaded;
    renderApprovalCard();
  }

  /**
   * Re-read the waiting draft's readbacks and write them into the card in
   * place: no re-render, so the review state and revision stay as they are.
   * One read at a time; a reply for an environment the card no longer shows
   * (the draft moved on, or the environment was re-read) is dropped.
   */
  async function refreshApprovalReadbacks() {
    const readbacks = await readApprovalReadbacks(api, state, now);
    if (!readbacks) return; // not due, failed, or moved on
    state.approvalEnv = { ...state.approvalEnv, readbacks };
    updateApprovalReadbacks(els.approval, readbacks);
  }

  const approvalGate = () => approvalGateFor(state);

  /**
   * Start the draft the clicked card was rendered for, and only that one.
   *
   * @param {{ jobId: string|null, revision: string|null }} shown
   */
  async function start(shown) {
    const { job, approval } = state;
    // The card clicked is not the one the state now holds (a poll moved the
    // draft on under the cursor): show the current one, start nothing.
    if (shown.jobId !== approval.jobId || shown.revision !== approval.revision) {
      return renderApprovalCard();
    }
    if (!approvalGate().enabled) return;
    state.starting = true;
    state.startError = null;
    renderApprovalCard();
    // The revision that rendered the surface the operator just read — never
    // a fresh read at click time.
    const res = await api.post(`relay/jobs/${seg(job.job_id)}/start`, undefined, {
      config_revision: shown.revision,
    });
    state.starting = false;
    if (res.ok) {
      state.job = res.data;
    } else if (isDraftChangedConflict(res.status, res.error)) {
      state.approval = reduceApproval(state.approval, { type: 'conflict' });
    } else {
      state.startError = res.error;
    }
    renderApprovalCard();
    await poller.now();
  }

  /** @param {'pause'|'resume'|'cancel'} action */
  async function control(action) {
    if (!state.job) return;
    if (action === 'cancel' && !state.cancelArmed) {
      state.cancelArmed = true;
      return renderRun();
    }
    state.cancelArmed = false;
    const res = await api.post(`relay/jobs/${seg(state.job.job_id)}/${action}`);
    state.runError = res.ok ? null : res.error;
    if (res.ok) state.job = res.data;
    renderRun();
    await poller.now();
  }

  function renderForm() {
    if (!state.env || !state.form) {
      const text =
        state.formMessage?.text ||
        (state.environment
          ? 'Loading environment…'
          : 'Pick a control target and an environment to draft a run.');
      return replaceChildren(
        els.form,
        h('p', { class: state.formMessage ? 'banner banner-err' : 'muted', text })
      );
    }
    const status = state.job?.status;
    const draft = status === 'created' ? state.job : null;
    renderDraftForm(
      els.form,
      state.form,
      {
        env: state.env,
        presets: state.presets,
        models: state.models,
        readbacks: state.readbacks,
        canCreate: !isActive(status),
        canUpdate: !!draft && state.approval.config?.environment_name === state.environment,
        lockedReason: isActive(status)
          ? `A run is ${status}; drafts wait until it ends.`
          : state.target
            ? null
            : 'Pick a control target first.',
        message: state.formMessage,
      },
      {
        onPreset: (name) => void seedForm(name),
        onRefreshReadbacks: () => void refreshReadbacks(),
        onLoadDraft:
          draft && state.approval.config
            ? () => void selectEnvironment(state.environment, state.approval.config)
            : null,
        onSubmit: (kind) => void submitDraft(kind),
      }
    );
  }

  function renderApprovalCard() {
    const loaded = state.approvalEnv.key === approvalEnvKey(state.approval.config);
    // What this card shows; its Start starts exactly this, or nothing.
    const shown = { jobId: state.approval.jobId, revision: state.approval.revision };
    renderApproval(
      els.approval,
      {
        approval: state.approval,
        env: loaded ? state.approvalEnv.details : null,
        models: loaded ? state.approvalEnv.models : null,
        readbacks: loaded ? state.approvalEnv.readbacks : {},
        gate: approvalGate(),
        status: state.job?.status ?? null,
        owner: state.owner,
        startError: state.startError,
      },
      {
        onStart: () => void start(shown),
        onReviewed: () => {
          state.approval = reduceApproval(state.approval, { type: 'reviewed' });
          renderApprovalCard();
        },
      }
    );
  }

  /**
   * Re-render what a poll changed, and only that: rebuilding the form or a
   * button under the operator's cursor every second would eat their focus
   * and their clicks.
   */
  function renderPolled() {
    const { job, approval } = state;
    const key = [job?.job_id, job?.status, approval.jobId, approval.revision, approval.needsReview];
    if (key.join('|') !== state.renderedKey) {
      state.renderedKey = key.join('|');
      renderApprovalCard();
      renderForm();
    }
    renderRun();
  }

  function renderRun() {
    const job = state.job;
    els.run.hidden = !job || job.status === 'created';
    if (els.run.hidden) return;
    const runKey = JSON.stringify([
      job.job_id,
      job.status,
      job.data_version,
      job.write_authorization,
      job.error_message,
      state.cancelArmed,
      state.runError,
      state.approval.jobId,
    ]);
    if (runKey === state.runKey) return;
    state.runKey = runKey;
    // The total before the first data arrives, from the job's own stored config.
    const planned = state.approval.jobId === job.job_id ? state.approval.config : null;
    renderRunStatus(
      byId(doc, 'run-status'),
      job,
      {
        cancelArmed: state.cancelArmed,
        runError: state.runError,
        plannedTotal: plannedEvaluations(planned),
      },
      { onControl: (action) => void control(action) }
    );
    const plotKey = `${job.job_id}:${job.data_version}`;
    if (plotKey !== state.plotKey && job.data) {
      if (renderLivePlot(els.plot, job)) state.plotKey = plotKey;
    }
  }

  /** @returns {Promise<import('./poller.js').TickResult>} */
  async function tick() {
    const cur = await api.get('jobs/current');
    if (!cur.ok) {
      setBanner(cur.error);
      return { failed: true };
    }
    setBanner(null);
    const id = cur.data?.job_id || null;
    // A pointer at a draft that expired (404) reads as no current job.
    const basic = id ? await api.get(`jobs/${seg(id)}`) : null;
    if (!id || !basic || basic.status === 404) {
      state.job = null;
      state.approval = reduceApproval(state.approval, { type: 'clear' });
    } else {
      if (!basic.ok) {
        setBanner(basic.error);
        return { failed: true };
      }
      let job = basic.data;
      if (job.status !== 'created') {
        const full = await api.get(`jobs/${seg(id)}`, { detail: 'full' });
        if (full.ok) job = full.data;
      }
      state.job = job;
      if (job.status === 'created' || state.approval.jobId !== id) {
        const cfg = await api.get(`jobs/${seg(id)}/config`);
        if (cfg.ok) {
          state.approval = reduceApproval(state.approval, {
            type: 'draft',
            jobId: id,
            revision: cfg.revision,
            config: cfg.data,
          });
          // Show the card for the state just adopted before any further
          // await, so no older card stays on screen while the form follows.
          renderPolled();
          void loadApprovalEnv();
          await followDraft(id, cfg.data, job.status);
        }
      }
    }
    if (!state.approval.config) void loadApprovalEnv();
    renderPolled();
    void refreshApprovalReadbacks();
    return { status: state.job?.status ?? null };
  }

  /**
   * Adopt a newly seen draft's target and environment, and load it into the
   * form once — later revisions never overwrite the operator's edits.
   *
   * @param {string} id
   * @param {any} config
   * @param {string} status
   */
  async function followDraft(id, config, status) {
    if (status !== 'created' || state.formJobId === id) return;
    state.formJobId = id;
    if (config.control_target && config.control_target !== state.target) {
      state.target = config.control_target;
      els.target.value = state.target;
    }
    await selectEnvironment(config.environment_name || '', config);
  }

  const poller = createPoller({ tick, setTimer, clearTimer });

  replaceChildren(
    els.target,
    h('option', { value: '', text: 'Target…' }),
    ...TARGETS.map((t) => h('option', { value: t, text: t }))
  );
  els.target.addEventListener('change', () => {
    state.target = els.target.value;
    void (state.environment ? selectEnvironment(state.environment) : refreshLed());
  });
  els.env.addEventListener('change', () => void selectEnvironment(els.env.value));
  subscribe(() => rethemeLivePlot(els.plot));

  return {
    state,
    poller,
    async boot() {
      setChip('connecting', 'connecting…');
      renderForm();
      renderApprovalCard();
      // Which account this session's writes carry; a failed read leaves it unknown.
      void api.session().then((owner) => (Object.assign(state, owner), renderApprovalCard()));
      await loadEnvironments();
      void refreshLed();
      poller.start();
      return setInterval(() => void refreshLed(), LED_MS);
    },
  };
}

if (typeof document !== 'undefined' && document.getElementById('optimizer-root')) {
  // Ready once boot() has drawn the first state (index.html declares the mark
  // pending until then).
  void createPanel({ api: createApi({ prefix: panelApiPrefix() }) })
    .boot()
    .finally(() => markPageReady());
}
