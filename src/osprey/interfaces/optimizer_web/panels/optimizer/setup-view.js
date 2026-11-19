// @ts-check
/**
 * The set-up screens (`setup`, `setup-issues`): a draft is made or edited
 * here, and saved. Saving checks the draft and stores it; it writes nothing
 * to the machine. The service takes a draft also while a run is under way.
 *
 * The stage shows what the draft does: the plain sentence, the objective with
 * its value now, and each knob on the knob bar (its safe range, its search
 * window, its value now). The inspector edits it: the goal, the knobs (the
 * knob picker), the search and the end action; its foot and the toolbar hold
 * Save.
 *
 * The form (draft-model.js) is the operator's editor. It is seeded once per
 * set-up and environment: a set-up that edits a draft only from a config read
 * for that draft (never from defaults: if none was read, it says so and holds
 * Save); a new run from the config it runs again, or from the environment
 * with the service's initial windows on the picked target. A poll never
 * re-seeds it. When the draft changes on the service while it is open here,
 * Save is held until the operator loads the new version or keeps the edits.
 * Another screen in between keeps the edits: the form waits for its set-up.
 *
 * A view that another screen replaced does nothing more: a late answer finds
 * it dead and touches neither the zones nor the toolbar.
 *
 * @module setup-view
 */

import { renderState } from '/design-system/js/panel-state.js';
import { initialSamples } from './approval-model.js';
import { buildConfig, formFromConfig, formFromEnv } from './draft-model.js';
import { draftStage } from './draft-stage.js';
import { h, replaceChildren } from './hyperscript.js';
import { createKnobPicker } from './knob-picker.js';
import {
  endFields,
  goalFields,
  issueLine,
  searchFields,
  whereFields,
  writeReadouts,
} from './setup-form.js';
import {
  draftMoved,
  groupIssues,
  issueFolders,
  knobIssues,
  missingKnobs,
  previewConfig,
  seedSource,
  withPreset,
} from './setup-model.js';
import {
  emptyStage,
  missingKnobsNotice,
  rereadButton,
  setupFolder,
  setupNotices,
  unreadList,
} from './setup-notices.js';
import { readingKey } from './store.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./draft-model.js').FormState} FormState
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 *
 * What the view holds of one set-up, kept across a remount.
 * @typedef {{
 *   form: FormState|null,
 *   seededFor: string,
 *   seededTarget: string,
 *   touched: boolean,
 *   accepted: string|null,
 *   message: string|null,
 *   folders: Record<string, boolean>,
 *   advancedOpen: boolean,
 * }} Memo
 */

/**
 * The form of the set-up the view showed when another screen replaced it, by
 * set-up id, so the operator's edits are there when that set-up shows again.
 *
 * @type {{ id: number, memo: Memo }|null}
 */
let kept = null;

/** @returns {Memo} */
function freshMemo() {
  return {
    form: null,
    seededFor: '',
    seededTarget: '',
    touched: false,
    accepted: null,
    message: null,
    folders: { goal: true, knobs: true, search: false, end: false },
    advancedOpen: false,
  };
}

/** @type {import('./view-host.js').MountView} */
export function mountSetupView(ctx) {
  const { actions, shell } = ctx;
  const scroll = h('div', { class: 'opt-stage-scroll opt-draft-stage' });
  ctx.stage.replaceChildren(scroll);

  const body = h('div', { class: 'opt-su-body' });
  const pending = h('span', { class: 'osprey-pending-line', 'aria-live': 'polite' });
  const saveBtn = /** @type {HTMLButtonElement} */ (
    h(
      'button',
      { type: 'button', class: 'osprey-btn primary', id: 'opt-su-save', onclick: () => submit() },
      'Save draft'
    )
  );
  const discardBtn = h('button', {
    type: 'button',
    class: 'osprey-btn ghost',
    id: 'opt-su-discard',
    text: 'Discard',
    'data-tip-detail': 'Closes the set-up. Nothing was written.',
    onclick: () => actions.closeSetup(),
  });
  const picker = createKnobPicker({
    onChange: (structural) => changed(structural),
    onPreset: (name) => void pickPreset(name),
  });

  let m = freshMemo();
  /** False once another screen replaced the view: nothing more is drawn or sent. */
  let alive = true;
  let restored = false;
  /** Bumped on each seed, so a late answer for an older one is dropped. */
  let seeds = 0;
  /** Bumped on each edit, so a seed that waited keeps what was typed meanwhile. */
  let edits = 0;
  let frameKey = '';
  let inspectorKey = '';
  let stageKey = '';
  let variablesRead = '';
  let objectiveRead = '';
  /** @type {State|null} */
  let latest = null;

  const where = () => ({
    environment: latest?.setup?.environment || '',
    target: latest?.setup?.target || '',
  });
  /** Read the environment of the set-up again (after a read that failed). */
  const readEnvironment = () => void actions.loadEnvironment(where().environment);
  /** Read the environments and the model files again (after a read that failed). */
  const readCatalog = () => void actions.loadCatalog();
  const details = () => latest?.envs[where().environment]?.details || null;
  const issues = () =>
    groupIssues(latest?.setup?.issues || [], Object.keys(details()?.variables || {}));
  /** Why Save is held besides a missing form or a save in flight, or null. */
  function held() {
    if (!latest?.setup) return null;
    if (latest.setup.jobId && !seedSource(latest).config) {
      return 'The stored draft could not be read.';
    }
    if (draftMoved(latest, m.accepted)) return 'The stored draft changed on the service.';
    return null;
  }

  /**
   * Seed a fresh form for the environment, with the service's initial
   * windows on the target when one is picked. What the operator typed while
   * the windows were read is kept.
   *
   * @param {string} preset  a preset to start from ('' for none)
   */
  async function seedFresh(preset) {
    const env = details();
    const { environment, target } = where();
    if (!env) return;
    const seed = ++seeds;
    const editsAtStart = edits;
    const varsAtStart = m.form ? JSON.stringify(m.form.vars) : '';
    // Taken now, so the poll's next update does not seed the same target again.
    m.seededTarget = target;
    m.message = null;
    let bounds = null;
    if (target) {
      const res = await actions.initialBounds(environment, target, preset);
      if (!alive || seed !== seeds) return;
      bounds = res.bounds;
      if (res.error) {
        m.message = `The initial windows could not be read, so the safe ranges are shown: ${res.error}`;
      }
    }
    const chosen = latest?.envs[environment]?.presets?.[preset] || null;
    const seeded = formFromEnv(env, { name: preset, preset: chosen, bounds });
    const editedMeanwhile = edits !== editsAtStart;
    if (m.form && preset) m.form = withPreset(m.form, seeded, chosen, env);
    else if (m.form && editedMeanwhile) {
      const varsKept = JSON.stringify(m.form.vars) !== varsAtStart;
      m.form = { ...m.form, vars: varsKept ? m.form.vars : seeded.vars };
    } else m.form = seeded;
    m.touched = !!preset || editedMeanwhile;
    rebuild();
  }

  /** @param {string} name */
  async function pickPreset(name) {
    if (!m.form) return;
    if (!name) {
      m.form.preset = '';
      return changed(true);
    }
    await seedFresh(name);
  }

  /** Seed the form when the set-up or the environment it edits changed. */
  function seed() {
    const state = latest;
    const setup = state?.setup;
    const env = details();
    if (!state || !setup || !env) return;
    const key = `${setup.id}|${setup.jobId}|${setup.environment}`;
    if (key === m.seededFor) {
      const source = setup.jobId && !m.form ? seedSource(state) : null;
      // A stored draft that could not be read when the set-up opened, read since.
      if (source?.config?.environment_name === setup.environment)
        seedFrom(source.config, source.revision, env);
      // A fresh form the operator has not touched follows the target: its
      // initial windows are where the knobs are on that target.
      else if (!setup.jobId && !setup.from && !m.touched && setup.target !== m.seededTarget)
        void seedFresh('');
      return;
    }
    m.seededFor = key;
    ++seeds;
    m.message = null;
    const source = seedSource(state);
    if (source.config && source.config.environment_name === setup.environment) {
      seedFrom(source.config, source.revision, env);
    } else if (setup.jobId && !source.config) {
      // Never defaults over a stored draft: no form, and Save is held.
      m.form = null;
      rebuild();
    } else {
      m.form = null;
      void seedFresh('');
    }
  }

  /**
   * @param {import('./approval-model.js').OptimizationConfig} config
   * @param {string|null} revision
   * @param {EnvironmentDetails} env
   */
  function seedFrom(config, revision, env) {
    m.form = formFromConfig(config, env);
    m.touched = false;
    m.seededTarget = latest?.setup?.target || '';
    m.accepted = revision;
    rebuild();
  }

  /** @param {boolean} structural  a knob, a constraint or a choice that redraws fields */
  function changed(structural) {
    m.touched = true;
    edits += 1;
    if (structural) rebuild();
    else refresh();
  }

  /** @param {'create'|'update'} [kind] */
  function submit(kind) {
    const setup = latest?.setup;
    if (!alive || !m.form || !setup || setup.saving || held()) return;
    const { config, errors } = buildConfig(m.form, {
      environment: setup.environment,
      target: setup.target,
    });
    if (!config) return actions.refuseSetup(errors);
    void actions.saveDraft(config, kind ?? (setup.jobId ? 'update' : 'create'));
  }

  /** Redraw the inspector's fields (after a change of their structure), keeping the focus. */
  function rebuild() {
    const state = latest;
    const setup = state?.setup;
    if (!alive || !state || !setup) return;
    const focused = /** @type {HTMLElement|null} */ (ctx.inspector.ownerDocument.activeElement);
    const focusId = focused && ctx.inspector.contains(focused) ? focused.id : '';
    const { form } = m;
    const env = details();
    const entry = state.envs[setup.environment];
    const byField = issues();
    const named = issueFolders(byField);
    const c = {
      form,
      env,
      environments: state.environments.list || [],
      environment: setup.environment,
      target: setup.target,
      models: state.models.list || [],
      issues: byField,
      unread: {
        environments: unreadList(
          'opt-su-env-reread',
          'The environments',
          state.environments.error,
          readCatalog
        ),
        models: unreadList(
          'opt-su-model-reread',
          'The model files',
          state.models.error,
          readCatalog
        ),
      },
    };
    const hooks = { where: actions.setSetupWhere, changed };
    const preview = form ? previewConfig(form, where()) : null;
    const auto = preview ? initialSamples({ ...preview, n_initial_samples: null }) : null;
    /** @type {(key: string, title: string, content: (Node|null)[]) => HTMLElement} */
    const folder = (key, title, content) =>
      setupFolder(key, title, content, m.folders[key] || named.has(key), (open) => {
        m.folders[key] = open;
      });
    replaceChildren(
      body,
      h('p', {
        class: 'opt-note',
        text: 'Saving checks the draft and stores it. It writes nothing to the machine.',
      }),
      ...setupNotices(state, {
        hasForm: !!form,
        unreadable: !!setup.jobId && !seedSource(state).config,
        moved: draftMoved(state, m.accepted),
        message: m.message,
        onSaveNew: () => submit('create'),
        onLoadNew: () => {
          const approval = latest?.approval;
          const now = details();
          if (approval?.config && now) seedFrom(approval.config, approval.revision, now);
        },
        onKeepMine: () => {
          m.accepted = latest?.approval.revision ?? m.accepted;
          rebuild();
        },
        onReadEnvironment: readEnvironment,
      }),
      folder('goal', 'Goal', [
        h(
          'div',
          { class: 'osprey-form-grid', style: '--cols: 1' },
          ...whereFields(c, hooks),
          ...goalFields(c, hooks)
        ),
      ]),
      form && env
        ? folder('knobs', 'Knobs', [
            missingKnobsNotice(missingKnobs(form.base, env)),
            ...[...(byField.knobs || []), ...(byField.preset || [])].map(issueLine),
            picker.el,
          ])
        : null,
      form
        ? folder(
            'search',
            'Search',
            searchFields(c, hooks, {
              initial: auto?.count == null ? 'auto' : `auto (${auto.count})`,
              advancedOpen: m.advancedOpen,
              onAdvanced: (open) => (m.advancedOpen = open),
            })
          )
        : null,
      form ? folder('end', 'When it ends', endFields(c, hooks)) : null
    );
    if (form && env) {
      const values =
        state.readings[readingKey('variables', setup.environment, setup.target)]?.values || {};
      picker.render({
        form,
        env,
        presets: entry?.presets || {},
        values,
        issues: knobIssues(byField),
      });
    }
    const back = focusId ? ctx.inspector.querySelector(`[id="${focusId}"]`) : null;
    /** @type {HTMLElement|null} */ (back)?.focus({ preventScroll: true });
    refresh();
  }

  /** Draw the stage and the figures of the inspector from the form as it stands. */
  function refresh() {
    const state = latest;
    const setup = state?.setup;
    if (!alive || !state || !setup) return;
    readNow();
    const env = details();
    const entry = state.envs[setup.environment];
    if (!m.form || !env) {
      renderState(scroll, {
        icon: 'sliders',
        ...emptyStage(setup, entry, held(), readEnvironment),
      });
    } else {
      drawStage(state, env);
      writeReadouts(body, m.form, env, previewConfig(m.form, where()));
    }
    const n = setup.issues.length;
    pending.textContent = setup.saving
      ? 'Saving…'
      : n
        ? `${n} problem${n === 1 ? '' : 's'}`
        : 'Not saved';
    const why =
      held() ||
      (setup.saving
        ? 'Saving…'
        : setup.environment
          ? 'The environment is loading.'
          : 'Pick an environment first.');
    saveBtn.disabled = !m.form || setup.saving || !!held();
    saveBtn.title = saveBtn.disabled ? why : 'Save draft';
    shell.setPrimary({
      label: 'Save draft',
      id: 'save-draft-btn',
      disabled: saveBtn.disabled,
      why,
      detail: 'Checks the draft and stores it. Writes nothing to the machine.',
      onClick: () => submit(),
    });
  }

  /**
   * @param {State} state
   * @param {EnvironmentDetails} env
   */
  function drawStage(state, env) {
    if (!m.form) return;
    const { environment, target } = where();
    const reading = state.readings[readingKey('variables', environment, target)];
    const observed = state.readings[readingKey('observables', environment, target)];
    const flags = Object.fromEntries(
      Object.entries(knobIssues(issues())).map(([name, list]) => [
        name,
        /** @type {import('./config-sections.js').Flag} */ ({ tone: 'err', text: list.join(' ') }),
      ])
    );
    const reread = target
      ? rereadButton(
          () => void actions.read('variables', environment, target, Object.keys(env.variables))
        )
      : null;
    replaceChildren(
      scroll,
      ...draftStage({
        config: previewConfig(m.form, { environment, target }),
        env,
        models: state.models.list,
        values: reading?.values || {},
        objectiveNow: observed?.values?.[m.form.objective],
        flags,
        tools: reread,
      }),
      reading?.error
        ? h('p', { class: 'opt-note', text: `The values now could not be read: ${reading.error}` })
        : null
    );
  }

  /** Ask for the values now of the knobs and the objective on the target. */
  function readNow() {
    const env = details();
    const { environment, target } = where();
    if (!env || !target) return;
    const vKey = `${environment}|${target}`;
    if (vKey !== variablesRead) {
      variablesRead = vKey;
      void actions.read('variables', environment, target, Object.keys(env.variables));
    }
    const objective = m.form?.objective || '';
    const oKey = `${vKey}|${objective}`;
    if (objective && oKey !== objectiveRead) {
      objectiveRead = oKey;
      void actions.read('observables', environment, target, [objective]);
    }
  }

  return {
    update(state) {
      const setup = state.setup;
      if (!alive || !setup) return;
      latest = state;
      if (!restored) {
        // The edits of this set-up, if another screen replaced the view meanwhile.
        restored = true;
        if (kept && setup.id && kept.id === setup.id) m = kept.memo;
        kept = null;
      }
      // Read once here; one that failed is read again by its Read again control.
      if (setup.environment && !state.envs[setup.environment])
        void actions.loadEnvironment(setup.environment);
      const desc = setup.jobId ? 'stored draft' : setup.from ? 'from a past run' : 'new run';
      if (desc !== frameKey) {
        frameKey = desc;
        shell.fillInspector({
          title: 'Set up the run',
          desc,
          body,
          foot: [pending, discardBtn, saveBtn],
        });
      }
      seed();
      readNow();
      const entry = state.envs[setup.environment];
      const nextInspector = JSON.stringify([
        setup.id,
        setup.environment,
        setup.target,
        setup.issues,
        setup.saving,
        !!entry?.details,
        entry?.error,
        state.environments.list,
        state.environments.error,
        state.models.list?.length,
        state.models.error,
        held(),
      ]);
      if (nextInspector !== inspectorKey) {
        inspectorKey = nextInspector;
        rebuild();
        stageKey = '';
      }
      const nextStage = JSON.stringify([
        state.readings[readingKey('variables', setup.environment, setup.target)]?.at,
        state.readings[readingKey('observables', setup.environment, setup.target)]?.at,
      ]);
      if (nextStage !== stageKey) {
        stageKey = nextStage;
        refresh();
      }
    },
    unmount() {
      alive = false;
      const id = latest?.setup?.id;
      kept = id && m.form ? { id, memo: m } : null;
    },
  };
}
