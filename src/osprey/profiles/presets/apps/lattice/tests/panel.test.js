import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  charts,
  metaLine,
  panelOf,
  query,
  refresh,
  showResult,
  showState,
} from '../panel/panel.js';

// The panel's markup, as index.html carries it.
const MARKUP = `
  <select data-field="model" hidden></select>
  <select data-field="energy" hidden></select>
  <span data-field="analyses"></span>
  <span data-field="status" hidden></span>
  <p data-field="error" hidden></p>
  <p data-field="meta" hidden></p>
  <div data-field="body"></div>`;

/** @param {string} name @param {number[]} [energies] */
function model(name, energies = []) {
  return {
    name,
    label: name.toUpperCase(),
    engine: 'pyat',
    engine_settings: { bend_pass: 'ExactSectorBendPass' },
    deck: `${name}.lte`,
    energies_gev: energies,
    analyses: ['optics', 'da'],
  };
}

/** @param {'ready' | 'queued' | 'running' | 'failed' | 'absent'} status @param {object} [over] */
function entry(status, over = {}) {
  return { status, key: 'k1', settings: { nturns: 128 }, deadline_s: 300, ...over };
}

/** @param {object} [over] */
function state(over = {}) {
  return {
    error: null,
    models: [model('sr'), model('booster', [0.425, 6])],
    analyses: [
      { name: 'optics', label: 'Optics' },
      { name: 'chromaticity', label: 'Chromaticity' },
      { name: 'da', label: 'Dynamic aperture' },
    ],
    selected: { model: 'sr', energy_gev: null },
    results: {
      optics: entry('ready', { scalars: { tune_x: 14.13 } }),
      da: entry('absent', { key: 'k2' }),
    },
    ...over,
  };
}

const OPTICS = {
  ...entry('ready', {
    scalars: { tune_x: 14.13, tune_y: 6.22, circumference_m: 234.1 },
    units: { circumference_m: 'm' },
  }),
  model: 'sr',
  analysis: 'optics',
  tables: {
    twiss: { columns: { s: [0, 1, 2], beta_x: [1, 2, 1], beta_y: [2, 1, 2], eta_x: [0, 0.1, 0] } },
  },
};

/** @param {unknown} body @param {number} [status] */
function answer(body, status = 200) {
  return { ok: status < 400, status, statusText: 'x', json: async () => body };
}

/** @type {ReturnType<typeof panelOf>} */
let panel;

beforeEach(() => {
  localStorage.clear();
  document.body.innerHTML = MARKUP;
  panel = panelOf(document.body);
});

describe('showState', () => {
  it('offers the models and the analyses the selected one runs', () => {
    showState(panel, state());
    expect(panel.model.hidden).toBe(false);
    expect([...panel.model.options].map((o) => o.textContent)).toEqual(['SR', 'BOOSTER']);
    const buttons = [...panel.analyses.querySelectorAll('button')];
    expect(buttons.map((b) => b.dataset.analysis)).toEqual(['optics', 'da']);
    expect(buttons.map((b) => b.getAttribute('aria-pressed'))).toEqual(['true', 'false']);
    expect(buttons[1].dataset.status).toBe('absent');
  });

  it('hides the energy picker for a model with one energy and shows it for a ramped one', () => {
    showState(panel, state());
    expect(panel.energy.hidden).toBe(true);
    showState(panel, state({ selected: { model: 'booster', energy_gev: 6 } }));
    expect(panel.energy.hidden).toBe(false);
    expect(panel.energy.value).toBe('6');
    expect(panel.choice).toMatchObject({ model: 'booster', energy: 6 });
  });

  it('says what stands behind the numbers', () => {
    showState(panel, state());
    expect(panel.status.textContent).toBe('ready');
    expect(panel.status.className).toContain('osprey-chip--ok');
    expect(panel.meta.textContent).toBe(
      'engine pyat · deck sr.lte · bend_pass ExactSectorBendPass · nturns 128'
    );
  });

  it('shows the backend error and nothing else when no model is offered', () => {
    showState(
      panel,
      state({ error: 'no lattice model is configured', models: [], selected: null, results: {} })
    );
    expect(panel.error.hidden).toBe(false);
    expect(panel.error.textContent).toBe('no lattice model is configured');
    expect(panel.analyses.children).toHaveLength(0);
  });

  it('leaves an unchanged select alone, so a poll does not close it', () => {
    showState(panel, state());
    const option = panel.model.options[0];
    showState(panel, state());
    expect(panel.model.options[0]).toBe(option);
  });
});

describe('showResult', () => {
  it('draws the charts, the scalars and both file links of a ready result', () => {
    panel.choice = { model: 'sr', energy: null, analysis: 'optics' };
    showResult(panel, OPTICS);
    expect(panel.body.querySelectorAll('svg')).toHaveLength(3);
    expect(panel.body.textContent).toContain('circumference_m234.1 m');
    const links = [...panel.body.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(links).toEqual([
      'api/export?model=sr&analysis=optics&format=csv',
      'api/export?model=sr&analysis=optics&format=sdds',
    ]);
  });

  it('shows a failure with its reason and a retry', () => {
    showResult(panel, {
      ...entry('failed', { error: 'stopped at its deadline of 300 s' }),
      model: 'sr',
      analysis: 'da',
    });
    expect(panel.body.textContent).toContain('stopped at its deadline of 300 s');
    expect(panel.body.querySelector('[data-action="retry"]')).not.toBeNull();
  });

  it('says a running analysis is shared and bounded', () => {
    showResult(panel, { ...entry('running', { elapsed_s: 41.6 }), model: 'sr', analysis: 'da' });
    expect(panel.body.textContent).toContain('da: computing for 42 s');
    expect(panel.body.textContent).toContain('deadline 300 s');
  });
});

describe('charts', () => {
  it('draws a footprint as a tune diagram and an amplitude map', () => {
    const result = {
      ...entry('ready'),
      model: 'sr',
      analysis: 'footprint',
      tables: {
        tunes: {
          columns: {
            x_mm: [1, 2],
            y_mm: [1, 1],
            tune_x: [0.13, 0.131],
            tune_y: [0.22, 0.221],
            diffusion: [-8, -3],
          },
        },
      },
    };
    const drawn = charts('footprint', result);
    expect(drawn).toHaveLength(2);
    expect(drawn[0]).toContain('class="lt-res"');
  });

  it('draws nothing for a result without the table', () => {
    expect(charts('lma', { ...entry('ready'), model: 'sr', analysis: 'lma', tables: {} })).toEqual(
      []
    );
  });
});

describe('query and metaLine', () => {
  it('names the choice', () => {
    expect(query({ model: 'booster', energy: 6, analysis: 'da' })).toBe(
      'model=booster&energy_gev=6&analysis=da'
    );
    expect(query({ model: null, energy: null, analysis: null })).toBe('');
  });

  it('adds when a ready result was computed', () => {
    const line = metaLine(
      model('sr'),
      entry('ready', { computed_at: '2026-11-10T10:00:00Z', elapsed_s: 12.5 })
    );
    expect(line).toContain('computed 2026-11-10T10:00:00Z in 12.5 s');
  });
});

describe('refresh', () => {
  it('reads the state, then the ready result once', async () => {
    const fetchMock = vi.fn(async (/** @type {string} */ url) =>
      answer(url.startsWith('api/state') ? state() : OPTICS)
    );
    vi.stubGlobal('fetch', fetchMock);
    await refresh(panel);
    await refresh(panel);
    const urls = fetchMock.mock.calls.map(([url]) => url);
    expect(urls).toEqual([
      'api/state?',
      'api/result?model=sr&analysis=optics',
      'api/state?model=sr',
    ]);
    expect(panel.body.querySelectorAll('svg')).toHaveLength(3);
  });

  it('asks for an absent analysis once and then follows it through the state', async () => {
    panel.choice.analysis = 'da';
    const running = state({ results: { da: entry('running', { key: 'k2', elapsed_s: 3 }) } });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(answer(state()))
      .mockResolvedValueOnce(
        answer({ ...entry('queued', { key: 'k2' }), model: 'sr', analysis: 'da' })
      )
      .mockResolvedValueOnce(answer(running));
    vi.stubGlobal('fetch', fetchMock);
    await refresh(panel);
    expect(panel.body.textContent).toContain('queued');
    await refresh(panel);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(panel.body.textContent).toContain('computing for 3 s');
  });

  it('retries a failed analysis through the result route', async () => {
    panel.choice.analysis = 'da';
    const failed = state({ results: { da: entry('failed', { key: 'k2', error: 'boom' }) } });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(answer(failed))
      .mockResolvedValueOnce(
        answer({ ...entry('queued', { key: 'k2' }), model: 'sr', analysis: 'da' })
      );
    vi.stubGlobal('fetch', fetchMock);
    await refresh(panel, { retry: true });
    expect(fetchMock.mock.calls[1][0]).toBe('api/result?model=sr&analysis=da&retry=true');
  });

  it('drops a stored choice the deployment no longer offers', async () => {
    panel.choice = { model: 'gone', energy: 3, analysis: 'optics' };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => answer({ detail: 'there is no model gone' }, 404))
    );
    await refresh(panel);
    expect(panel.choice).toEqual({ model: null, energy: null, analysis: 'optics' });
  });

  it('says so when the backend does not answer', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => Promise.reject(new Error('offline')))
    );
    await refresh(panel);
    expect(panel.error.textContent).toBe('The lattice backend is not answering: offline');
  });
});
