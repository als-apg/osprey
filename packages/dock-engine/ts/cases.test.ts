// C-CASES: every case file is well formed and names a rule, the set covers
// what the plan requires, and the C-LAYOUT cases replay against layout.ts.
import { describe, expect, it } from 'vitest';
import {
  CASE_KINDS,
  checkCase,
  diffApprox,
  layoutEngine,
  loadCases,
  runCase,
  runOpCase,
} from './cases.js';
import type { Engine, OpCase, OpResult } from './cases.js';
import { LAYOUT_REASONS } from './layout.js';

describe.each(CASE_KINDS.map((kind) => [kind]))('%s cases', (kind) => {
  const loaded = loadCases(kind);

  it('has cases', () => {
    expect(loaded.length).toBeGreaterThan(0);
  });

  it.each(loaded.map((c) => [c.file, c]))('%s is well formed and names a rule', (_file, c) => {
    expect(checkCase(c)).toEqual([]);
  });
});

describe('coverage', () => {
  const names = new Set(CASE_KINDS.flatMap((kind) => loadCases(kind).map((c) => c.data.name)));
  const required = [
    // the four tasks
    't1-new-outer-bottom',
    't2-new-centre',
    't3-tile-onto-border',
    't4-step-1-new-edge-bottom',
    't4-step-2-tile-edge-bottom',
    // rules 1-7
    'rule1-new-tab-index-0',
    'rule1-sole-tab-empties-first-tile',
    'rule2-new-edge-right',
    'rule3-new-border-row',
    'rule3-border-between-branches',
    'rule4-tile-outer-left',
    'rule5-move-out-collapses-column',
    'rule6-tile-onto-centre-keeps-showing',
    'open-new-into-focused-tile',
    // self-drops
    'self-tile-own-centre',
    'self-tile-own-edge',
    'self-tile-own-right-border',
    'self-tile-own-left-border',
    'self-sole-tab-own-centre',
    'self-sole-tab-own-edge',
    'self-sole-tab-own-border',
    'self-sole-tab-own-tab-bar',
    // 40 px clamp
    'min-size-edge-on-70px-tile',
    'min-size-infeasible-edge',
    // close and the leave rule
    'close-tile-to-split-from',
    'close-tile-to-previous',
    'close-tile-to-next',
    'close-tile-from-split-again',
    'close-others-within-tile',
    'close-right-within-tile',
    'reopen-tabs-group',
    'open-set-layout-with-skipped-member',
    'reopen-layout-group',
    'replace-showing-key',
    'open-set-side-one-row',
    'open-set-side-grid',
    'open-set-side-refused-for-agent',
    'open-pick-target-edge-beside-terminal',
    'open-hidden-tab-leaves-terminal-tile',
    'no-depth-cap-deep-alternation',
    // hit
    'hit-bar-top-pixel-row',
    'hit-bar-bottom-pixel-row',
    'hit-strip-top-under-header',
    'hit-past-top-strip-is-tile-top-edge',
    'hit-header-over-top-strip-is-tab',
    'hit-near-split-line-is-left-tile-right-edge',
    'hit-strip-left',
    'hit-past-left-strip-is-tile-left-edge',
    'hit-strip-right',
    'hit-strip-bottom',
    'hit-one-tile-top-strip',
    'hit-edge-left',
    'hit-edge-right',
    'hit-edge-top',
    'hit-edge-bottom',
    'hit-centre',
    'hit-one-tile-strip',
    'hit-narrow-tile-right-side-is-centre',
    'hit-self-tile-own-centre',
    'hit-band-left-is-31-percent',
    'hit-band-30-percent-of-800-is-edge',
    'hit-band-32-percent-of-800-is-centre',
    'hit-band-top-is-31-percent',
    'hit-band-corner-is-left',
    'hit-no-shared-edge-neighbour-band-splits',
    'hit-header-toolbar-appends-tab',
    'hit-neighbour-tile-highlights-hovered-half',
  ];

  it.each(required.map((name) => [name]))('includes %s', (name) => {
    expect(names.has(name)).toBe(true);
  });

  it('names every validate reason at least once', () => {
    const reasons = new Set(
      loadCases('validate').flatMap((c) => ('reason' in c.data ? [c.data.reason] : []))
    );
    expect([...reasons].sort()).toEqual(Object.values(LAYOUT_REASONS).sort());
  });

  it('has validate and normalize cases for both forms', () => {
    const forms = (cases: { options?: { form?: string }; form?: string }[]): Set<string> =>
      new Set(cases.map((c) => c.options?.form ?? c.form ?? 'dock'));
    expect([...forms(loadCases('validate').map((c) => c.data))].sort()).toEqual([
      'collection',
      'dock',
    ]);
    expect([...forms(loadCases('normalize').map((c) => c.data))].sort()).toEqual([
      'collection',
      'dock',
    ]);
  });

  it('names every case once across kinds', () => {
    const all = CASE_KINDS.flatMap((kind) => loadCases(kind).map((c) => c.data.name));
    expect(all.length).toBe(new Set(all).size);
  });
});

describe('C-LAYOUT cases replay against layout.ts', () => {
  for (const kind of ['validate', 'normalize'] as const) {
    it.each(loadCases(kind).map((c) => [c.data.name, c.data]))('%s', (_name, c) => {
      expect(runCase(kind, c, layoutEngine)).toEqual([]);
    });
  }
});

describe('the runner', () => {
  const [drop] = loadCases('drop').filter((c) => c.data.name === 't2-new-centre');
  const t2 = drop?.data as OpCase;

  it('fails a case whose function the engine lacks', () => {
    expect(runOpCase(t2, {})).toEqual(['engine has no planDrop']);
  });

  it('passes an engine that returns the expected layout, within tolerance', () => {
    const after = structuredClone(t2.after) as OpResult['layout'];
    const engine: Engine = { planDrop: () => ({ layout: after, changed: true }) };
    expect(runOpCase(t2, engine)).toEqual([]);
  });

  it('fails an engine that changes nothing on a changing case', () => {
    const engine: Engine = { planDrop: (layout) => ({ layout, changed: false }) };
    expect(runOpCase(t2, engine).length).toBeGreaterThan(0);
  });

  it('accepts sizes within 0.005 and refuses beyond', () => {
    expect(diffApprox({ size: 0.334 }, { size: 0.3333 }, 0.005)).toEqual([]);
    expect(diffApprox({ size: 0.34 }, { size: 0.3333 }, 0.005)).toHaveLength(1);
    expect(diffApprox({ x: 1 }, { x: 1, y: 2 }, 0)).toEqual(['$.y: missing']);
  });
});

describe('checkCase refuses a broken case', () => {
  const [good] = loadCases('drop').filter((c) => c.data.name === 't3-tile-onto-border');
  if (good === undefined) throw new Error('anchor case missing');
  const broken = (edit: (data: Record<string, unknown>) => void, bytes?: string) => {
    const data = structuredClone(good.data) as unknown as Record<string, unknown>;
    edit(data);
    return checkCase({
      ...good,
      data: data as unknown as OpCase,
      bytes: bytes ?? `${JSON.stringify(data, null, 2)}\n`,
    });
  };

  it('a case without a known rule', () => {
    expect(broken((d) => (d.rule = 'rule-99'))).toEqual(['$.rule: rule-99 names no rule']);
  });

  it('a case whose bytes are not canonical', () => {
    expect(broken(() => undefined, JSON.stringify(good.data))).toEqual([
      'bytes: not the committed JSON form',
    ]);
  });

  it('an after whose sizes do not sum to 1', () => {
    const problems = broken((d) => {
      const after = d.after as { items: { size: number }[] };
      if (after.items[0]) after.items[0].size = 0.5;
    });
    expect(problems.some((p) => p.startsWith('$.after: bad_size'))).toBe(true);
  });

  it('a target naming a tile that is not there', () => {
    const problems = broken((d) => {
      (d.op as { target: unknown }).target = { tile: 't9', zone: 'centre' };
    });
    expect(problems).toContain('$.op.target: names a tile not in before');
  });
});

describe('checkCase refuses a refusing validate case without a where', () => {
  const [refusing] = loadCases('validate').filter(
    (c) => c.data.name === 'validate-dock-old-slot-shape'
  );
  if (refusing === undefined) throw new Error('anchor case missing');

  it('the anchor itself is well formed', () => {
    expect(checkCase(refusing)).toEqual([]);
  });

  it('drops the where', () => {
    const data = structuredClone(refusing.data) as unknown as Record<string, unknown>;
    delete data.where;
    const problems = checkCase({
      ...refusing,
      data: data as unknown as typeof refusing.data,
      bytes: `${JSON.stringify(data, null, 2)}\n`,
    });
    expect(problems).toEqual(['$: reason without a where']);
  });
});
