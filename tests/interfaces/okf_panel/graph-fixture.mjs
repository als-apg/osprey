// @ts-check
/**
 * A small `/api/graph` payload for the Graph view's tests: one machine, a
 * sector inside it, two devices of two classes, their channels and signals, in the
 * shape okf_panel/graph.py serves.
 */

/** @returns {import('../../../src/osprey/interfaces/okf_panel/static/js/graph-model.js').GraphPayload} */
export function payload() {
  return {
    nodes: [
      { id: 'p:M', kind: 'place', label: 'M', type: null, props: {}, record: 'M' },
      { id: 'p:M/S1', kind: 'place', label: 'M/S1', type: null, props: {}, record: 'M/S1' },
      {
        id: 'd:M/BPM1',
        kind: 'device',
        label: 'M/BPM1',
        type: 'BeamPositionMonitor',
        props: { placePath: 'M/S1', description: 'A monitor.', altLabel: 'BPM 1, Orbit monitor 1' },
        record: 'M/BPM1',
        pages: ['devices/bpm'],
      },
      {
        id: 'd:M/Q1',
        kind: 'device',
        label: 'M/Q1',
        type: 'Quadrupole',
        props: {},
        record: 'M/Q1',
      },
      {
        id: 'c:M:BPM1:X',
        kind: 'channel',
        label: 'M:BPM1:X',
        type: null,
        props: { unit: 'mm' },
        record: 'M:BPM1:X',
      },
      {
        id: 'c:M:Q1:SP',
        kind: 'channel',
        label: 'M:Q1:SP',
        type: null,
        props: {},
        record: 'M:Q1:SP',
      },
      { id: 's:position_x', kind: 'signal', label: 'position_x', type: null, props: {} },
      { id: 's:current', kind: 'signal', label: 'current', type: null, props: {} },
      {
        id: 'BeamPositionMonitor',
        kind: 'class',
        label: 'BeamPositionMonitor',
        type: null,
        props: { altLabel: 'BPM' },
      },
      { id: 'Quadrupole', kind: 'class', label: 'Quadrupole', type: null, props: {} },
    ],
    edges: [
      [2, 4, 'hasBinding'],
      [3, 5, 'hasBinding'],
      [4, 6, 'readsSignal'],
      [5, 7, 'writesSignal'],
      [2, 8, 'type'],
      [3, 9, 'type'],
      [2, 1, 'locatedIn'],
      [3, 1, 'locatedIn'],
      [3, 4, 'calibrates'],
      [1, 0, 'partOf'],
    ],
  };
}
