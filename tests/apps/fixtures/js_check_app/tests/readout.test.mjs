import { describe, expect, it } from 'vitest';
import { readout } from '../panel/readout.js';

describe('readout', () => {
  it('shows the value and remembers it', () => {
    expect(readout('42').textContent).toBe('42');
    expect(localStorage.getItem('readout')).toBe('42');
  });
});
