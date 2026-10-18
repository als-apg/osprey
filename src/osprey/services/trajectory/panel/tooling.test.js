import { describe, expect, it } from 'vitest';

// Proves the trajectory panel directory is inside the vitest include and runs
// under the shared happy-dom environment like interface specs do.
describe('trajectory panel test harness', () => {
  it('runs panel specs in a browser-like environment', () => {
    const node = document.createElement('div');
    node.textContent = 'trajectory';
    expect(node.textContent).toBe('trajectory');
  });
});
