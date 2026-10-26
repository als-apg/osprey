// @vitest-environment node
// C-ENGINE: the committed dist/ files replay the whole C-CASES set, export
// exactly the CUTOVER.json names, stand alone (no imports, no absolute paths,
// declarations that type-check by themselves) and match dist/manifest.json.
// Freshness against the sources is not checked here: that is
// `node packages/dock-engine/scripts/build.mjs --check` (D21).
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { CASE_KINDS, CASES_DIR, loadCases, runCase } from '../dist/cases.mjs';
import * as engine from '../dist/dock-engine.mjs';

const path = (rel: string): string => fileURLToPath(new URL(rel, import.meta.url));

const PKG = path('../');
const REPO = path('../../../');
const DIST = path('../dist/');
const CASES = path('../cases/');
const TSC = path('../../../node_modules/typescript/bin/tsc');

interface Cutover {
  exports: string[];
  case_counts: Record<string, number>;
}

interface Manifest {
  files: Record<string, string>;
  cases: Record<string, string>;
  rules_digest: string;
  exports: string[];
}

const readJson = <T>(file: string): T => JSON.parse(readFileSync(file, 'utf8')) as T;
const cutover = readJson<Cutover>(path('../CUTOVER.json'));
const manifest = readJson<Manifest>(join(DIST, 'manifest.json'));
const sha256 = (bytes: Buffer): string => createHash('sha256').update(bytes).digest('hex');

/** Every path a bundle must never name, with and without the trailing separator. */
const MACHINE_PATHS = [PKG, REPO].flatMap((dir) => [dir, dir.replace(/[\\/]$/, '')]);

/** The import specifiers of an ES module's static import statements. */
function importSpecifiers(code: string): string[] {
  return [...code.matchAll(/^\s*import\s(?:[^'";]*?\sfrom\s*)?['"]([^'"]+)['"]/gm)].map(
    (m) => m[1] as string
  );
}

describe.each(CASE_KINDS.map((kind) => [kind]))(
  '%s cases replay against dist/dock-engine.mjs',
  (kind) => {
    const loaded = loadCases(kind);

    it('has cases', () => {
      expect(loaded.length).toBeGreaterThan(0);
    });

    it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
      expect(runCase(kind, c.data, engine)).toEqual([]);
    });
  }
);

describe('dist/dock-engine.mjs', () => {
  const js = readFileSync(join(DIST, 'dock-engine.mjs'), 'utf8');

  it('exports exactly the CUTOVER.json names', () => {
    expect(Object.keys(engine).sort()).toEqual([...cutover.exports].sort());
  });

  it('imports nothing and names no machine path', () => {
    expect(js).not.toMatch(/^\s*import\s/m);
    expect(js).not.toMatch(/\brequire\(/);
    for (const dir of MACHINE_PATHS) expect(js).not.toContain(dir);
  });

  it('says it is generated', () => {
    const lines = js.split('\n');
    expect(lines[0]).toMatch(/^\/\/ @als-apg\/dock-engine: /);
    expect(lines[1]).toBe(
      '// Generated from packages/dock-engine/ts/ by packages/dock-engine/scripts/build.mjs.'
    );
  });
});

describe('dist/dock-engine.d.mts', () => {
  const dts = readFileSync(join(DIST, 'dock-engine.d.mts'), 'utf8');

  it('declares exactly the runtime exports', () => {
    const declared = [...dts.matchAll(/^export declare (?:function|const) (\w+)/gm)].map(
      (m) => m[1]
    );
    expect([...new Set(declared)].sort()).toEqual(Object.keys(engine).sort());
  });

  it('stands alone: no imports, and the declarations type-check by themselves', () => {
    expect(dts).not.toMatch(/^\s*import\s/m);
    // Throws, with tsc's diagnostics on stdout, when the check fails.
    execFileSync(process.execPath, [TSC, '--noEmit', '-p', join(PKG, 'tsconfig.dts.json')], {
      cwd: REPO,
      encoding: 'utf8',
      stdio: 'pipe',
    });
  });
});

describe('dist/cases.mjs', () => {
  const js = readFileSync(join(DIST, 'cases.mjs'), 'utf8');

  it('imports exactly the four allowed specifiers', () => {
    expect([...new Set(importSpecifiers(js))].sort()).toEqual([
      './dock-engine.mjs',
      'node:fs',
      'node:module',
      'node:path',
    ]);
  });

  it('names no machine path', () => {
    for (const dir of MACHINE_PATHS) expect(js).not.toContain(dir);
  });

  it('resolves CASES_DIR to the package cases directory', () => {
    expect(CASES_DIR).toBe(CASES.replace(/[\\/]$/, ''));
  });

  it('has a case count for every kind and no other', () => {
    expect(Object.keys(cutover.case_counts).sort()).toEqual([...CASE_KINDS].sort());
  });

  describe.each(CASE_KINDS.map((kind) => [kind]))('%s', (kind) => {
    const loaded = loadCases(kind);

    it('loads exactly the CUTOVER.json case count', () => {
      expect(loaded.length).toBeGreaterThan(0);
      expect(loaded.length).toBe(cutover.case_counts[kind]);
    });

    it('replays every case clean through runCase', () => {
      const failures = loaded.flatMap((c) =>
        runCase(kind, c.data, engine).map((f) => `${c.file}: ${f}`)
      );
      expect(failures).toEqual([]);
    });
  });
});

describe('dist/manifest.json', () => {
  it('hashes the dist files as they are on disk', () => {
    const onDisk = Object.fromEntries(
      Object.keys(manifest.files).map((f) => [f, sha256(readFileSync(join(DIST, f)))])
    );
    expect(Object.keys(manifest.files).sort()).toEqual([
      'cases.d.mts',
      'cases.mjs',
      'dock-engine.d.mts',
      'dock-engine.mjs',
    ]);
    expect(onDisk).toEqual(manifest.files);
  });

  it('hashes exactly the case files on disk', () => {
    const onDisk: Record<string, string> = {};
    for (const kind of readdirSync(CASES, { withFileTypes: true })) {
      if (!kind.isDirectory()) continue;
      for (const file of readdirSync(join(CASES, kind.name))) {
        if (!file.endsWith('.json')) continue;
        onDisk[`${kind.name}/${file}`] = sha256(readFileSync(join(CASES, kind.name, file)));
      }
    }
    expect(Object.keys(manifest.cases).sort()).toEqual(Object.keys(onDisk).sort());
    expect(onDisk).toEqual(manifest.cases);
  });

  it('lists the CUTOVER.json exports and a rules digest', () => {
    expect([...manifest.exports].sort()).toEqual([...cutover.exports].sort());
    expect(manifest.rules_digest).toMatch(/^[0-9a-f]{64}$/);
  });
});
