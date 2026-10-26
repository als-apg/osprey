#!/usr/bin/env node
// C-BUILD: the dock engine (ts/) as committed, generated artifacts in dist/:
//
//   dist/dock-engine.mjs    the seven engine modules as one dependency-free,
//                           unminified ESM file (vite / rolldown)
//   dist/dock-engine.d.mts  their isolated declarations, joined into one file
//   dist/cases.mjs          the cases.ts runner, transpiled, with the engine
//                           imported from ./dock-engine.mjs (D34)
//   dist/cases.d.mts        its declarations, likewise rewritten
//   dist/manifest.json      C-MANIFEST: hashes, rules_digest, exports
//
//   node packages/dock-engine/scripts/build.mjs           write every file
//   node packages/dock-engine/scripts/build.mjs --check   exit 1 when any is stale
//
// Declarations come from the TypeScript 6 JS API (the `typescript-6` alias):
// the repo's TypeScript 7 has no JS API (D19).
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/** The engine modules, in dependency order. cases.ts (node:fs) is never one. */
export const MODULES = ['layout', 'geometry', 'drop', 'open', 'close', 'hitTest', 'resultRect'];

// Resolved from this file, never from the cwd.
export const PKG = fileURLToPath(new URL('../', import.meta.url));
export const TS_DIR = fileURLToPath(new URL('../ts/', import.meta.url));
export const CASES = fileURLToPath(new URL('../cases/', import.meta.url));
export const DIST = fileURLToPath(new URL('../dist/', import.meta.url));

/** Every dist file, in write order; the manifest hashes the ones before it. */
export const FILES = ['dock-engine.mjs', 'dock-engine.d.mts', 'cases.mjs', 'cases.d.mts', 'manifest.json'];

/** The only import specifiers dist/cases.mjs may carry (D34, C-ENGINE). */
const CASES_IMPORTS = ['./dock-engine.mjs', 'node:fs', 'node:module', 'node:path'];

const banner = (what) =>
  [
    `// @als-apg/dock-engine: ${what}.`,
    '// Generated from packages/dock-engine/ts/ by packages/dock-engine/scripts/build.mjs.',
    '// Do not edit; rebuild with `node packages/dock-engine/scripts/build.mjs`',
    '// (`npm run build:dock-engine`).',
    '',
  ].join('\n');

const ENTRY = '\0dock-engine-entry';

function assertNoAbsolutePath(name, code) {
  for (const dir of [PKG, TS_DIR]) {
    if (code.includes(dir) || code.includes(dir.replace(/[\\/]$/, ''))) {
      throw new Error(`dock-engine: ${name} carries an absolute path`);
    }
  }
}

/** The bundle: vite (rolldown) over a virtual entry that re-exports the modules. */
export async function buildJs() {
  const { build } = await import('vite');
  const entryPlugin = {
    name: 'dock-engine-entry',
    enforce: 'pre',
    resolveId(id) {
      // vite hands the lib entry over resolved against the root.
      return id === ENTRY || id.endsWith('dock-engine-entry') ? ENTRY : null;
    },
    load(id) {
      if (id !== ENTRY) return null;
      return MODULES.map((m) => `export * from ${JSON.stringify(join(TS_DIR, `${m}.ts`))};`).join('\n');
    },
  };
  const result = await build({
    configFile: false,
    root: PKG,
    logLevel: 'silent',
    plugins: [entryPlugin],
    build: {
      write: false,
      minify: false,
      sourcemap: false,
      emptyOutDir: false,
      lib: { entry: 'dock-engine-entry', formats: ['es'], fileName: () => 'dock-engine.mjs' },
    },
  });
  const outputs = (Array.isArray(result) ? result : [result]).flatMap((r) => r.output);
  const chunks = outputs.filter((o) => o.type === 'chunk');
  if (chunks.length !== 1) throw new Error(`dock-engine: expected one chunk, got ${chunks.length}`);
  let code = chunks[0].code;
  if (/^\s*import\s/m.test(code) || /\bimport\s*\(/.test(code)) {
    throw new Error('dock-engine: the bundle imports something');
  }
  // rolldown labels regions by cwd-relative path (`packages/dock-engine/ts/layout.ts`
  // from the repo root); C-BUILD labels them by module name, which also keeps the
  // bytes independent of the cwd.
  code = code.replace(/^\/\/#region (.*)$/gm, (line, label) => {
    const m = MODULES.find((name) => resolve(process.cwd(), label) === join(TS_DIR, `${name}.ts`));
    if (!m) throw new Error(`dock-engine: unexpected bundle region ${label}`);
    return `//#region ${m}`;
  });
  assertNoAbsolutePath('the bundle', code);
  return banner('the dock layout engine, one dependency-free ESM file') + code.replace(/\s*$/, '\n');
}

let tsApi;
/** The TypeScript 6 JS API (D19). */
async function typescript() {
  tsApi ??= (await import('typescript-6')).default;
  return tsApi;
}

function declarationOptions(ts) {
  return {
    module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext,
    target: ts.ScriptTarget.ES2023,
    strict: true,
    declaration: true,
    isolatedDeclarations: true,
  };
}

/** One isolated declaration emit, failing on any diagnostic. */
function emitDeclaration(ts, name) {
  const file = join(TS_DIR, `${name}.ts`);
  const out = ts.transpileDeclaration(readFileSync(file, 'utf8'), {
    fileName: `${name}.ts`,
    compilerOptions: declarationOptions(ts),
    reportDiagnostics: true,
  });
  if (out.diagnostics?.length) {
    const text = out.diagnostics.map((d) => ts.flattenDiagnosticMessageText(d.messageText, '\n'));
    throw new Error(`dock-engine: ${name}.ts declarations: ${text.join('; ')}`);
  }
  return out.outputText;
}

/** The name a top-level declaration declares, or null (imports, `export {}`). */
function declName(ts, statement) {
  if (ts.isImportDeclaration(statement) || ts.isExportDeclaration(statement)) return null;
  if (ts.isVariableStatement(statement)) {
    return statement.declarationList.declarations.map((d) => d.name.getText()).join(',');
  }
  if (statement.name) return statement.name.getText();
  throw new Error(`dock-engine: unnamed declaration ${ts.SyntaxKind[statement.kind]}`);
}

/** The engine declarations: one isolated declaration emit per module, joined. */
export async function buildDts() {
  const ts = await typescript();
  const seen = new Map(); // name -> [module, text]
  const parts = [];
  for (const m of MODULES) {
    const source = ts.createSourceFile(`${m}.d.ts`, emitDeclaration(ts, m), ts.ScriptTarget.ES2023, true);
    const kept = [];
    const local = new Set();
    for (const statement of source.statements) {
      if (ts.isImportDeclaration(statement)) {
        const spec = statement.moduleSpecifier.text;
        if (!spec.startsWith('./')) throw new Error(`dock-engine: ${m}.ts imports ${spec}`);
        continue;
      }
      const name = declName(ts, statement);
      if (name === null) continue; // `export {};`
      const text = statement.getFullText(source).replace(/^\s*\n/, '').replace(/\s+$/, '');
      const prior = seen.get(name);
      if (prior && prior[0] !== m) {
        if (prior[1] !== text) throw new Error(`dock-engine: ${name} differs in ${prior[0]}.ts and ${m}.ts`);
        continue; // declared identically upstream: keep the first
      }
      if (!local.has(name)) seen.set(name, [m, text]);
      else seen.set(name, [m, `${seen.get(name)[1]}\n${text}`]);
      local.add(name);
      kept.push(text);
    }
    parts.push(`//#region ${m}\n${kept.join('\n')}\n//#endregion`);
  }
  return `${banner('declarations of dock-engine.mjs')}${parts.join('\n\n')}\n`;
}

/** Rewrite every `from './layout.js'` to the bundle; `exactly` asserts the count. */
function rewriteEngineImports(name, code, exactly) {
  const count = code.split("from './layout.js'").length - 1;
  if (exactly !== undefined && count !== exactly) {
    throw new Error(`dock-engine: ${name} has ${count} engine imports, expected ${exactly}`);
  }
  return code.split("from './layout.js'").join("from './dock-engine.mjs'");
}

/** Every module specifier the code imports or re-exports. */
function importSpecifiers(ts, name, code) {
  const source = ts.createSourceFile(name, code, ts.ScriptTarget.ES2023, true);
  const specs = new Set();
  const visit = (node) => {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier) {
      specs.add(node.moduleSpecifier.text);
    } else if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      throw new Error(`dock-engine: ${name} has a dynamic import`);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return [...specs].sort();
}

/** dist/cases.mjs: cases.ts transpiled, the engine imported from the bundle (D34). */
export async function buildCasesJs() {
  const ts = await typescript();
  const out = ts.transpileModule(readFileSync(join(TS_DIR, 'cases.ts'), 'utf8'), {
    fileName: 'cases.ts',
    reportDiagnostics: true,
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2023,
      verbatimModuleSyntax: true,
    },
  });
  if (out.diagnostics?.length) {
    const text = out.diagnostics.map((d) => ts.flattenDiagnosticMessageText(d.messageText, '\n'));
    throw new Error(`dock-engine: cases.ts transpile: ${text.join('; ')}`);
  }
  const code = rewriteEngineImports('cases.mjs', out.outputText, 1);
  const specs = importSpecifiers(ts, 'cases.mjs', code);
  if (JSON.stringify(specs) !== JSON.stringify(CASES_IMPORTS)) {
    throw new Error(`dock-engine: cases.mjs imports ${specs.join(', ')}, expected ${CASES_IMPORTS.join(', ')}`);
  }
  assertNoAbsolutePath('cases.mjs', code);
  return banner('the golden-case runner (cases.ts) over dock-engine.mjs') + code.replace(/\s*$/, '\n');
}

/** dist/cases.d.mts: cases.ts declarations, type imports pointed at the bundle. */
export async function buildCasesDts() {
  const ts = await typescript();
  const code = rewriteEngineImports('cases.d.mts', emitDeclaration(ts, 'cases'));
  for (const spec of importSpecifiers(ts, 'cases.d.mts', code)) {
    if (spec !== './dock-engine.mjs') throw new Error(`dock-engine: cases.d.mts imports ${spec}`);
  }
  return banner('declarations of cases.mjs') + code.replace(/\s*$/, '\n');
}

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

/** Every case file as `<kind>/<name>.json`, sorted by path string. */
export function casePaths() {
  return readdirSync(CASES, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .flatMap((d) =>
      readdirSync(join(CASES, d.name))
        .filter((f) => f.endsWith('.json'))
        .map((f) => `${d.name}/${f}`),
    )
    .sort();
}

/**
 * C-MANIFEST rules_digest (D25): sha256 of
 *   utf8(JSON.stringify(Object.values(LAYOUT_REASONS))) ‖ "\n"
 *   ‖ Σ over sorted normalize/*.json ∪ validate/*.json: utf8(p) ‖ "\n" ‖ bytes(p)
 *   ‖ bytes(ts/layout.ts)
 */
export function rulesDigest(layoutReasons, paths = casePaths()) {
  const hash = createHash('sha256');
  hash.update(Buffer.from(`${JSON.stringify(Object.values(layoutReasons))}\n`, 'utf8'));
  for (const p of paths.filter((q) => q.startsWith('normalize/') || q.startsWith('validate/')).sort()) {
    hash.update(Buffer.from(`${p}\n`, 'utf8'));
    hash.update(readFileSync(join(CASES, p)));
  }
  hash.update(readFileSync(join(TS_DIR, 'layout.ts')));
  return hash.digest('hex');
}

/** dist/manifest.json over the other four files (C-MANIFEST). */
export async function buildManifest(dist) {
  // The freshly built bundle, not ts/: the manifest describes what ships.
  const url = `data:text/javascript;base64,${Buffer.from(dist['dock-engine.mjs']).toString('base64')}`;
  const engine = await import(url);
  const exportNames = Object.keys(engine).sort();
  const cutover = JSON.parse(readFileSync(join(PKG, 'CUTOVER.json'), 'utf8')).exports;
  if (JSON.stringify([...cutover].sort()) !== JSON.stringify(exportNames)) {
    throw new Error('dock-engine: the bundle exports differ from CUTOVER.json exports[]');
  }
  const paths = casePaths();
  const manifest = {
    files: Object.fromEntries(FILES.filter((f) => f !== 'manifest.json').sort().map((f) => [f, sha256(Buffer.from(dist[f], 'utf8'))])),
    cases: Object.fromEntries(paths.map((p) => [p, sha256(readFileSync(join(CASES, p)))])),
    rules_digest: rulesDigest(engine.LAYOUT_REASONS, paths),
    exports: exportNames,
  };
  return `${JSON.stringify(manifest, null, 2)}\n`;
}

/** Every dist file, by name, as the committed bytes should be. */
export async function buildDockEngine() {
  const dist = {
    'dock-engine.mjs': await buildJs(),
    'dock-engine.d.mts': await buildDts(),
    'cases.mjs': await buildCasesJs(),
    'cases.d.mts': await buildCasesDts(),
  };
  dist['manifest.json'] = await buildManifest(dist);
  return dist;
}

async function main() {
  const dist = await buildDockEngine();
  const rel = (f) => relative(process.cwd(), join(DIST, f)) || join(DIST, f);
  if (process.argv.includes('--check')) {
    const stale = FILES.filter((f) => {
      try {
        return readFileSync(join(DIST, f), 'utf8') !== dist[f];
      } catch {
        return true;
      }
    });
    for (const f of stale) console.error(`dock-engine: stale ${rel(f)} (rebuild with \`npm run build:dock-engine\`)`);
    process.exit(stale.length ? 1 : 0);
  }
  mkdirSync(DIST, { recursive: true });
  for (const f of FILES) writeFileSync(join(DIST, f), dist[f]);
  console.log(FILES.map((f) => `wrote ${rel(f)} (${dist[f].length} bytes)`).join('\n'));
}

// Importing this module (bundle.test.ts does) must not build or write.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  });
}
