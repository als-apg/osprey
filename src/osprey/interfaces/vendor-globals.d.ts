// Ambient declarations for vendored classic-script globals loaded via <script>
// tags (Plotly, marked, highlight.js, KaTeX). These libraries are not ES
// modules and attach themselves to the global scope, so `// @ts-check`
// modules that reference them need a type to resolve against. `any` is a
// deliberate starting point — tighten to real vendor types incrementally.
declare const Plotly: any;
declare const marked: any;
declare const DOMPurify: any;
declare const hljs: any;
declare const katex: any;
declare const Terminal: any;
declare const FitAddon: any;
declare const WebLinksAddon: any;

// The per-user URL prefix injected into every served HTML document by the
// web-terminal app (`window.__OSPREY_PREFIX__ = "/u/<user>"` or "" for the
// single-origin/dev case). Read by the prefix-aware fetch/ws helpers. Optional
// because it is absent until the injecting <script> runs.
interface Window {
  __OSPREY_PREFIX__?: string;
}
