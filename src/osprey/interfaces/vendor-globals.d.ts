// Ambient declarations for vendored classic-script globals loaded via <script>
// tags (Plotly, marked, highlight.js, KaTeX). These libraries are not ES
// modules and attach themselves to the global scope, so `// @ts-check`
// modules that reference them need a type to resolve against. `any` is a
// deliberate starting point — tighten to real vendor types incrementally.
declare const Plotly: any;
declare const marked: any;
declare const hljs: any;
declare const katex: any;
