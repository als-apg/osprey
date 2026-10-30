**Breaking change:** JupyterLab in the console no longer runs script in
notebook outputs: `IPython.display.Javascript`, `<script>` in HTML outputs and
bokeh's in-notebook renderer stop working, and
`@jupyterlab/javascript-extension` is disabled. Vega and Vega-Lite outputs,
and so Altair charts, no longer render: `@jupyterlab/vega5-extension` is
disabled. matplotlib, Plotly, ipywidgets, tables, markdown, math and images
still render. The policy keeps `'unsafe-eval'`, as the eval check did not
clear it.
