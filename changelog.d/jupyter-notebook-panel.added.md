Web terminal: a **JUPYTER** panel (`jupyter`) serving JupyterLab in a tab,
with kernels that read and write the control system through `osprey.runtime`
under the same write gates as the agent's own Python. A kernel follows the
terminal session most recently attached. It keeps the target it started with
and can never write more than it could at launch, so switching the control
target or turning writes on needs a kernel restart; turning writes off takes
effect on its next write. Notebooks live in `notebooks/` under the agent-data
root, so they survive a redeploy and the OSPREY agent can edit them there. The
file browser downloads and deletes files under `notebooks/` only, and a
deleted notebook is gone rather than trashed. The
panel is selected by default in the `control-assistant` preset family.
JupyterLab, `jupyter-server` and `ipykernel` are now core dependencies, so
images grow by roughly 50 MB whether or not the panel is selected.
