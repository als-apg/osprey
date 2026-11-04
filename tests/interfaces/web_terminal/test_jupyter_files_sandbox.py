"""jupyter-server's own sandbox on ``/files`` and ``/nbconvert``, pinned through the proxy.

A user's ``.html`` file and a notebook exported to HTML are documents a cell
(or an agent) wrote, served by the notebook sidecar on the terminal's origin.
Two policies keep each in an opaque origin, and both are pinned here against a
real ``jupyter_server`` reached only through ``/panel/jupyter/*``:

* jupyter-server 2.21's own — ``FilesHandler`` and ``NbconvertFileHandler``
  append ``sandbox allow-scripts`` to their ``Content-Security-Policy`` (the
  latter unless ``nbconvert_csp_sandbox`` is turned off) — relayed untouched;
* the terminal's (``panel_sandbox``), its own header beside it, because neither
  path is one of JupyterLab's entry pages.

If an upgrade drops the upstream directive, the first assertion of each test
says so; the terminal's policy still holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import jupyter_server
import pytest
from fastapi.testclient import TestClient

# The real sidecar and the terminal in front of it, shared with the notebook
# integration suite (module-scoped there, so this module starts its own).
from .test_proxy_jupyter_integration import notebook_env, proxied, sidecar  # noqa: F401

pytestmark = pytest.mark.slow

PANEL = "/panel/jupyter"
TERMINAL_POLICY = "sandbox allow-scripts allow-modals allow-popups allow-downloads"

_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# exported\n", "<script>window.__pwned = 1</script>"],
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _notebooks_dir(notebook_env: Path) -> Path:  # noqa: F811
    root = notebook_env / "shared" / "notebooks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _upstream_policies(response) -> list[str]:
    return [
        policy
        for policy in response.headers.get_list("content-security-policy")
        if policy != TERMINAL_POLICY
    ]


def _assert_both_policies(response) -> None:
    upstream = _upstream_policies(response)
    assert any("sandbox allow-scripts" in policy for policy in upstream), (
        f"jupyter-server {jupyter_server.__version__} no longer sandboxes this route: {upstream}"
    )
    assert not any("allow-same-origin" in policy for policy in upstream)
    assert TERMINAL_POLICY in response.headers.get_list("content-security-policy")
    assert response.headers["x-content-type-options"] == "nosniff"


def test_files_carries_jupyters_sandbox_and_the_terminals(
    proxied: TestClient,  # noqa: F811
    notebook_env: Path,  # noqa: F811
) -> None:
    report = _notebooks_dir(notebook_env) / "sandbox-report.html"
    report.write_text("<html><body><script>1</script></body></html>", encoding="utf-8")

    served = proxied.get(f"{PANEL}/files/sandbox-report.html")

    assert served.status_code == 200
    assert served.headers["content-type"].startswith("text/html")
    _assert_both_policies(served)


def test_nbconvert_carries_jupyters_sandbox_and_the_terminals(
    proxied: TestClient,  # noqa: F811
    notebook_env: Path,  # noqa: F811
) -> None:
    notebook = _notebooks_dir(notebook_env) / "sandbox-export.ipynb"
    notebook.write_text(json.dumps(_NOTEBOOK), encoding="utf-8")

    exported = proxied.get(f"{PANEL}/nbconvert/html/sandbox-export.ipynb?download=false")

    assert exported.status_code == 200, exported.text[:500]
    assert exported.headers["content-type"].startswith("text/html")
    _assert_both_policies(exported)


def test_the_lab_application_is_not_sandboxed(proxied: TestClient) -> None:  # noqa: F811
    lab = proxied.get(f"{PANEL}/lab")

    assert lab.status_code == 200
    assert TERMINAL_POLICY not in lab.headers.get_list("content-security-policy")
