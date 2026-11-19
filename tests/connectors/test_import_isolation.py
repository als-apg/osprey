"""Import isolation for the lean control-system connector chain.

External consumers (a site's tuning-scripts backend, say) import only the
control-system connectors and their support modules. That chain must not
eagerly load the archiver stack (pandas) or any LLM/agent machinery.

The control-context record and the acting-identity ladder are held to a
stricter rule still: they are read inside connector-host children, executor
sandboxes and notebook kernels, so they may not reach ``osprey`` at all.

The connectors distribution as a whole runs with no ``osprey`` on the path.
The mock archiver is the one connector that reads project config beyond its
own block, so it is proven here by an actual ``connect``, not by an import.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.facility.served_tree import served_tree

SRC = str(Path(__file__).resolve().parents[2] / "src")

# Sentinels for the two dependency trees that must stay out of the lean chain:
# the archiver's dataframe stack and the LLM/agent platform.
FORBIDDEN = ("pandas", "litellm", "openai", "anthropic", "fastapi", "playwright")


def _run_clean(code: str) -> None:
    """Run ``code`` in a fresh interpreter; it must exit 0 and print CLEAN."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=SRC),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_control_system_chain_imports_without_heavy_deps():
    code = (
        "import osprey_connectors.control_system._epics_connector;"
        "import osprey_connectors.control_system._limits_validator;"
        "import osprey_connectors.errors, osprey_connectors.config, osprey_connectors.logger;"
        "import osprey.simulation, sys;"
        f"bad = sorted({{m.split('.')[0] for m in sys.modules}} & set({FORBIDDEN!r}));"
        "assert not bad, f'lean connector chain eagerly imported: {bad}';"
        "print('CLEAN')"
    )
    _run_clean(code)


def test_limits_validator_reaches_for_no_control_system_client():
    """The shared validator holds no client of its own.

    It used to ``import epics`` inside the ``max_step`` check, which measured
    the step over Channel Access whatever control system the write was bound
    for. The read now comes from the connector doing the write, so nothing in
    this module may reach for a client library. Importing the module is not
    enough to prove that -- the old import was inside a function -- so the
    check is run against a validator that actually performs a step check.
    """
    code = (
        "import sys;"
        "from osprey_connectors.control_system._limits_validator import ("
        "    ChannelLimitsConfig, LimitsValidator);"
        "v = LimitsValidator("
        "    {'FOO': ChannelLimitsConfig(address='FOO', max_step=5.0, writable=True)},"
        "    {'mode': 'exclusive'}, {});"
        "v.validate('FOO', 1.0, read_current=lambda _a: 0.0);"
        "bad = sorted({'epics', 'p4p', 'tango', 'caproto', 'doocs4py'} & set(sys.modules));"
        "assert not bad, f'the limits validator imported a control-system client: {bad}';"
        "print('CLEAN')"
    )
    _run_clean(code)


def test_control_context_imports_no_osprey_module():
    """The record reader must not pull in the framework it is read beneath.

    ``osprey`` is importable in this subprocess — ``src`` is on the path — so
    an accidental import would succeed rather than fail loudly. The assertion
    is therefore on what landed in ``sys.modules``, not on whether the import
    worked. ``osprey_connectors`` is a different distribution and does not
    match: only ``osprey`` itself and its submodules do.
    """
    code = (
        "import osprey_connectors.control_context as cc, sys;"
        "bad = sorted(m for m in sys.modules if m == 'osprey' or m.startswith('osprey.'));"
        "assert not bad, f'the control-context record eagerly imported: {bad}';"
        "assert cc.RECORD_FILENAME == 'control_context.json';"
        "print('CLEAN')"
    )
    _run_clean(code)


def test_identity_imports_no_osprey_module():
    """The ladder must not pull in the framework it resolves an identity for.

    Same shape and same reason as the control-context case: a sandbox child or
    a notebook kernel resolves its own audit and control-state directory
    through this ladder in an interpreter where ``osprey`` is absent. ``src``
    is on the path here, so an accidental import would succeed quietly rather
    than fail -- the assertion is on what landed in ``sys.modules``. The call
    is made, not just the import, because a rung added inside the function
    would escape an import-time check.
    """
    code = (
        "import osprey_connectors.identity as identity, sys;"
        "assert identity.acting_identity(), 'the ladder resolved nothing';"
        "bad = sorted(m for m in sys.modules if m == 'osprey' or m.startswith('osprey.'));"
        "assert not bad, f'the identity ladder eagerly imported: {bad}';"
        "print('CLEAN')"
    )
    _run_clean(code)


@pytest.mark.parametrize(
    ("driver", "type_name", "registry", "connector_module"),
    [
        ("tango", "tango", "_control_system_connectors", "control_system._tango_connector"),
        ("doocs4py", "doocs", "_control_system_connectors", "control_system._doocs_connector"),
        (
            "doocs4py",
            "doocs_archiver",
            "_service_backends['archive']",
            "archiver._doocs_archiver_connector",
        ),
        (
            "pymongo",
            "mongodb_archiver",
            "_service_backends['archive']",
            "archiver._mongodb_archiver_connector",
        ),
    ],
    ids=["tango", "doocs", "doocs_archiver", "mongodb_archiver"],
)
def test_builtin_registration_needs_no_driver(driver, type_name, registry, connector_module):
    """A built-in registers on a machine where its driver cannot be imported.

    Each connector imports its driver inside ``connect()``, and each archive when
    it is built, never at module scope. If that import ever
    moved to module scope, registering the built-ins would raise ImportError on
    every machine without that control system and take the framework down with
    it. This has to run in a fresh interpreter: in the test process the
    connector module is already cached, so its module body -- where a
    module-scope driver import would live -- never runs again.
    """
    module = f"osprey_connectors.{connector_module}"
    code = (
        "import sys;"
        f"sys.modules[{driver!r}] = None;"
        f"assert {module!r} not in sys.modules;"
        "from osprey_connectors.factory import ConnectorFactory;"
        "ConnectorFactory._ensure_builtins();"
        f"assert {module!r} in sys.modules, 'the connector module was never imported';"
        f"registered = ConnectorFactory.{registry};"
        f"assert {type_name!r} in registered, sorted(registered);"
        "print('CLEAN')"
    )
    _run_clean(code)


def test_mock_archiver_serves_a_built_view_without_osprey(tmp_path):
    """The mock archiver serves a built simulator view with ``osprey`` unimportable.

    A finder at the front of ``sys.meta_path`` refuses ``osprey`` and its
    submodules, because the dev environment has the framework installed and
    no path setting can hide it. The child first proves the finder is live,
    then builds the archive over the view and reads a channel's history
    back.
    """
    view = served_tree(tmp_path / "served", ["T:Q1:CUR:SP"])
    code = textwrap.dedent(
        """
        import asyncio
        import math
        import sys
        from datetime import UTC, datetime


        class _RefuseOsprey:
            def find_spec(self, name, path=None, target=None):
                if name == "osprey" or name.startswith("osprey."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None


        sys.meta_path.insert(0, _RefuseOsprey())
        try:
            import osprey  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            sys.exit("blocker inert")

        from osprey_connectors.archiver._mock_archiver_connector import MockArchiverConnector
        from osprey_connectors.service_answers import BackendSpec


        async def main():
            connector = MockArchiverConnector(
                BackendSpec(
                    kind="archive",
                    type="mock_archiver",
                    target="va",
                    settings={"simulator_view": sys.argv[1]},
                    rehearsal=False,
                )
            )
            df = await connector.history(
                ["T:Q1:CUR:SP"],
                start_at=datetime(2024, 1, 1, tzinfo=UTC),
                end_at=datetime(2024, 1, 1, 1, tzinfo=UTC),
            )
            values = df.loc[df["address"] == "T:Q1:CUR:SP", "value"].tolist()
            assert values and all(math.isfinite(v) for v in values), values
            await connector.aclose()


        asyncio.run(main())
        bad = sorted(m for m in sys.modules if m == "osprey" or m.startswith("osprey."))
        assert not bad, f"the mock archiver imported: {bad}"
        print("CLEAN")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(view)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_write_door_imports_no_osprey_module():
    """The door is read by the raw-client guard beneath the framework.

    The guard runs in executor sandboxes and notebook kernels, so the module it
    asks must be stdlib-only. Same shape as the ladder case: ``src`` is on the
    path, so the assertion is on ``sys.modules``, and the door is opened and
    read rather than only imported.
    """
    code = (
        "from osprey_connectors.control_system import write_door;"
        "import sys;"
        "assert not write_door.door_is_open();"
        "cm = write_door.open_door(); cm.__enter__();"
        "assert write_door.door_is_open();"
        "cm.__exit__(None, None, None);"
        "bad = sorted(m for m in sys.modules if m == 'osprey' or m.startswith('osprey.'));"
        "assert not bad, f'the write door eagerly imported: {bad}';"
        "print('CLEAN')"
    )
    _run_clean(code)
