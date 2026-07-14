"""The operator-facing notebook header carries the facility-local wall clock.

The attempt-notebook "Created:" header is read by humans, so per the timezone
rule it must render in the facility zone (with a zone label), not the deploy
host's local clock. The strftime *filename* timestamps in the same module stay
naive on purpose (aware ISO's ':'/'+' would corrupt paths); this pins only the
display header. ``_create_attempt_notebook_content`` does not touch instance
state, so it is exercised on a bare instance without the FileManager harness.
"""

from zoneinfo import ZoneInfo

from osprey.services.python_executor.services import NotebookManager

TOKYO = ZoneInfo("Asia/Tokyo")  # %Z -> 'JST', distinct from any test host zone


def test_attempt_header_renders_facility_zone(monkeypatch):
    monkeypatch.setattr(
        "osprey.services.python_executor.services.get_facility_timezone",
        lambda: TOKYO,
    )
    manager = NotebookManager.__new__(NotebookManager)  # bypass FileManager init
    notebook = manager._create_attempt_notebook_content(
        attempt_number=1, stage="generate", code="print(1)"
    )

    header = notebook.cells[0].source
    assert "**Created:**" in header
    # Zone label proves the header is facility-local, not box-local/naive.
    assert "JST" in header
