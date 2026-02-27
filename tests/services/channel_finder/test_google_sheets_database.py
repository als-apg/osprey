"""
Tests for Google Sheets Channel Database.

All tests mock gspread — no network calls or credentials needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from osprey.services.channel_finder.core.exceptions import DatabaseLoadError
from osprey.services.channel_finder.databases.google_sheets import (
    GoogleSheetsChannelDatabase,
)

SAMPLE_RECORDS = [
    {"channel": "CH1", "address": "ADDR1", "description": "Desc1"},
    {"channel": "CH2", "address": "ADDR2", "description": "Desc2"},
    {"channel": "CH3", "address": "ADDR3", "description": "Desc3"},
]


def _make_mock_gspread(records=None):
    """Create a mock gspread module with pre-configured worksheet."""
    if records is None:
        records = SAMPLE_RECORDS

    mock_worksheet = MagicMock()
    mock_worksheet.get_all_records.return_value = records

    mock_spreadsheet = MagicMock()
    mock_spreadsheet.worksheet.return_value = mock_worksheet

    mock_gspread = MagicMock()
    mock_gspread.service_account.return_value.open_by_key.return_value = mock_spreadsheet

    return mock_gspread, mock_spreadsheet, mock_worksheet


class TestGoogleSheetsChannelDatabase:
    """Tests for GoogleSheetsChannelDatabase."""

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_load_database(self, mock_gspread_module):
        """Channels are loaded from mocked sheet data."""
        mock_gspread, _, mock_worksheet = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

        assert len(db.channels) == 3
        assert db.channels[0]["channel"] == "CH1"
        assert db.channels[2]["address"] == "ADDR3"
        assert "CH2" in db.channel_map

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_get_statistics_includes_source(self, mock_gspread_module):
        """Statistics include google_sheets source info."""
        mock_gspread, _, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="abc123", worksheet="MyTab")
        stats = db.get_statistics()

        assert stats["source"] == "google_sheets"
        assert stats["spreadsheet_id"] == "abc123"
        assert stats["worksheet"] == "MyTab"
        assert stats["total_channels"] == 3

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_inherited_methods_work(self, mock_gspread_module):
        """Inherited methods (get_channel, validate_channel, format_for_prompt, chunk_database) work."""
        mock_gspread, _, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

        # get_channel
        ch = db.get_channel("CH1")
        assert ch is not None
        assert ch["address"] == "ADDR1"
        assert db.get_channel("NONEXISTENT") is None

        # validate_channel
        assert db.validate_channel("CH2") is True
        assert db.validate_channel("NOPE") is False

        # format_for_prompt
        prompt = db.format_for_prompt()
        assert "CH1" in prompt
        assert "CH3" in prompt

        # chunk_database
        chunks = db.chunk_database(chunk_size=2)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_add_channel(self, mock_gspread_module):
        """add_channel appends row and reloads cache."""
        mock_gspread, _, mock_worksheet = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

        # After add, the sheet returns updated records
        new_records = SAMPLE_RECORDS + [
            {"channel": "CH4", "address": "ADDR4", "description": "Desc4"}
        ]
        mock_worksheet.get_all_records.return_value = new_records

        db.add_channel("CH4", "ADDR4", "Desc4")

        mock_worksheet.append_row.assert_called_once_with(["CH4", "ADDR4", "Desc4"])
        assert len(db.channels) == 4
        assert "CH4" in db.channel_map

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_add_duplicate_raises(self, mock_gspread_module):
        """Adding a duplicate channel raises ValueError."""
        mock_gspread, _, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

        with pytest.raises(ValueError, match="already exists"):
            db.add_channel("CH1", "ADDR_NEW", "New desc")

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_delete_channel(self, mock_gspread_module):
        """delete_channel finds the row, deletes it, and reloads."""
        mock_gspread, _, mock_worksheet = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

        mock_cell = MagicMock()
        mock_cell.row = 2
        mock_worksheet.find.return_value = mock_cell

        # After delete, sheet returns fewer records
        mock_worksheet.get_all_records.return_value = SAMPLE_RECORDS[1:]

        db.delete_channel("CH1")

        mock_worksheet.find.assert_called_once_with("CH1", in_column=1)
        mock_worksheet.delete_rows.assert_called_once_with(2)
        assert len(db.channels) == 2

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_delete_nonexistent_raises(self, mock_gspread_module):
        """Deleting a nonexistent channel raises ValueError."""
        mock_gspread, _, mock_worksheet = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")
        mock_worksheet.find.return_value = None

        with pytest.raises(ValueError, match="not found"):
            db.delete_channel("NONEXISTENT")

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_update_channel(self, mock_gspread_module):
        """update_channel finds the row, updates cells, and reloads."""
        mock_gspread, _, mock_worksheet = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

        mock_cell = MagicMock()
        mock_cell.row = 3
        mock_worksheet.find.return_value = mock_cell

        db.update_channel("CH2", new_description="Updated desc", new_address="NEW_ADDR")

        mock_worksheet.find.assert_called_once_with("CH2", in_column=1)
        mock_worksheet.update_cell.assert_any_call(3, 2, "NEW_ADDR")
        mock_worksheet.update_cell.assert_any_call(3, 3, "Updated desc")

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_update_nonexistent_raises(self, mock_gspread_module):
        """Updating a nonexistent channel raises ValueError."""
        mock_gspread, _, mock_worksheet = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")
        mock_worksheet.find.return_value = None

        with pytest.raises(ValueError, match="not found"):
            db.update_channel("NONEXISTENT", new_description="foo")

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_refresh(self, mock_gspread_module):
        """refresh() reloads data from the sheet."""
        mock_gspread, _, mock_worksheet = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")
        assert len(db.channels) == 3

        # Simulate sheet data changing
        mock_worksheet.get_all_records.return_value = SAMPLE_RECORDS[:1]
        db.refresh()
        assert len(db.channels) == 1

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_worksheet_name_selection(self, mock_gspread_module):
        """worksheet parameter selects the correct tab."""
        mock_gspread, mock_spreadsheet, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id", worksheet="CustomTab")

        mock_spreadsheet.worksheet.assert_called_with("CustomTab")
        assert db.worksheet_name == "CustomTab"

    def test_missing_gspread_gives_clear_error(self):
        """ImportError with helpful message when gspread is not installed."""
        from osprey.services.channel_finder.databases import google_sheets as gs_module

        # Temporarily set gspread to None to simulate missing package
        original = gs_module.gspread
        gs_module.gspread = None
        try:
            with pytest.raises(ImportError, match="pip install osprey-framework\\[sheets\\]"):
                gs_module.GoogleSheetsChannelDatabase(spreadsheet_id="test_id")
        finally:
            gs_module.gspread = original

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_credentials_path_passed_to_service_account(self, mock_gspread_module):
        """credentials_path is forwarded to gspread.service_account(filename=...)."""
        mock_gspread, _, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        GoogleSheetsChannelDatabase(
            spreadsheet_id="test_id",
            credentials_path="/path/to/creds.json",
        )

        mock_gspread.service_account.assert_called_once_with(filename="/path/to/creds.json")

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_default_credentials(self, mock_gspread_module):
        """Without credentials_path, gspread.service_account() is called with no args."""
        mock_gspread, _, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

        mock_gspread.service_account.assert_called_once_with()

    # --- New tests for error paths ---

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_missing_columns_raises_database_load_error(self, mock_gspread_module):
        """Sheet with wrong columns raises DatabaseLoadError."""
        bad_records = [{"name": "x", "value": "y"}]
        mock_gspread, _, _ = _make_mock_gspread(records=bad_records)
        mock_gspread_module.service_account = mock_gspread.service_account

        with pytest.raises(DatabaseLoadError, match="missing required columns"):
            GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_spreadsheet_not_found_raises_database_load_error(self, mock_gspread_module):
        """Nonexistent spreadsheet raises DatabaseLoadError."""
        mock_gspread, _, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account
        mock_gspread_module.exceptions = MagicMock()

        # Make open_by_key raise SpreadsheetNotFound
        exc_class = type("SpreadsheetNotFound", (Exception,), {})
        mock_gspread_module.exceptions.SpreadsheetNotFound = exc_class
        mock_gspread.service_account.return_value.open_by_key.side_effect = exc_class("nope")

        with pytest.raises(DatabaseLoadError, match="Spreadsheet not found"):
            GoogleSheetsChannelDatabase(spreadsheet_id="bad_id")

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_worksheet_not_found_raises_database_load_error(self, mock_gspread_module):
        """Nonexistent worksheet raises DatabaseLoadError."""
        mock_gspread, mock_spreadsheet, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account
        mock_gspread_module.exceptions = MagicMock()

        # Make worksheet() raise WorksheetNotFound
        exc_class = type("WorksheetNotFound", (Exception,), {})
        mock_gspread_module.exceptions.WorksheetNotFound = exc_class
        mock_spreadsheet.worksheet.side_effect = exc_class("nope")

        with pytest.raises(DatabaseLoadError, match="Worksheet.*not found"):
            GoogleSheetsChannelDatabase(spreadsheet_id="test_id", worksheet="BadTab")

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_missing_credentials_raises_database_load_error(self, mock_gspread_module):
        """Missing credentials file raises DatabaseLoadError."""
        mock_gspread, _, _ = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        mock_gspread.service_account.side_effect = FileNotFoundError("no such file")

        with pytest.raises(DatabaseLoadError, match="credentials file not found"):
            GoogleSheetsChannelDatabase(
                spreadsheet_id="test_id",
                credentials_path="/nonexistent/creds.json",
            )

    @patch("osprey.services.channel_finder.databases.google_sheets.gspread")
    def test_update_channel_no_changes_is_noop(self, mock_gspread_module):
        """update_channel with no new_description or new_address makes no network calls."""
        mock_gspread, _, mock_worksheet = _make_mock_gspread()
        mock_gspread_module.service_account = mock_gspread.service_account

        db = GoogleSheetsChannelDatabase(spreadsheet_id="test_id")

        # Reset call counts after init
        mock_worksheet.find.reset_mock()
        mock_worksheet.update_cell.reset_mock()
        mock_worksheet.get_all_records.reset_mock()

        db.update_channel("CH1")

        mock_worksheet.find.assert_not_called()
        mock_worksheet.update_cell.assert_not_called()
        mock_worksheet.get_all_records.assert_not_called()
