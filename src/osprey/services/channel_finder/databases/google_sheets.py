"""
Google Sheets Channel Database

Manages a channel database stored in a Google Sheets spreadsheet,
enabling collaborative editing via Google Sheets instead of a local JSON file.
Subclasses FlatChannelDatabase to inherit chunking, formatting, and validation.
"""

try:
    import gspread
except ImportError:
    gspread = None

from ..core.exceptions import DatabaseLoadError
from .flat import ChannelDatabase as FlatChannelDatabase


class GoogleSheetsChannelDatabase(FlatChannelDatabase):
    """Channel database backed by a Google Sheets spreadsheet.

    Expects the spreadsheet to have columns: channel, address, description
    (as the first row / header).

    This class is not thread-safe. External synchronization is required
    for concurrent access.
    """

    def __init__(
        self, spreadsheet_id: str, worksheet: str | None = None, credentials_path: str | None = None
    ):
        """Initialize Google Sheets channel database.

        Args:
            spreadsheet_id: The Google Sheets spreadsheet ID
            worksheet: Worksheet/tab name (defaults to "Sheet1")
            credentials_path: Path to service account JSON credentials.
                If None, uses default ~/.config/gspread/service_account.json

        Raises:
            ImportError: If gspread is not installed
        """
        self.spreadsheet_id = spreadsheet_id
        self.worksheet_name = worksheet or "Sheet1"
        self.credentials_path = credentials_path

        # Initialize attributes that FlatChannelDatabase.__init__ would set
        self.channels: list[dict] = []
        self.channel_map: dict[str, dict] = {}
        # Synthetic db_path for logging and get_statistics
        self.db_path = f"google_sheets://{spreadsheet_id}"

        # Do NOT call super().__init__() — BaseDatabase.__init__ reads
        # self.db_path as a file path. We set attributes manually instead.
        self._init_client()
        self.load_database()

    def _init_client(self):
        """Initialize the gspread client and open the worksheet.

        Raises:
            ImportError: If gspread is not installed
            DatabaseLoadError: If credentials, spreadsheet, or worksheet not found
        """
        if gspread is None:
            raise ImportError(
                "gspread is required for Google Sheets database support. "
                "Install it with: pip install osprey-framework[sheets]"
            )

        try:
            if self.credentials_path:
                client = gspread.service_account(filename=self.credentials_path)
            else:
                client = gspread.service_account()
        except FileNotFoundError as e:
            path = self.credentials_path or "~/.config/gspread/service_account.json"
            raise DatabaseLoadError(f"Google Sheets credentials file not found: {path}") from e

        try:
            spreadsheet = client.open_by_key(self.spreadsheet_id)
        except gspread.exceptions.SpreadsheetNotFound as e:
            raise DatabaseLoadError(f"Spreadsheet not found: {self.spreadsheet_id}") from e

        try:
            self._sheet = spreadsheet.worksheet(self.worksheet_name)
        except gspread.exceptions.WorksheetNotFound as e:
            raise DatabaseLoadError(
                f"Worksheet '{self.worksheet_name}' not found in spreadsheet {self.spreadsheet_id}"
            ) from e

    def load_database(self):
        """Load channel data from the Google Sheet."""
        records = self._sheet.get_all_records()

        # Validate columns from first record
        required = {"channel", "address", "description"}
        if records:
            found = set(records[0].keys())
            missing = required - found
            if missing:
                raise DatabaseLoadError(
                    f"Sheet is missing required columns: {sorted(missing)}. "
                    f"Found columns: {sorted(found)}"
                )

        self.channels = [
            {
                "channel": row["channel"],
                "address": row["address"],
                "description": row["description"],
            }
            for row in records
        ]
        self.channel_map = {ch["channel"]: ch for ch in self.channels}

    def refresh(self):
        """Refresh channel data from the Google Sheet."""
        self.load_database()

    def add_channel(self, channel: str, address: str, description: str):
        """Add a new channel to the spreadsheet.

        Args:
            channel: Channel name
            address: Channel address
            description: Channel description

        Raises:
            ValueError: If channel already exists
        """
        if channel in self.channel_map:
            raise ValueError(f"Channel '{channel}' already exists")
        self._sheet.append_row([channel, address, description])
        self.load_database()

    def update_channel(
        self, channel: str, new_description: str | None = None, new_address: str | None = None
    ):
        """Update an existing channel in the spreadsheet.

        Args:
            channel: Channel name to update
            new_description: New description (if provided)
            new_address: New address (if provided)

        Raises:
            ValueError: If channel does not exist
        """
        if new_description is None and new_address is None:
            return
        cell = self._sheet.find(channel, in_column=1)
        if cell is None:
            raise ValueError(f"Channel '{channel}' not found")
        if new_address is not None:
            self._sheet.update_cell(cell.row, 2, new_address)
        if new_description is not None:
            self._sheet.update_cell(cell.row, 3, new_description)
        self.load_database()

    def delete_channel(self, channel: str):
        """Delete a channel from the spreadsheet.

        Args:
            channel: Channel name to delete

        Raises:
            ValueError: If channel does not exist
        """
        cell = self._sheet.find(channel, in_column=1)
        if cell is None:
            raise ValueError(f"Channel '{channel}' not found")
        self._sheet.delete_rows(cell.row)
        self.load_database()

    def get_statistics(self) -> dict:
        """Get database statistics including Google Sheets source info."""
        stats = super().get_statistics()
        stats.update(
            {
                "source": "google_sheets",
                "spreadsheet_id": self.spreadsheet_id,
                "worksheet": self.worksheet_name,
            }
        )
        return stats
