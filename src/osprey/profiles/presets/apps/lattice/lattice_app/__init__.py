"""An OSPREY application: lattice analyses of the facility's decks, computed once and shared."""

#: The application's name, which is also its panel id and its tool server's name.
APP_NAME = "lattice"
#: The backend routes the tool server reaches through the panel door.
STATE_ROUTE = "/api/state"
RESULT_ROUTE = "/api/result"
EXPORT_ROUTE = "/api/export"
#: The longest a result request waits for a running analysis before it answers "running".
MAX_WAIT_S = 45.0
